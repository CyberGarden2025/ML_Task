import argparse
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"RefNo": str})
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(how="all")
    primary = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    backup = pd.to_datetime(df.get("Date.1", df["Date"]), dayfirst=True, errors="coerce")
    df["Date"] = primary.fillna(backup)
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Withdrawal", "Deposit", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["NetAmount"] = df["Deposit"].fillna(0) - df["Withdrawal"].fillna(0)
    df["AbsAmount"] = df["Withdrawal"].fillna(0) + df["Deposit"].fillna(0)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.to_period("M")
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfMonth"] = df["Day"]
    df["CumNetAmount"] = df["NetAmount"].cumsum()
    df["Ref_Group"] = df["RefNo"].astype(str).str[:6]
    df["Category"] = df["Category"].fillna("Unknown").astype(str)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RefNo"] = df["RefNo"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["Ref_Group"] = df["RefNo"].str[:6]
    df["Ref_Prefix"] = df["RefNo"].str[:3]
    df["Ref_Length"] = df["RefNo"].str.len()
    df["Ref_Digit_Count"] = df["RefNo"].str.count(r"\d")
    df["SignedAmount"] = np.where(df["Withdrawal"] > 0, -df["Withdrawal"], df["Deposit"])
    df["AbsAmount"] = df["Withdrawal"].fillna(0) + df["Deposit"].fillna(0)
    df["NetAmount"] = df["Deposit"].fillna(0) - df["Withdrawal"].fillna(0)
    df["Amount"] = df["AbsAmount"]
    df["day"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["week_of_month"] = (df["day"] - 1) // 7 + 1
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = (df["day"] <= 5).astype(int)
    df["is_month_end"] = (df["day"] >= 25).astype(int)
    df["is_mid_month"] = ((df["day"] > 10) & (df["day"] < 20)).astype(int)
    df["is_day_1_3"] = (df["day"] <= 3).astype(int)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["Is_Deposit"] = (df["Deposit"] > 0).astype(int)
    df["Is_Withdrawal"] = (df["Withdrawal"] > 0).astype(int)
    df["Log_Amount"] = np.log1p(df["Amount"])
    df["Sqrt_Amount"] = np.sqrt(df["Amount"])
    df["Cbrt_Amount"] = np.cbrt(df["Amount"])
    df["Amount_Squared"] = df["Amount"] ** 2
    df["Log_Balance"] = np.log1p(df["Balance"])
    df["Amount_to_Balance"] = df["Amount"] / (df["Balance"] + 1)
    df["Log_Amount_to_Balance"] = np.log1p(df["Amount_to_Balance"])
    df["Amount_Bin"] = pd.cut(
        df["Amount"], bins=[-np.inf, 50, 150, 500, 2000, np.inf], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df["Signed_bin"] = pd.qcut(df["SignedAmount"], q=10, duplicates="drop")
    df["Abs_bin"] = pd.qcut(df["AbsAmount"], q=10, duplicates="drop")
    df["is_tiny"] = (df["Amount"] < 100).astype(int)
    df["is_small"] = ((df["Amount"] >= 100) & (df["Amount"] < 500)).astype(int)
    df["is_medium"] = ((df["Amount"] >= 500) & (df["Amount"] < 2000)).astype(int)
    df["is_large"] = ((df["Amount"] >= 2000) & (df["Amount"] < 5000)).astype(int)
    df["is_huge"] = (df["Amount"] >= 5000).astype(int)
    df["is_rent_range"] = ((df["Amount"] >= 3000) & (df["Amount"] <= 8000)).astype(int)
    df["is_rent_pattern"] = ((df["day"] <= 3) & (df["Amount"] >= 3000)).astype(int)
    df = df.sort_values("Date").reset_index(drop=True)
    df["Prev_Amount"] = df.groupby("Ref_Group")["Amount"].shift(1)
    df["Prev_Amount_2"] = df.groupby("Ref_Group")["Amount"].shift(2)
    df["Prev_Amount_3"] = df.groupby("Ref_Group")["Amount"].shift(3)
    df["Days_Since_Last"] = df.groupby("Ref_Group")["Date"].diff().dt.days
    df["Amount_Diff"] = df["Amount"] - df["Prev_Amount"]
    df["Amount_Pct_Change"] = df["Amount_Diff"] / (df["Prev_Amount"] + 1)
    for window in [3, 7, 14]:
        df[f"Rolling_Mean_{window}d"] = df.groupby("Ref_Group")["Amount"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f"Rolling_Std_{window}d"] = (
            df.groupby("Ref_Group")["Amount"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
            .fillna(0)
        )
    ref_stats = (
        df.groupby("Ref_Group")["Amount"]
        .agg(["mean", "std", "count", "min", "max", "median"])
        .reset_index()
    )
    ref_stats.columns = ["Ref_Group", "Ref_Mean", "Ref_Std", "Ref_Count", "Ref_Min", "Ref_Max", "Ref_Median"]
    df = df.merge(ref_stats, on="Ref_Group", how="left")
    df["Amount_vs_Ref_Mean"] = df["Amount"] / (df["Ref_Mean"] + 1)
    df["Amount_vs_Ref_Median"] = df["Amount"] / (df["Ref_Median"] + 1)
    df["Amount_Zscore"] = (df["Amount"] - df["Ref_Mean"]) / (df["Ref_Std"] + 1)
    df["Amount_Range_Position"] = (df["Amount"] - df["Ref_Min"]) / (df["Ref_Max"] - df["Ref_Min"] + 1)
    df["Is_Ref_Frequent"] = (df["Ref_Count"] > 5).astype(int)
    df["Tx_Index_In_Ref"] = df.groupby("Ref_Group").cumcount()
    df["Ref_prev_date"] = df.groupby("Ref_Group")["Date"].shift(1)
    df["Ref_inter_days"] = (df["Date"] - df["Ref_prev_date"]).dt.days
    df["Ref_inter_days"] = df["Ref_inter_days"].fillna(df["Ref_inter_days"].median())
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df


def split_with_rare_in_test(
    df: pd.DataFrame,
    target: str = "Category",
    rare_thr: int = 20,
    min_count_rare_train: int = 50,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    group_col: str = "Ref_Group",
):
    vc = df[target].value_counts()
    rare_classes = set(vc[vc < rare_thr].index)
    rare_df = df[df[target].isin(rare_classes)].copy()
    freq_df = df[~df[target].isin(rare_classes)].copy()

    if len(freq_df) > 0:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
        trainval_idx, test_idx = next(gss_test.split(freq_df, freq_df[target], groups=freq_df[group_col]))
        freq_trainval = freq_df.iloc[trainval_idx]
        freq_test = freq_df.iloc[test_idx]

        remain_frac = 1 - test_frac
        val_size_rel = val_frac / remain_frac if remain_frac > 0 else 0.2
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_rel, random_state=42)
        train_idx, val_idx = next(gss_val.split(freq_trainval, freq_trainval[target], groups=freq_trainval[group_col]))
        freq_train = freq_trainval.iloc[train_idx]
        freq_val = freq_trainval.iloc[val_idx]
    else:
        freq_train = freq_val = freq_test = freq_df

    test_parts, val_parts = [], []
    max_split = 20
    for cls, cnt in rare_df[target].value_counts().items():
        subset = rare_df[rare_df[target] == cls]
        val_count = min(max_split, int(np.ceil(cnt * 0.5)))
        test_count = min(max_split, cnt - val_count)
        val_part = subset.sample(n=val_count, random_state=42) if val_count > 0 else subset.iloc[0:0]
        remain = subset.drop(val_part.index)
        test_part = remain.sample(n=test_count, random_state=42) if test_count > 0 else remain
        val_parts.append(val_part)
        test_parts.append(test_part)
    val_rare = pd.concat(val_parts, ignore_index=True) if val_parts else rare_df.iloc[0:0]
    test_rare = pd.concat(test_parts, ignore_index=True) if test_parts else rare_df.iloc[0:0]

    train_rare_syn = []
    for cls, cnt in rare_df[target].value_counts().items():
        need = max(0, min_count_rare_train - cnt)
        if need > 0:
            add = rare_df[rare_df[target] == cls].sample(n=need, replace=True, random_state=42)
            train_rare_syn.append(add)
    train_rare_syn = pd.concat(train_rare_syn, ignore_index=True) if train_rare_syn else rare_df.iloc[0:0]

    train_df = pd.concat([freq_train, train_rare_syn], ignore_index=True).sample(frac=1, random_state=42)
    val_df = pd.concat([freq_val, val_rare], ignore_index=True).sample(frac=1, random_state=42)
    test_df = pd.concat([freq_test, test_rare], ignore_index=True).sample(frac=1, random_state=42)
    return train_df, val_df, test_df


def report(name: str, y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1).astype(str)
    y_pred = np.asarray(y_pred).reshape(-1).astype(str)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("Confusion:\n", confusion_matrix(y_true, y_pred, labels=labels))
    print("F1 macro:", f1_score(y_true, y_pred, average="macro"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="TransFiles_3rzk6/ci_data.csv")
    parser.add_argument("--rare-thr", type=int, default=20)
    parser.add_argument("--min-rare-train", type=int, default=50)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--neighbors", type=int, default=7)
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Optuna tuning and use preset CatBoost params")
    args = parser.parse_args()

    df_raw = load_raw(args.data)
    df_feat = engineer_features(df_raw)
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]

    train_df, val_df, test_df = split_with_rare_in_test(
        df_feat,
        "Category",
        rare_thr=args.rare_thr,
        min_count_rare_train=args.min_rare_train,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        group_col="Ref_Group",
    )
    for df_tmp in (train_df, val_df, test_df):
        df_tmp["Category"] = df_tmp["Category"].astype(str)

    cat_cols = ["Ref_Group", "Ref_Prefix", "Signed_bin", "Abs_bin", "Amount_Bin"]
    for c in cat_cols:
        for df_tmp in (train_df, val_df, test_df):
            df_tmp[c] = df_tmp[c].astype(str)
    num_cols = [
        c for c in train_df.columns if c not in ["Category"] + cat_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    X_train = train_df[num_cols + cat_cols]
    y_train = train_df["Category"]
    X_val = val_df[num_cols + cat_cols]
    y_val = val_df["Category"]
    X_test = test_df[num_cols + cat_cols]
    y_test = test_df["Category"]

    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )
    knn_pipe = make_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=args.neighbors))
    knn_pipe.fit(X_train, y_train)
    report("KNN Val", y_val, knn_pipe.predict(X_val))
    report("KNN Test", y_test, knn_pipe.predict(X_test))

    train_cb = train_df.copy()
    val_cb = val_df.copy()
    test_cb = test_df.copy()
    feat_cols_cb = num_cols + cat_cols
    cat_idx = [feat_cols_cb.index(c) for c in cat_cols]

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 400, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "depth": trial.suggest_int("depth", 6, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "loss_function": "MultiClass",
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": False,
        }
        model = CatBoostClassifier(**params)
        train_pool = Pool(train_cb[feat_cols_cb], train_cb["Category"], cat_features=cat_idx)
        val_pool = Pool(val_cb[feat_cols_cb], val_cb["Category"], cat_features=cat_idx)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        pred = model.predict(val_cb[feat_cols_cb])
        return 1 - f1_score(val_cb["Category"].astype(str), pred.astype(str), average="macro")

    if args.skip_optuna:
        best_params = {
            "iterations": 500,
            "learning_rate": 0.07,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "loss_function": "MultiClass",
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": 200,
        }
    else:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params.update(
            {"loss_function": "MultiClass", "auto_class_weights": "Balanced", "random_seed": 42, "verbose": 200}
        )
        print("Best params:", best_params)

    model = CatBoostClassifier(**best_params)
    train_pool = Pool(train_cb[feat_cols_cb], train_cb["Category"], cat_features=cat_idx)
    val_pool = Pool(val_cb[feat_cols_cb], val_cb["Category"], cat_features=cat_idx)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    report("CatBoost Val", val_cb["Category"], model.predict(val_cb[feat_cols_cb]))
    report("CatBoost Test", test_cb["Category"], model.predict(test_cb[feat_cols_cb]))

    Path("outputs").mkdir(exist_ok=True, parents=True)
    joblib.dump(knn_pipe, "outputs/tx_category_knn.pkl")
    model.save_model("outputs/tx_category_catboost_full.cbm")
    print("Saved KNN to outputs/tx_category_knn.pkl")
    print("Saved CatBoost to outputs/tx_category_catboost_full.cbm")

    sample = test_df.sample(1, random_state=42)
    sample_knn_pred = knn_pipe.predict(sample[num_cols + cat_cols])
    sample_cb_pred = model.predict(sample[feat_cols_cb])
    print(
        "\nSample true:",
        sample["Category"].values[0],
        "KNN pred:",
        sample_knn_pred[0],
        "CatBoost pred:",
        sample_cb_pred[0],
    )


if __name__ == "__main__":
    main()

