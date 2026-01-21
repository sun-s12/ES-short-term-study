# Feature-set evaluation (No-FSS, Corr-FSS, ML-only-FSS, Hybrid-FSS)
# - Outer: repeated 5-fold CV (5 x 20 repeats; shared splits across FSS)
# - Inner: 5-fold grid search
# - Metrics per split: R2, RMSE, MAE; paired Wilcoxon vs No-FSS
# - Robustness index (RI):
#   RI = mean(R2)/(1+sd(R2)) + mean(RMSE)/(1+sd(RMSE)) + mean(MAE)/(1+sd(MAE))

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import LeaveOneOut

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ===================== USER SETTINGS =====================
DATA_PATH = "YOUR_DATA.xlsx"
SHEET_NAME = None
ID_COL = "ID"
TARGET_COL = "Biological endpoint"    
DROP_COLS = []

# Corr-FSS thresholds
R_TH = 0.3
COL_TH = 0.9

RANDOM_STATE = 42
OUTER_SPLITS = 5
REPEATS = 20
INNER_SPLITS = 5

# ======= XGB grid for E-cad & Vim (feature-set evaluation) =======
# fixed: learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method=hist
# ranges: max_depth=[3,4], min_child_weight=[1,3], n_estimators=[50,100,200,400,800], reg_lambda=[1.0,3.0]
XGB_FIXED = {
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "reg:squarederror",
}
XGB_GRID = {
    "max_depth": [3, 4],
    "min_child_weight": [1, 3],
    "n_estimators": [50, 100, 200, 400, 800],
    "reg_lambda": [1.0, 3.0],
}

# ======= SVM grid for α-SMA (feature-set evaluation) =======
SVM_GRID = {
    "C": [0.3, 1, 3, 10, 30, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
    "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
}
# =========================================================


def load_data():
    if DATA_PATH.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    return pd.read_csv(DATA_PATH)


def infer_feature_cols(df):
    drop = {ID_COL, TARGET_COL} | set(DROP_COLS)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in drop]


def get_Xy(df):
    feats = infer_feature_cols(df)
    X = df[feats].copy()
    y = df[TARGET_COL].copy()
    mask = ~y.isna()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), feats


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def corr_filter_train_only(Xtr, ytr, r_th=0.3, col_th=0.9):
    corrs = Xtr.apply(lambda c: c.corr(ytr), axis=0)
    keep = corrs.index[corrs.abs() >= r_th].tolist()
    if len(keep) <= 1:
        return keep

    Xk = Xtr[keep]
    C = Xk.corr().abs()
    upper = C.where(np.triu(np.ones(C.shape), k=1).astype(bool))

    relevance = corrs.abs().to_dict()
    to_drop = set()
    for col in upper.columns:
        high = upper[col][upper[col] > col_th].index.tolist()
        for row in high:
            if col in to_drop or row in to_drop:
                continue
            if relevance.get(col, 0) < relevance.get(row, 0):
                to_drop.add(col)
            else:
                to_drop.add(row)
    return [c for c in keep if c not in to_drop]


def lassolars_select_train_only(Xtr, ytr):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xtr.values)
    loo = LeaveOneOut()
    model = LassoLarsCV(cv=loo).fit(Xs, ytr.values)
    coefs = model.coef_.ravel()
    idx = np.where(np.abs(coefs) > 1e-12)[0]
    return [Xtr.columns[i] for i in idx]


def pick_model():
    # Selcted models after comparision: XGB for E-cad & Vim; SVM for α-SMA
    if TARGET_COL in ["E-cad", "Vim"]:
        est = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **XGB_FIXED)
        scale = False
        grid = XGB_GRID
    else:
        est = SVR(kernel="rbf")
        scale = True
        grid = SVM_GRID
    return est, scale, grid


def make_pipe(est, scale):
    if scale:
        return Pipeline([("scaler", StandardScaler()), ("model", est)])
    return Pipeline([("model", est)])


def ri_formula(perf_df: pd.DataFrame) -> float:
    """RI = mean(R2)/(1+sd_R2) + mean(RMSE)/(1+sd_RMSE) + mean(MAE)/(1+sd_MAE)"""
    vals = {}
    for k in ["R2", "RMSE", "MAE"]:
        x = perf_df[k].astype(float).values
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return np.nan
        vals[k] = (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)

    m_r2, sd_r2 = vals["R2"]
    m_rmse, sd_rmse = vals["RMSE"]
    m_mae, sd_mae = vals["MAE"]

    return (m_r2 / (1.0 + sd_r2)) + (m_rmse / (1.0 + sd_rmse)) + (m_mae / (1.0 + sd_mae))


def select_features(strategy, Xtr_full, ytr):
    if strategy == "No-FSS":
        return Xtr_full.columns.tolist()
    if strategy == "Corr-FSS":
        return corr_filter_train_only(Xtr_full, ytr, r_th=R_TH, col_th=COL_TH)
    if strategy == "ML-only-FSS":
        return lassolars_select_train_only(Xtr_full, ytr)
    if strategy == "Hybrid-FSS":
        a = set(corr_filter_train_only(Xtr_full, ytr, r_th=R_TH, col_th=COL_TH))
        b = set(lassolars_select_train_only(Xtr_full, ytr))
        return sorted(list(a.union(b)))
    raise ValueError(strategy)


def run():
    df = load_data()
    X, y, feats = get_Xy(df)

    est, scale, grid = pick_model()
    pipe = make_pipe(est, scale)

    outer = RepeatedKFold(n_splits=OUTER_SPLITS, n_repeats=REPEATS, random_state=RANDOM_STATE)
    inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    outer_splits = list(outer.split(X))  # shared across feature sets

    strategies = ["No-FSS", "Corr-FSS", "ML-only-FSS", "Hybrid-FSS"]
    rows = []

    for split_id, (tr, te) in enumerate(outer_splits):
        Xtr_full, Xte_full = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        for strat in strategies:
            feats_sel = select_features(strat, Xtr_full, ytr)
            if len(feats_sel) == 0:
                rows.append({"split": split_id, "FSS": strat, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n_feat": 0})
                continue

            Xtr = Xtr_full[feats_sel]
            Xte = Xte_full[feats_sel]

            param_grid = {f"model__{k}": v for k, v in grid.items()}
            gs = GridSearchCV(
                pipe, param_grid=param_grid,
                cv=inner, scoring="neg_root_mean_squared_error",
                n_jobs=-1, refit=True
            )
            gs.fit(Xtr, ytr)
            pred = gs.predict(Xte)
            m = metrics(yte, pred)
            rows.append({"split": split_id, "FSS": strat, **m, "n_feat": len(feats_sel)})

    perf_long = pd.DataFrame(rows)
    perf_long.to_csv(f"feature_set_evaluation_outer_metrics_{TARGET_COL}.csv", index=False)

    # summary mean±sd
    summary = perf_long.groupby("FSS")[["R2", "RMSE", "MAE", "n_feat"]].agg(["mean", "std"]).reset_index()

    # paired Wilcoxon vs No-FSS using shared splits
    base = perf_long[perf_long["FSS"] == "No-FSS"].set_index("split")
    tests = []
    for strat in ["Corr-FSS", "ML-only-FSS", "Hybrid-FSS"]:
        cur = perf_long[perf_long["FSS"] == strat].set_index("split")
        common = base.index.intersection(cur.index)

        x_rmse = base.loc[common, "RMSE"].values
        y_rmse = cur.loc[common, "RMSE"].values
        x_r2 = base.loc[common, "R2"].values
        y_r2 = cur.loc[common, "R2"].values

        mask_rmse = ~(np.isnan(x_rmse) | np.isnan(y_rmse))
        mask_r2 = ~(np.isnan(x_r2) | np.isnan(y_r2))

        # RMSE: expect strat < baseline => test baseline > strat
        p_rmse = wilcoxon(x_rmse[mask_rmse], y_rmse[mask_rmse], alternative="greater").pvalue if mask_rmse.sum() else np.nan
        # R2: expect strat > baseline => test baseline < strat
        p_r2 = wilcoxon(x_r2[mask_r2], y_r2[mask_r2], alternative="less").pvalue if mask_r2.sum() else np.nan

        # RI computed on the current strategy's outer-split metrics
        ri_val = ri_formula(cur.loc[common, ["R2", "RMSE", "MAE"]])
        tests.append({"FSS": strat, "p_Wilcoxon_RMSE_vs_NoFSS": float(p_rmse), "p_Wilcoxon_R2_vs_NoFSS": float(p_r2), "RI": float(ri_val)})

    tests_df = pd.DataFrame(tests)
    out = summary.merge(tests_df, on="FSS", how="left")
    out.to_csv(f"feature_set_evaluation_summary_{TARGET_COL}.csv", index=False)

    print(out)
    print(f"\nSaved:\n- feature_set_evaluation_outer_metrics_{TARGET_COL}.csv\n- feature_set_evaluation_summary_{TARGET_COL}.csv")


if __name__ == "__main__":
    run()
