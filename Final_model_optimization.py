# Final model optimization using nested LOOCV (outer LOO, inner 5-fold grid search)
# - Feature set: Hybrid-FSS (Corr-FSS ∪ LassoLarsCV-LOO)
# - Final performance reported across LOOCV folds (R2/RMSE/MAE)
# SHAP:
# - XGB: TreeExplainer
# - SVM: model-agnostic Explainer with Independent masker

import numpy as np
import pandas as pd

import shap
from sklearn.model_selection import LeaveOneOut, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LassoLarsCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ===================== USER SETTINGS =====================
DATA_PATH = "YOUR_DATA.xlsx"
SHEET_NAME = None
ID_COL = "ID"
TARGET_COL = "Biological endpoint"          # E-cad / Vim / α-SMA
DROP_COLS = []           

# Hybrid-FSS thresholds
R_TH = 0.3
COL_TH = 0.9

RANDOM_STATE = 42
INNER_SPLITS = 5

# ======= final XGB grid for E-cad & Vim =======
# fixed: subsample=0.8, colsample_bytree=0.8, tree_method=hist
# ranges: learning_rate=[0.005,0.01,0.05,0.1,0.2,0.3]
#         max_depth=[1,2,3,4]
#         min_child_weight=[2.5,3,3.5,4]
#         n_estimators=[100, 200, 300, 400, 500, 600, 700, 800]
#         reg_lambda=[2.0,2.5,3.0,3.5,4.0]
XGB_FIXED = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "reg:squarederror",
}
XGB_GRID = {
    "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    "max_depth": [1, 2, 3, 4],
    "min_child_weight": [2.5, 3, 3.5, 4],
    "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800],
    "reg_lambda": [2.0, 2.5, 3.0, 3.5, 4.0],
}

# ======= final SVM grid for α-SMA =======
SVM_GRID = {
    "C": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "gamma": [0.005, 0.01, 0.015, 0.02],
    "epsilon": [0.005, 0.01, 0.015, 0.02],
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


def hybrid_features_train_only(Xtr, ytr):
    a = set(corr_filter_train_only(Xtr, ytr, r_th=R_TH, col_th=COL_TH))
    b = set(lassolars_select_train_only(Xtr, ytr))
    return sorted(list(a.union(b)))


def pick_model():
    if TARGET_COL in ["E-cad", "Vim", "Vimentin"]:
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


def run_loocv():
    df = load_data()
    X, y, feats = get_Xy(df)

    est, scale, grid = pick_model()
    pipe = make_pipe(est, scale)

    loo = LeaveOneOut()
    inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    y_pred = np.zeros(len(y), dtype=float)
    n_feat = np.zeros(len(y), dtype=int)
    best_params_list = []

    for tr, te in loo.split(X):
        Xtr_full, Xte_full = X.iloc[tr], X.iloc[te]
        ytr = y.iloc[tr]

        feats_sel = hybrid_features_train_only(Xtr_full, ytr)
        if len(feats_sel) == 0:
            feats_sel = Xtr_full.columns.tolist()

        Xtr = Xtr_full[feats_sel]
        Xte = Xte_full[feats_sel]

        param_grid = {f"model__{k}": v for k, v in grid.items()}
        gs = GridSearchCV(
            pipe, param_grid=param_grid,
            cv=inner, scoring="neg_root_mean_squared_error",
            n_jobs=-1, refit=True
        )
        gs.fit(Xtr, ytr)

        y_pred[te[0]] = gs.predict(Xte)[0]
        n_feat[te[0]] = len(feats_sel)
        best_params_list.append(gs.best_params_)

    overall = metrics(y, y_pred)
    details = pd.DataFrame({
        "sample_idx": np.arange(len(y)),
        "y_true": y.values,
        "y_pred": y_pred,
        "abs_err": np.abs(y.values - y_pred),
        "n_feat": n_feat
    })

    bp = pd.DataFrame(best_params_list)
    param_freq = {c: bp[c].value_counts(dropna=False).to_dict() for c in bp.columns}

    print("\n Final LOOCV overall:", overall)
    details.to_csv(f"final_LOOCV_{TARGET_COL}.csv", index=False)
    print(f"Saved: final_LOOCV_{TARGET_COL}.csv")

    return X, y, details, overall, param_freq


def fit_final_model_on_full_data(X, y):
    """Fit a single final model on full data (for SHAP),
    using Hybrid-FSS on full dataset + 5-fold grid search."""
    est, scale, grid = pick_model()
    pipe = make_pipe(est, scale)

    feats_sel = sorted(list(
        set(corr_filter_train_only(X, y, r_th=R_TH, col_th=COL_TH))
        | set(lassolars_select_train_only(X, y))
    ))
    if len(feats_sel) == 0:
        feats_sel = X.columns.tolist()

    Xs = X[feats_sel].copy()

    inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {f"model__{k}": v for k, v in grid.items()}
    gs = GridSearchCV(
        pipe, param_grid=param_grid,
        cv=inner, scoring="neg_root_mean_squared_error",
        n_jobs=-1, refit=True
    )
    gs.fit(Xs, y)
    best_model = gs.best_estimator_
    return best_model, Xs, feats_sel


def run_shap(X, y):
    model, Xs, feats_sel = fit_final_model_on_full_data(X, y)

    if TARGET_COL in ["E-cad", "Vim"]:
        xgb = model.named_steps["model"]
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(Xs)
        shap_mat = np.array(shap_values)
    else:
        masker = shap.maskers.Independent(Xs, max_samples=200)
        explainer = shap.Explainer(model.predict, masker)
        exp = explainer(Xs)
        shap_mat = np.array(exp.values)

    mean_abs = np.mean(np.abs(shap_mat), axis=0)
    imp = pd.DataFrame({"feature": feats_sel, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(f"SHAP_importance_{TARGET_COL}.csv", index=False)

    shap_df = pd.DataFrame(shap_mat, columns=feats_sel)
    shap_df.insert(0, "y", y.values)
    shap_df.insert(0, "sample_idx", np.arange(len(y)))
    shap_df.to_csv(f"SHAP_values_{TARGET_COL}.csv", index=False)

    print(f"Saved:\n- SHAP_importance_{TARGET_COL}.csv\n- SHAP_values_{TARGET_COL}.csv")


if __name__ == "__main__":
    X, y, details, overall, param_freq = run_loocv()
    run_shap(X, y)
