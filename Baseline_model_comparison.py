# Baseline benchmarking of 8 regressors with full feature set
# - Outer: 5-fold (shared splits across algorithms)
# - Inner: 3-fold grid search (light tuning)
# - Metrics: R2, RMSE, MAE on out-of-fold predictions

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ===================== USER SETTINGS =====================
DATA_PATH = "YOUR_DATA.xlsx"   # or csv
SHEET_NAME = None             # set sheet name if needed
ID_COL = "ID"
TARGET_COL = "Biological endpoint"          
DROP_COLS = []                # optional extra non-feature cols

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
# =========================================================


def load_data():
    if DATA_PATH.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def infer_feature_cols(df: pd.DataFrame):
    drop = {ID_COL, TARGET_COL} | set(DROP_COLS)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in drop]
    return feats


def get_Xy(df: pd.DataFrame):
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


def build_models():
    return {
        "Ridge": (Ridge(), True),
        "EN": (ElasticNet(max_iter=20000), True),
        "PLSR": (PLSRegression(), True),
        "SVM": (SVR(kernel="rbf"), True),
        "GPR": (GaussianProcessRegressor(), True),
        "KNN": (KNeighborsRegressor(), True),
        "RF": (RandomForestRegressor(random_state=RANDOM_STATE), False),
        "XGB": (XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1
        ), False),
    }


def baseline_grids():
    # light tuning grids (small)
    return {
        "Ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "EN": {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        "PLSR": {"n_components": [2, 3, 4, 5, 6]},
        "SVM": {"C": [1, 10, 100], "gamma": ["scale", 0.1, 0.01], "epsilon": [0.01, 0.1, 0.2]},
        "GPR": {"alpha": [1e-10, 1e-6, 1e-3]},
        "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
        "RF": {"n_estimators": [300, 800], "max_depth": [None, 3, 5]},
        "XGB": {"n_estimators": [300, 800], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 4]},
    }


def make_pipeline(estimator, scale: bool):
    if scale:
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    return Pipeline([("model", estimator)])


def run():
    df = load_data()
    X, y, feats = get_Xy(df)

    outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    outer_splits = list(outer.split(X))  
    grids = baseline_grids()
    models = build_models()

    rows = []
    for name, (est, scale) in models.items():
        pipe = make_pipeline(est, scale)
        param_grid = {f"model__{k}": v for k, v in grids[name].items()}
        y_oof = np.zeros(len(y), dtype=float)

        for tr, te in outer_splits:
            gs = GridSearchCV(
                pipe, param_grid=param_grid,
                cv=inner, scoring="neg_root_mean_squared_error",
                n_jobs=-1, refit=True
            )
            gs.fit(X.iloc[tr], y.iloc[tr])
            y_oof[te] = gs.predict(X.iloc[te])

        m = metrics(y, y_oof)
        rows.append({"Algorithm": name, **m})

    out = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    print(out)
    out.to_csv(f"baseline_{TARGET_COL}.csv", index=False)
    print(f"\nSaved: baseline_{TARGET_COL}.csv")


if __name__ == "__main__":
    run()
