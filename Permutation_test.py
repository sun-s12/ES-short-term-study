# Y-scrambling permutation test for final model (nested LOOCV + inner 5-fold grid search)
# Uses the exact same workflow/settings as the final models

import numpy as np
import pandas as pd

from Final_model_optimization import (
    load_data, get_Xy, pick_model, make_pipe,
    hybrid_features_train_only, metrics, TARGET_COL, RANDOM_STATE
)

from sklearn.model_selection import LeaveOneOut, KFold, GridSearchCV


# ===================== USER SETTINGS =====================
N_PERM = 200
SEED = 42
INNER_SPLITS = 5  # same as final model optimization
# =========================================================


def run_nested_loocv_once(X, y):
    est, scale, grid = pick_model()
    pipe = make_pipe(est, scale)

    loo = LeaveOneOut()
    inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    y_pred = np.zeros(len(y), dtype=float)

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

    return metrics(y, y_pred)


def main():
    rng = np.random.default_rng(SEED)

    df = load_data()
    X, y, feats = get_Xy(df)

    true_m = run_nested_loocv_once(X, y)
    rows = [{"perm": -1, "is_true": True, **true_m}]

    y0 = y.values.copy()
    for i in range(N_PERM):
        yp = y0.copy()
        rng.shuffle(yp)
        m = run_nested_loocv_once(X, pd.Series(yp))
        rows.append({"perm": i, "is_true": False, **m})

    out = pd.DataFrame(rows)
    out.to_csv(f"permutation_{TARGET_COL}.csv", index=False)
    print(f"Saved: permutation_{TARGET_COL}.csv")

    perm = out[~out["is_true"]]
    p_r2 = (np.sum(perm["R2"].values >= true_m["R2"]) + 1) / (len(perm) + 1)
    p_rmse = (np.sum(perm["RMSE"].values <= true_m["RMSE"]) + 1) / (len(perm) + 1)
    p_mae = (np.sum(perm["MAE"].values <= true_m["MAE"]) + 1) / (len(perm) + 1)

    print({"true": true_m, "p_R2": float(p_r2), "p_RMSE": float(p_rmse), "p_MAE": float(p_mae)})


if __name__ == "__main__":
    main()
