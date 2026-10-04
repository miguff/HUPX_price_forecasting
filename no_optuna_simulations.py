import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from no_optuna_dnn_model import UniversalTorchWrapper
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_model_from_params(model_value: str, params: dict, feature_cols: list):
    # Remove custom structural keys before passing to model constructors
    model_params = {k: v for k, v in params.items() if k not in ["synth_weight", "retrain_every"]}
    
    if model_value == "lightgbm":
        return lgb.LGBMRegressor(**model_params)
    elif model_value == "xgboost":
        return xgb.XGBRegressor(**model_params)
    elif model_value == "catboost":
        return CatBoostRegressor(**model_params)
    elif model_value == "rf":
        return RandomForestRegressor(**model_params)
    elif model_value == "dnn":
        arch = params.get("architecture", "DNN")
        return UniversalTorchWrapper(model_type=arch, params=params, input_dim=len(feature_cols))
    else:
        raise ValueError(f"Unknown model: {model_value}")


def run_dnn_evaluation(ds, FEATURES, train_days, test_days, params, seed):
    set_seed(seed)

    ds = ds.sort_values("day", kind="stable").reset_index(drop=True)

    ds_train = ds[ds["day"].isin(train_days)].copy().sort_values("day", kind="stable")
    ds_test  = ds[ds["day"].isin(test_days)].copy().sort_values("day", kind="stable")

    X_train = ds_train[FEATURES].copy()
    y_train = ds_train["y_target"].copy()

    assert all(np.issubdtype(dt, np.number) for dt in X_train.dtypes), \
        f"Non-numeric columns in FEATURES: {X_train.dtypes}"

    model = get_model_from_params("dnn", params, FEATURES)
    model.fit(X_train, y_train)

    win = model.window_size

    if len(X_train) < win:
        raise RuntimeError("Not enough history for DNN window")

    context = X_train.tail(win)
    X_test_full = pd.concat([context, ds_test[FEATURES]], axis=0)

    preds = model.predict(X_test_full)
    y_true = ds_test["y_target"].values
    preds = preds[:len(y_true)]

    if len(preds) != len(y_true):
        raise RuntimeError(f"Prediction mismatch: {len(preds)} vs {len(y_true)}")

    rmse = root_mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)



    return {
        "mae": mae,
        "rmse": rmse,
        "y_true": y_true,
        "y_pred": preds
    }


def walk_forward_predict(
    ds,
    params: dict,
    train_days_pool: np.ndarray, 
    test_days: np.ndarray,         
    feature_cols,
    model_type: str,
    target_col="y_target",
    day_col="day",
    synth_col="is_synthetic"
):
    test_days = np.sort(np.array(test_days))

    ds_train_pool = ds[ds[day_col].isin(train_days_pool)].copy()
    ds_test_pool  = ds[ds[day_col].isin(test_days)].copy()

    synth_weight = params.get("synth_weight", 1.0)
    retrain_every = int(params.get("retrain_every", 1))

    preds = []
    trues = []
    day_index = []
    row_index = []

    fitted = None

    for i, D in enumerate(test_days):
        train_slice = ds_train_pool[ds_train_pool[day_col] < D].copy()
        day_rows = ds_test_pool[ds_test_pool[day_col] == D].copy()
        
        if train_slice.empty or day_rows.empty:
            continue

        if (i % retrain_every == 0) or (fitted is None):
            w = np.where(train_slice[synth_col].values == 1, synth_weight, 1.0).astype(float)
            model = get_model_from_params(model_type, params, feature_cols)
            
            if hasattr(model, "fit"):
                try:
                    model.fit(train_slice[feature_cols], train_slice[target_col], sample_weight=w)
                except TypeError:
                    model.fit(train_slice[feature_cols], train_slice[target_col])
            fitted = model

        y_hat = fitted.predict(day_rows[feature_cols])
        y_true = day_rows[target_col].values

        preds.append(y_hat)
        trues.append(y_true)
        day_index.append(np.full(len(day_rows), D))
        row_index.append(day_rows.index.values)

    if not preds:
        raise RuntimeError("No predictions were made on test_days. Check day filters / pools.")

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    days_out = np.concatenate(day_index)
    rows_out = np.concatenate(row_index)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)


    return {
        "mae": mae,
        "rmse": rmse,
        "y_pred": y_pred,
        "y_true": y_true,
        "days": days_out,
        "row_index": rows_out
    }