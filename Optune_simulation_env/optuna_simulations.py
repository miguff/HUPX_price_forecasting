import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from catboost import CatBoostRegressor
import wandb
from sklearn.ensemble import RandomForestRegressor
from .dnn_model import UniversalTorchWrapper

def get_best_params(
    ds: pd.DataFrame,   
    train_days_pool: np.ndarray,
    val_days: np.ndarray,
    n_trials: int,
    FEATURE_COLS : list,
    model_type : str
):
    
    #// It does not train every day, it does not add to the valid to the next train cycle
    ds_train_pool = ds[ds["day"].isin(train_days_pool)].copy()
    ds_val_pool = ds[ds["day"].isin(val_days)].copy()

    def objective(trial: optuna.trial.Trial):

        preds = []
        trues = []

        synth_weight = trial.suggest_float("synth_weight", 0.5, 1, log=True)
        retrain_every = trial.suggest_int("retrain_every", 1, 10)

        # --- All other single-stage models ---
        for i, D in enumerate(val_days):
            train_slice = ds_train_pool[ds_train_pool["day"] < D].copy()
            day_rows = ds_val_pool[ds_val_pool["day"] == D].copy()
            if train_slice.empty or day_rows.empty:
                continue

            w = np.where(train_slice["is_synthetic"].values == 1, synth_weight, 1.0).astype(float)

            if  i % retrain_every == 0:
                # Need a fresh model each retrain for sklearn pipelines

                model = get_model(model_type, trial, FEATURE_COLS)


                if hasattr(model, "fit"):
                    try:
                        model.fit(train_slice[FEATURE_COLS], train_slice["y_target"], sample_weight=w)
                    except TypeError:
                        model.fit(train_slice[FEATURE_COLS], train_slice["y_target"])
                else:
                    raise RuntimeError("Model has no fit().")
                fitted = model

            y_hat = fitted.predict(day_rows[FEATURE_COLS])
            preds.append(y_hat)
            trues.append(day_rows["y_target"].values)

        if not preds:
            return float("inf")
        
        y_true = np.concatenate(trues)
        y_hat = np.concatenate(preds)

        mae = mean_absolute_error(y_true, y_hat)
        wandb.log({"trial_mae": mae, "trial_number": trial.number})
        return mae
    
    study = optuna.create_study(direction="minimize", study_name="HUPX_test")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best overall value:", study.best_value)
    print("Best overall params:", study.best_params)
    wandb.log({"best_mae": study.best_value, "best_params": study.best_params})
    return study


def walk_forward_predict_test(
    ds,
    best_params: dict,
    train_days_pool: np.ndarray,   # days you allow for training (e.g., tune_days)
    test_days: np.ndarray,         # final_test_days
    feature_cols,
    model_type : str,
    target_col="y_target",
    day_col="day",
    synth_col="is_synthetic",
    
):
    test_days = np.sort(np.array(test_days))

    ds_train_pool = ds[ds[day_col].isin(train_days_pool)].copy()
    ds_test_pool  = ds[ds[day_col].isin(test_days)].copy()

    synth_weight = best_params["synth_weight"]

    # Build LGB params from Optuna best params (drop non-LGB keys)
    

    preds = []
    trues = []
    day_index = []
    row_index = []

    fitted = None

    retrain_every = int(best_params.get("retrain_every", 1))

    for i, D in enumerate(test_days):
        train_slice = ds_train_pool[ds_train_pool[day_col] < D].copy()
        day_rows = ds_test_pool[ds_test_pool[day_col] == D].copy()
        if train_slice.empty or day_rows.empty:
            continue

        # retrain schedule (same idea as your objective)
        if (i % retrain_every == 0) or (fitted is None):
            w = np.where(train_slice[synth_col].values == 1, synth_weight, 1.0).astype(float)


            #Ide kell a modellt behozni

            model, params = get_trained_model(model_type, best_params, feature_cols)
            model.fit(train_slice[feature_cols], train_slice[target_col], sample_weight=w)
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

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "days": days_out,
        "row_index": rows_out,
        "mae": mae,
        "last_model": fitted,  # the last fitted model (trained for last retrain point)
        "lgb_params": params,
        "synth_weight": synth_weight,
    }


def fit_final_model_before_test(
    ds,
    best_params: dict,
    train_days_pool: np.ndarray,
    first_test_day,
    feature_cols,
    target_col="y_target",
    day_col="day",
    synth_col="is_synthetic",
):
    ds_train_pool = ds[ds[day_col].isin(train_days_pool)].copy()
    train_slice = ds_train_pool[ds_train_pool[day_col] < first_test_day].copy()
    if train_slice.empty:
        raise RuntimeError("Training slice is empty before first_test_day.")

    synth_weight = best_params["synth_weight"]
    w = np.where(train_slice[synth_col].values == 1, synth_weight, 1.0).astype(float)

    lgb_params = {
        "objective": "regression",
        "n_estimators": best_params["lgb_n_estimators"],
        "learning_rate": best_params["lgb_lr"],
        "num_leaves": best_params["lgb_num_leaves"],
        "min_child_samples": best_params["lgb_min_child_samples"],
        "subsample": best_params["lgb_subsample"],
        "colsample_bytree": best_params["lgb_colsample"],
        "reg_alpha": best_params["lgb_reg_alpha"],
        "reg_lambda": best_params["lgb_reg_lambda"],
        "random_state": 42,
        "n_jobs": -1,
    }

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(train_slice[feature_cols], train_slice[target_col], sample_weight=w)
    return model


def get_model(model_value : str, trial, FEATURE_COLS: list):


    match model_value:
        case "lightgbm":
            model = lgb.LGBMRegressor(
                    objective="regression",
                    n_estimators=trial.suggest_int("lgb_n_estimators", 400, 2500),
                    learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.08, log=True),
                    num_leaves=trial.suggest_int("lgb_num_leaves", 16, 256, log=True),
                    min_child_samples=trial.suggest_int("lgb_min_child_samples", 10, 200),
                    subsample=trial.suggest_float("lgb_subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("lgb_colsample", 0.6, 1.0),
                    reg_alpha=trial.suggest_float("lgb_reg_alpha", 1e-8, 10.0, log=True),
                    reg_lambda=trial.suggest_float("lgb_reg_lambda", 1e-8, 10.0, log=True),
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            return model
        case "xgboost":
            model = xgb.XGBRegressor(
                    # objective
                    objective="reg:squarederror",
                    eval_metric="mae",

                    # core boosting params (roughly analogous to LGB)
                    n_estimators=trial.suggest_int("xgb_n_estimators", 400, 2500),
                    learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.08, log=True),

                    # tree shape / complexity
                    max_depth=trial.suggest_int("xgb_max_depth", 3, 12),
                    min_child_weight=trial.suggest_float("xgb_min_child_weight", 1e-2, 50.0, log=True),

                    # sampling (same names as LGB for these)
                    subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("xgb_colsample", 0.6, 1.0),

                    # regularization (same names as LGB)
                    reg_alpha=trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
                    reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True),

                    # optional knobs often worth tuning
                    gamma=trial.suggest_float("xgb_gamma", 0.0, 10.0),
                    max_delta_step=trial.suggest_int("xgb_max_delta_step", 0, 10),

                    random_state=42,
                    n_jobs=-1,

                    tree_method="hist",
                    verbosity=0,
                )
            return model
        case "catboost":
            model = CatBoostRegressor(
                    depth=trial.suggest_int('depth', 4, 8),
                    learning_rate=trial.suggest_float('learning_rate', 1e-3, 1, log=True),
                    iterations=trial.suggest_int('iterations', 100, 500),
                    l2_leaf_reg=trial.suggest_int('l2_leaf_reg', 1, 10),
                    silent=True,
                    objective='RMSE',
                    task_type='GPU',
                    boosting_type='Plain',
                )
            return model
        case "rf":
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_depth = trial.suggest_int('max_depth', 2, 32)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])


            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )
            return model
        case "dnn":
            current_input_dim = len(FEATURE_COLS)
            arch = trial.suggest_categorical("architecture", ["DNN", "LSTM", "GRU"])
    
            # 2. Optuna picks the depth (number of layers)
            n_layers = trial.suggest_int("n_layers", 1, 5) # 1 to 5 layers deep
            
            params = {
                "architecture": arch,
                "n_layers": n_layers,
                "h1": trial.suggest_int("h1", 16, 256), # Neurons per layer
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "epochs": trial.suggest_int("epoch", 5, 50), 
            }

            return UniversalTorchWrapper(model_type=arch, params=params, input_dim=current_input_dim)
        
def get_trained_model(model_value:  str, best_params, FEATURE_COLS:list):
    
    
    match model_value:
        case "lightgbm":
            params = {
                "objective": "regression",
                "n_estimators": best_params["lgb_n_estimators"],
                "learning_rate": best_params["lgb_lr"],
                "num_leaves": best_params["lgb_num_leaves"],
                "min_child_samples": best_params["lgb_min_child_samples"],
                "subsample": best_params["lgb_subsample"],
                "colsample_bytree": best_params["lgb_colsample"],
                "reg_alpha": best_params["lgb_reg_alpha"],
                "reg_lambda": best_params["lgb_reg_lambda"],
                "random_state": 42,
                "n_jobs": -1,
            }
            model = lgb.LGBMRegressor(**params)
            return model, params
        
        case "xgboost":
            # Build LGB params from Optuna best params (drop non-LGB keys)
            params = {
                "objective": "reg:squarederror",              # training loss (can be reg:absoluteerror too)
                "n_estimators": best_params["xgb_n_estimators"],
                "learning_rate": best_params["xgb_lr"],
                "max_depth": best_params["xgb_max_depth"],
                "min_child_weight": best_params["xgb_min_child_weight"],
                "subsample": best_params["xgb_subsample"],
                "colsample_bytree": best_params["xgb_colsample"],
                "reg_alpha": best_params["xgb_reg_alpha"],
                "reg_lambda": best_params["xgb_reg_lambda"],
                "gamma": best_params.get("xgb_gamma", 0.0),
                "max_delta_step": best_params.get("xgb_max_delta_step", 0),
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",                         # use "gpu_hist" if you have CUDA
                "verbosity": 0,
                # Optional: align evaluation metric with your reporting metric
                "eval_metric": "mae",
            }

            model = xgb.XGBRegressor(**params)
            return model, params
        case "catboost":
            params = {
                "depth" : best_params["depth"],
                "iterations" : best_params["iterations"],
                "l2_leaf_reg" : best_params["l2_leaf_reg"],
                "learning_rate" : best_params['learning_rate'],
                "objective": "RMSE",
                "random_state": 42,
                "silent" : True,
                "task_type" : "GPU",
                "boosting_type":'Plain',
            }

            model = CatBoostRegressor(**params)
            return model, params
        case "rf":
 
            params = {
                "n_estimators":best_params['n_estimators'],
                "max_depth" :best_params['max_depth'],
                "min_samples_split" : best_params['min_samples_split'],
                "min_samples_leaf": best_params['min_samples_leaf'],
                "max_features" : best_params['max_features'],
                "random_state" : 42,
                "n_jobs" : -1
            }

            model = RandomForestRegressor(**params)
            return model, params

        case "dnn":
            current_input_dim = len(FEATURE_COLS)
            arch = best_params['architecture']
            params = {
                "architecture": best_params['architecture'],
                "n_layers": best_params['n_layers'],
                "h1": best_params["h1"],
                "lr": best_params["lr"],
                "dropout": best_params["dropout"],
                "batch_size": best_params["batch_size"],
                "epochs": best_params["epoch"] 
            }

            model = UniversalTorchWrapper(model_type=arch, params=params, input_dim=current_input_dim)

            return model, params
    