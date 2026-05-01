import pandas as pd
import optuna
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import shap
import wandb
from Optune_simulation_env import (
    get_best_params,
    walk_forward_predict_test,
    get_trained_model,
    run_dnn_pipeline
)
from utils import load_data
from scipy.stats import ttest_rel
from catboost import CatBoostRegressor
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import json
import argparse
import os
import random
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# ARGPARSE
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run Optuna pipeline")

    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--final_test_days", type=int, default=30)
    parser.add_argument("--optuna_val_days", type=int, default=30)
    parser.add_argument("--country", type=str, default="PL")
    parser.add_argument("--model", type=str, default="catboost")

    return parser.parse_args()


args = parse_args()

N_TRIALS = args.n_trials
FINAL_TEST_DAYS = args.final_test_days
OPTUNA_VAL_DAYS = args.optuna_val_days
COUNTRY = args.country
set_seed(42)
MODEL = args.model


# ---------------------------
# MAIN
# ---------------------------
def main(args):

    # 👉 Slurm array ID = unique run
    RUN_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    seed = RUN_ID

    print(f"\n============================")
    print(f"RUN ID: {RUN_ID}")
    print(f"MODEL: {MODEL}")
    print(f"COUNTRY: {COUNTRY}")
    print(f"============================\n")

    real_ds = load_data("real", COUNTRY)
    synt_ds_lgbm = load_data("lgbm", COUNTRY)
    synt_ds_spline = load_data("spline", COUNTRY)
    synt_ds_intra = load_data("intra", COUNTRY)

    # ---------------------------
    # SINGLE RUN ONLY (no loop!)
    # ---------------------------
    res_real = run_optuna_once(real_ds, MODEL, seed)
    res_synth_lgbm = run_optuna_once(synt_ds_lgbm, MODEL, seed)
    res_synth_spline = run_optuna_once(synt_ds_spline, MODEL, seed)
    res_synth_intra = run_optuna_once(synt_ds_intra, MODEL, seed)

    # ---------------------------
    # SAVE RESULTS (run-specific!)
    # ---------------------------
    os.makedirs(f"outputs/{COUNTRY}", exist_ok=True)

    np.save(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_real_rmse.npy", res_real["rmse"])
    np.save(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_lgbm_rmse.npy", res_synth_lgbm["rmse"])
    np.save(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_spline_rmse.npy", res_synth_spline["rmse"])
    np.save(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_intra_rmse.npy", res_synth_intra["rmse"])

    np.save(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_real_mae.npy", res_real["mae"])
    np.save(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_lgbm_mae.npy", res_synth_lgbm["mae"])
    np.save(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_spline_mae.npy", res_synth_spline["mae"])
    np.save(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_intra_mae.npy", res_synth_intra["mae"])

    # save params
    with open(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_params_real.json", "w") as f:
        json.dump(res_real["best_params"], f, indent=2)

    with open(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_params_lgbm.json", "w") as f:
        json.dump(res_synth_lgbm["best_params"], f, indent=2)

    with open(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_params_spline.json", "w") as f:
        json.dump(res_synth_spline["best_params"], f, indent=2)

    with open(f"outputs/{COUNTRY}/{MODEL}_run{RUN_ID}_params_intra.json", "w") as f:
        json.dump(res_synth_intra["best_params"], f, indent=2)


# ---------------------------
# OPTUNA RUN
# ---------------------------
def run_optuna_once(ds: pd.DataFrame, model: str, seed: int):

    np.random.seed(seed)

    FEATURES = features()
    all_days = np.array(sorted(ds["day"].unique()))

    final_test_days = all_days[-FINAL_TEST_DAYS:]
    tune_days = all_days[:-FINAL_TEST_DAYS]

    optuna_val_days = tune_days[-OPTUNA_VAL_DAYS:]
    optuna_train_days = tune_days[:-OPTUNA_VAL_DAYS]

    # ==========================================
    # 🔥 BRANCH HERE
    # ==========================================
    if model == "dnn":

        return run_dnn_pipeline(
            ds,
            FEATURES,
            optuna_train_days,
            optuna_val_days,
            final_test_days,
            n_trials=N_TRIALS,
            seed = seed,
            study_name = f"EnergyPrice_{model}"
        )

    # ==========================================
    # 🌲 DEFAULT: WALK-FORWARD MODELS
    # ==========================================
    optuna_train_days_pool = all_days

    study = get_best_params(
        ds=ds,
        train_days_pool=optuna_train_days_pool,
        val_days=optuna_val_days,
        n_trials=N_TRIALS,
        FEATURE_COLS=FEATURES,
        model_type=model,
        seed=seed,
        study_name=f"EnergyPrice_{model}"
    )

    best_params = study.best_params
    best_params["seed"] = seed

    test_res = walk_forward_predict_test(
        ds=ds,
        best_params=best_params,
        train_days_pool=optuna_train_days_pool,
        test_days=final_test_days,
        feature_cols=FEATURES,
        model_type=model
    )

    y_true = test_res["y_true"]
    y_pred = test_res["y_pred"]

    rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)

    return {
        "mae": test_res["mae"],
        "rmse": rmse,
        "best_params": best_params
    }


# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
def features():

    STATE_LAGS = [1, 4, 8, 24, 96, 192, 672]
    STATE_ROLL_WINS = [24, 96, 672]

    STATE_FEATURES = (
        ["last_y"]
        + [f"lag_{L}_t0" for L in STATE_LAGS]
        + ["ramp_1h_t0", "ramp_6h_t0", "ramp_1d_t0"]
        + [f"roll_mean_{w}_t0" for w in STATE_ROLL_WINS]
        + [f"roll_std_{w}_t0" for w in STATE_ROLL_WINS]
    )

    HORIZON_FEATURES = [
        "h", "q_in_hour_target", "qod_target", "hod_target", "dow_target",
        "month_target", "is_weekend_target",
        "load_fc_target", "load_ramp_1h_target", "load_ramp_6h_target",
        "renewables_solar_fc", "renewables_wind_fc",
        "load_day_mean", "load_day_max", "load_day_min",
        "q_in_hour_sin", "q_in_hour_cos",
        "qod_sin", "qod_cos",
        "hod_sin", "hod_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos"
    ]

    WEIGHT_FEATURES = [
        'daily_weight_lag_1d', 'daily_weight_lag_2d', 'daily_weight_lag_1w',
        'hour_weight_lag_1d', 'hour_weight_lag_2d', 'hour_weight_lag_1w',
        'daily_avg_weight_lag_1d', 'daily_avg_weight_lag_2d',
        'daily_avg_weight_lag_1w',
        'hour_avg_weight_lag_1d', 'hour_avg_weight_lag_2d',
        'hour_avg_weight_lag_1w'
    ]

    return STATE_FEATURES + HORIZON_FEATURES + WEIGHT_FEATURES


# ---------------------------
if __name__ == "__main__":
    main(args)