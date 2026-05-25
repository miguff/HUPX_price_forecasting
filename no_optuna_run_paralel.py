import pandas as pd
import numpy as np
import os
import json
import argparse
import random
import torch

from no_optuna_simulations import walk_forward_predict, run_dnn_evaluation
from utils import load_data
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation Pipeline")
    parser.add_argument("--final_test_days", type=int, default=30)
    parser.add_argument("--country", type=str, default="HU")
    parser.add_argument("--model", type=str, default="catboost")
    return parser.parse_args()


args = parse_args()

FINAL_TEST_DAYS = args.final_test_days
COUNTRY = args.country
MODEL = args.model



def get_default_params(model_name: str, seed: int):
    """
    Define fixed hyperparameters, but pass the dynamic seed!
    """
    base_params = {"synth_weight": 1.0, "retrain_every": 1}
    
    if model_name == "lightgbm":
        model_params = {
            "objective": "regression", "n_estimators": 1000, "learning_rate": 0.05,
            "num_leaves": 31, "min_child_samples": 20, "subsample": 0.8,
            "colsample_bytree": 0.8, "random_state": seed, "n_jobs": -1 # <--- SEED HERE
        }
    elif model_name == "xgboost":
        model_params = {
            "objective": "reg:squarederror", "n_estimators": 1000, "learning_rate": 0.05,
            "max_depth": 6, "min_child_weight": 1.0, "subsample": 0.8,
            "colsample_bytree": 0.8, "random_state": seed, "n_jobs": -1, # <--- SEED HERE
            "tree_method": "hist", "device": "cuda", "eval_metric": "mae"
        }
    elif model_name == "catboost":
        model_params = {
            "depth": 6, "learning_rate": 0.05, "iterations": 500,
            "l2_leaf_reg": 3, "silent": True, "objective": "RMSE",
            "task_type": "GPU", "boosting_type": "Plain", "devices": "0",
            "random_seed": seed # <--- SEED HERE
        }
    elif model_name == "rf":
        model_params = {
            "n_estimators": 200, "max_depth": 10, "min_samples_split": 5,
            "min_samples_leaf": 2, "max_features": "sqrt", "random_state": seed, # <--- SEED HERE
            "n_jobs": -1
        }
    elif model_name == "dnn":
        model_params = {
            "architecture": "LSTM", "n_layers": 2, "h1": 64, "lr": 1e-3,
            "dropout": 0.1, "batch_size": 32, "epochs": 20, "window_size": 96
            # DNN handles seed globally via set_seed(), so no parameter change needed here
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return {**base_params, **model_params}


def main(args):
    #MODEL_library = ["catboost", "lightgbm", "rf", "xgboost"]
    MODEL_library = ["dnn"]
    for MODEL in MODEL_library:
        print(f"\n============================")
        print(f"MODEL: {MODEL}")
        print(f"COUNTRY: {COUNTRY}")
        print(f"============================\n")

        # 1. Load data ONCE outside the loop to save time
        print("Loading datasets...")
        datasets = {
            "real": load_data("real", COUNTRY),
            "lgbm": load_data("lgbm", COUNTRY),
            "spline": load_data("spline", COUNTRY),
            "intra": load_data("intra", COUNTRY)
        }

        # 2. Setup output directory
        os.makedirs(f"outputs/{COUNTRY}", exist_ok=True)
        
        TOTAL_RUNS = 15

        # 3. The 15-Run Loop
        for RUN_ID in range(TOTAL_RUNS):
            seed = RUN_ID  # The seed changes with every loop iteration (0 to 19)
            print(f"\n--- Starting RUN {RUN_ID} (Seed {seed}) ---")

            results = {}
            
            # Run evaluations for this specific seed
            pbar = tqdm(datasets.items(), total=len(datasets), desc=f"Run {RUN_ID} Initializing...")
            
            for ds_name, ds in pbar:
                pbar.set_description(f"Run {RUN_ID} | Evaluating: {ds_name.upper()}")
                # Pass the dynamically changing seed down to the evaluation function
                results[ds_name] = run_evaluation_once(ds, MODEL, seed)

            # 4. Save results for this specific run
            print(f"Saving results for RUN {RUN_ID}...")
            for ds_name, res in results.items():
                np.save(f"outputs/{COUNTRY}/{MODEL}_ablation_run{RUN_ID}_{ds_name}_rmse.npy", res["rmse"])
                np.save(f"outputs/{COUNTRY}/{MODEL}_ablation_run{RUN_ID}_{ds_name}_mae.npy", res["mae"])
                
                # Save predictions for Diebold-Mariano test
                np.save(f"outputs/{COUNTRY}/{MODEL}_ablation_run{RUN_ID}_{ds_name}_y_true.npy", res["y_true"])
                np.save(f"outputs/{COUNTRY}/{MODEL}_ablation_run{RUN_ID}_{ds_name}_y_pred.npy", res["y_pred"])

        # 5. Save the fixed hyperparameter configuration once at the end
        # (Passing 42 just to grab the dictionary structure; the actual seeds changed dynamically)
        params_used = get_default_params(MODEL, seed=42) 
        with open(f"outputs/{COUNTRY}/{MODEL}__ablation_base_params.json", "w") as f:
            json.dump(params_used, f, indent=2)
            
        print(f"\nAll {TOTAL_RUNS} runs completed successfully!")


def run_evaluation_once(ds: pd.DataFrame, model: str, seed: int):
    # 1. Set global environments (PyTorch, Numpy)
    set_seed(seed) 
    
    FEATURES = features()
    all_days = np.array(sorted(ds["day"].unique()))

    final_test_days = all_days[-FINAL_TEST_DAYS:]
    train_days_pool = all_days[:-FINAL_TEST_DAYS]
    
    # 2. Get params with the DYNAMIC seed
    params = get_default_params(model, seed) 

    if model == "dnn":
        return run_dnn_evaluation(ds, FEATURES, train_days_pool, final_test_days, params, seed)

    return walk_forward_predict(
        ds=ds,
        params=params,
        train_days_pool=train_days_pool,
        test_days=final_test_days,
        feature_cols=FEATURES,
        model_type=model
    )


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


if __name__ == "__main__":
    main(args)