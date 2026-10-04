"""Quick test: run DNN for base study, 1 seed, 1 epoch, check y_true alignment."""
import numpy as np
import glob
import pandas as pd
from no_optuna_simulations import run_dnn_evaluation
from no_optuna_run_paralel import features, get_default_params

COUNTRY = "HU"
DATASETS = ["real", "intra", "spline", "lgbm"]
SEED = 0
FINAL_TEST_DAYS = 30

BASE_CSV_MAP = {
    "real": f"processed_data/{COUNTRY}/Processed_data_real.csv",
    "lgbm": f"processed_data/{COUNTRY}/Processed_data_all.csv",
    "intra": f"processed_data/{COUNTRY}/Intra_Pattern_Processed_data_all.csv",
    "spline": f"processed_data/{COUNTRY}/Spline_Processed_data_all.csv",
}

# Override epochs to 1 for speed
params = get_default_params("dnn", SEED)
params["epochs"] = 1

FEATURES = features()

print("Loading base study datasets...")
datasets = {ds: pd.read_csv(path, index_col=0) for ds, path in BASE_CSV_MAP.items()}

# Load catboost base y_true for comparison
cb_true_files = sorted(glob.glob(f"outputs/{COUNTRY}/catboost_run0_real_y_true.npy"))
if cb_true_files:
    cb_true = np.load(cb_true_files[0])
    print(f"CatBoost base y_true loaded: shape={cb_true.shape}, first 5={cb_true[:5]}")
else:
    cb_true = None
    print("WARNING: No catboost base y_true found for comparison")

for ds_name, ds in datasets.items():
    print(f"\n--- Running DNN on {ds_name} (seed={SEED}, epochs=1) ---")
    all_days = np.array(sorted(ds["day"].unique()))
    final_test_days = all_days[-FINAL_TEST_DAYS:]
    train_days_pool = all_days[:-FINAL_TEST_DAYS]

    result = run_dnn_evaluation(ds, FEATURES, train_days_pool, final_test_days, params, SEED)

    y_true = result["y_true"]
    y_pred = result["y_pred"]
    print(f"  y_true shape={y_true.shape}, y_pred shape={y_pred.shape}")
    print(f"  y_true first 5: {y_true[:5]}")
    print(f"  y_pred first 5: {y_pred[:5]}")
    print(f"  MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")

    if ds_name == "real" and cb_true is not None:
        n_match = int(np.sum(y_true == cb_true))
        print(f"\n  ALIGNMENT CHECK vs CatBoost base y_true:")
        print(f"  Exact matches: {n_match}/{len(y_true)}")
        if n_match == len(y_true):
            print("  ✓ PERFECT ALIGNMENT - y_true values match exactly!")
        else:
            diff_idx = np.where(y_true != cb_true)[0]
            print(f"  ✗ MISMATCH at {len(diff_idx)} indices")
            print(f"    First 5 diff indices: {diff_idx[:5]}")
            for idx in diff_idx[:5]:
                print(f"    idx={idx} (hour={idx/4:.2f}): DNN={y_true[idx]:.2f} CB={cb_true[idx]:.2f}")

print("\nDone.")
