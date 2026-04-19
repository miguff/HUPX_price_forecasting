import pandas as pd
import optuna
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import shap
import wandb
from Optune_simulation_env import get_best_params, walk_forward_predict_test, get_trained_model
from utils import load_data
from scipy.stats import ttest_rel
from catboost import CatBoostRegressor
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Optuna pipeline")

    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--final_test_days", type=int, default=30)
    parser.add_argument("--optuna_val_days", type=int, default=30)
    parser.add_argument("--n_optuna_runs", type=int, default=17)
    parser.add_argument("--country", type=str, default="PL")
    parser.add_argument("--model", type=str, default="catboost")

    return parser.parse_args()



args = parse_args()

N_TRIALS = args.n_trials
FINAL_TEST_DAYS = args.final_test_days
OPTUNA_VAL_DAYS = args.optuna_val_days
N_Optuna_Runs = args.n_optuna_runs
COUNTRY = args.country
MODEL = args.model

def main(args):
    real_ds = load_data("real", COUNTRY)
    synt_ds_lgbm = load_data("lgbm", COUNTRY)
    synt_ds_spline = load_data("spline", COUNTRY)
    synt_ds_intra = load_data("intra", COUNTRY)

    results_real_rmse = []
    results_synth_lgbm_rmse = []
    results_synth_spline_rmse = []
    results_synth_intra_rmse = []
    results_real_mae = []
    results_synth_lgbm_mae = []
    results_synth_spline_mae = []
    results_synth_intra_mae = []
    all_best_params_real = []
    all_best_params_lgbm = []
    all_best_params_spline = []
    all_best_params_intra = []


    for i in range(N_Optuna_Runs):
        print(f"Run {i}")

        res_real = run_optuna_once(real_ds, MODEL, seed=i)
        res_synth_lgbm = run_optuna_once(synt_ds_lgbm, MODEL,seed=i)
        res_synth_spline = run_optuna_once(synt_ds_spline, MODEL,seed=i)
        res_synth_intra = run_optuna_once(synt_ds_intra, MODEL,seed=i)

        results_real_rmse.append(res_real["rmse"])
        results_synth_lgbm_rmse.append(res_synth_lgbm["rmse"])
        results_synth_spline_rmse.append(res_synth_spline["rmse"])
        results_synth_intra_rmse.append(res_synth_intra["rmse"])
        
        results_real_mae.append(res_real["mae"])
        results_synth_lgbm_mae.append(res_synth_lgbm["mae"])
        results_synth_spline_mae.append(res_synth_spline["mae"])
        results_synth_intra_mae.append(res_synth_intra["mae"])

        all_best_params_real.append(res_real["best_params"])
        all_best_params_lgbm.append(res_synth_lgbm["best_params"])
        all_best_params_spline.append(res_synth_spline["best_params"])
        all_best_params_intra.append(res_synth_intra["best_params"])

    real_rmse = np.array(results_real_rmse)
    lgbm_rmse = np.array(results_synth_lgbm_rmse)
    spline_rmse = np.array(results_synth_spline_rmse)
    intra_rmse = np.array(results_synth_intra_rmse)

    real_mae = np.array(results_real_mae)
    lgbm_mae = np.array(results_synth_lgbm_mae)
    spline_mae = np.array(results_synth_spline_mae)
    intra_mae = np.array(results_synth_intra_mae)

    results = {
        "real": {"rmse": real_rmse, "mae": real_mae},
        "synth_lgbm": {"rmse": lgbm_rmse, "mae": lgbm_mae},
        "synth_spline": {"rmse": spline_rmse, "mae": spline_mae},
        "synth_intra": {"rmse": intra_rmse, "mae": intra_mae}
    }

    #// Save the values
    #// RMSE
    np.save(f"outputs/{COUNTRY}/{MODEL}_real_rmse.npy", real_rmse)
    np.save(f"outputs/{COUNTRY}/{MODEL}_lgbm_rmse.npy", lgbm_rmse)
    np.save(f"outputs/{COUNTRY}/{MODEL}_spline_rmse.npy", spline_rmse)
    np.save(f"outputs/{COUNTRY}/{MODEL}_intra_rmse.npy", intra_rmse)

    #// MAE
    np.save(f"outputs/{COUNTRY}/{MODEL}_real_mae.npy", real_mae)
    np.save(f"outputs/{COUNTRY}/{MODEL}_lgbm_mae.npy", lgbm_mae)
    np.save(f"outputs/{COUNTRY}/{MODEL}_spline_mae.npy", spline_mae)
    np.save(f"outputs/{COUNTRY}/{MODEL}_intra_mae.npy", intra_mae)

    with open(f"outputs/{COUNTRY}/{MODEL}_best_params_real.json", "w") as f:
        json.dump(all_best_params_real, f, indent=2)

    with open(f"outputs/{COUNTRY}/{MODEL}_best_params_lgbm.json", "w") as f:
        json.dump(all_best_params_lgbm, f, indent=2)

    with open(f"outputs/{COUNTRY}/{MODEL}_best_params_spline.json", "w") as f:
        json.dump(all_best_params_spline, f, indent=2)

    with open(f"outputs/{COUNTRY}/{MODEL}_best_params_intra.json", "w") as f:
        json.dump(all_best_params_intra, f, indent=2)

    evaluate_metric(results, metric="rmse", plot=True)
    evaluate_metric(results, metric="mae", plot=True)


def confidence_interval(diff, alpha=0.05):
    mean = np.mean(diff)
    std = np.std(diff, ddof=1)
    n = len(diff)

    from scipy.stats import t
    t_val = t.ppf(1 - alpha/2, df=n-1)

    margin = t_val * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


def cohens_d(diff):
    return diff.mean() / diff.std(ddof=1)



def features():
    STATE_LAGS = [1, 4, 8, 24, 96, 192, 672]   # 15m, 1h, 2h, 6h, 1d, 2d, 1w
    STATE_ROLL_WINS = [24, 96, 672]            # rolling windows on past y (6h, 1d, 1w)
    
    # Feature columns

    STATE_FEATURES = (
        ["last_y"]
        + [f"lag_{L}_t0" for L in STATE_LAGS]
        + ["ramp_1h_t0", "ramp_6h_t0", "ramp_1d_t0"]
        + [f"roll_mean_{w}_t0" for w in STATE_ROLL_WINS]
        + [f"roll_std_{w}_t0" for w in STATE_ROLL_WINS]
    )

    HORIZON_FEATURES = [
        "h", "q_in_hour_target", "qod_target", "hod_target", "dow_target", "month_target", "is_weekend_target",
        "load_fc_target", "load_ramp_1h_target", "load_ramp_6h_target", "renewables_solar_fc","renewables_wind_fc",
        "load_day_mean", "load_day_max", "load_day_min", "q_in_hour_sin", "q_in_hour_cos", "qod_sin", "qod_cos", "hod_sin", "hod_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"
    ]

    WEIGHT_FEATURES = [
        'daily_weight_lag_1d',
        'daily_weight_lag_2d', 'daily_weight_lag_1w', 'hour_weight_lag_1d',
        'hour_weight_lag_2d', 'hour_weight_lag_1w', 'daily_avg_weight_lag_1d',
        'daily_avg_weight_lag_2d', 'daily_avg_weight_lag_1w',
        'hour_avg_weight_lag_1d', 'hour_avg_weight_lag_2d',
        'hour_avg_weight_lag_1w'
    ]

    FEATURE_COLS = STATE_FEATURES + HORIZON_FEATURES + WEIGHT_FEATURES

    return FEATURE_COLS


def run_optuna_once(ds: pd.DataFrame, model: str, seed: int):
    np.random.seed(seed)
    
    FEATURES = features()
    all_days = np.array(sorted(ds["day"].unique()))
    final_test_days = all_days[-FINAL_TEST_DAYS:]
    tune_days = all_days[:-FINAL_TEST_DAYS]

    optuna_val_days = tune_days[-OPTUNA_VAL_DAYS:]
    optuna_train_days_pool = all_days

    study = get_best_params(
        ds=ds,
        train_days_pool=optuna_train_days_pool,
        val_days=optuna_val_days,
        n_trials=N_TRIALS,
        FEATURE_COLS=FEATURES,
        model_type=model,
        seed=seed,  # IMPORTANT if supported
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



def plot_paper_results(real, methods, metric="RMSE"):
    
    names = list(methods.keys())
    arrays = list(methods.values())

    means = [real.mean()] + [m.mean() for m in arrays]

    # CI computation
    def ci(a):
        m = np.mean(a)
        se = np.std(a, ddof=1) / np.sqrt(len(a))
        ci = 1.96 * se
        return m, ci

    real_m, real_ci = ci(real)

    cis = [real_ci]
    for m in arrays:
        _, c = ci(m)
        cis.append(c)

    labels = ["Real"] + names

    # -----------------------
    # Mean + CI Bar Plot
    # -----------------------
    plt.figure(figsize=(6,4))

    x = np.arange(len(means))

    plt.bar(x, means, yerr=cis, capsize=5)

    plt.xticks(x, labels)
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison (Mean ±95% CI)")

    plt.tight_layout()
    plt.savefig(f"outputs/{COUNTRY}/{MODEL}_compare.svg", format="svg")
    plt.close()

    # -----------------------
    # Improvement Plot
    # -----------------------
    diffs = [m - real for m in arrays]

    plt.figure(figsize=(6,4))

    plt.boxplot(diffs)

    plt.axhline(0, linestyle="--")

    plt.xticks(range(1, len(names)+1), names)

    plt.ylabel(f"{metric} Difference vs Real")

    plt.title(f"Paired {metric} Improvements")

    plt.tight_layout()
    plt.savefig(f"outputs/{COUNTRY}/{MODEL}_improvement.svg", format="svg")
    plt.close()


def evaluate_metric(results: dict, metric="rmse", plot=True):
    """
    metric: 'rmse' or 'mae'
    """
    # Select correct arrays
    real = results["real"][metric]
    lgbm = results["synth_lgbm"][metric]
    spline = results["synth_spline"][metric]
    intra = results["synth_intra"][metric]

    methods = {
        "Synth LGBM": lgbm,
        "Spline": spline,
        "Intra": intra
    }

    rows = []

    for name, method in methods.items():
        diff = method - real

        mean_metric = method.mean()
        mean_diff, ci_low, ci_high = confidence_interval(diff)
        stat, p = ttest_rel(method, real)
        d = cohens_d(diff)

        rows.append({
            "method": name,
            f"{metric}_mean": mean_metric,
            "diff_mean": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p,
            "effect_size_d": d
        })

    df = pd.DataFrame(rows)

    # Multiple testing correction
    reject, pvals_corrected, _, _ = multipletests(df["p_value"], method='holm')
    df["p_corrected"] = pvals_corrected
    df["significant"] = reject

    # Improvement %
    df["improvement_%"] = 100 * (-df["diff_mean"] / real.mean())

    # Pretty display
    df_display = df.copy()
    df_display[f"{metric}_mean"] = df_display[f"{metric}_mean"].round(4)
    df_display["diff_mean"] = df_display["diff_mean"].round(4)
    df_display["ci_low"] = df_display["ci_low"].round(4)
    df_display["ci_high"] = df_display["ci_high"].round(4)
    df_display["p_value"] = df_display["p_value"].apply(lambda x: f"{x:.2e}")
    df_display["p_corrected"] = df_display["p_corrected"].apply(lambda x: f"{x:.2e}")
    df_display["effect_size_d"] = df_display["effect_size_d"].round(3)
    df_display["improvement_%"] = df_display["improvement_%"].round(2)

    print(f"\n===== {metric.upper()} RESULTS =====")
    df_display.to_csv(f"outputs/{COUNTRY}/{MODEL}_{metric}.csv", index=False)
    # ----------------
    # Plots
    # ----------------
    if plot:
        plot_paper_results(
        real,
        methods,
        metric=metric.upper()
    )

    return df



if __name__ == '__main__':
    args = parse_args()
    main(args)