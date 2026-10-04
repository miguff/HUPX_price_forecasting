import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

COUNTRY = "HU"
SYNTH_DATASETS = ["intra", "spline", "lgbm"]
ALL_DATASETS = ["real"] + SYNTH_DATASETS
MODELS_ALL = ["catboost", "lightgbm", "xgboost", "rf", "dnn"]
MODELS_LSTM = ["dnn"]  # Only LSTM (DNN)
STUDY_TYPES = [("Base (March)", ""), ("Ablation (April)", "ablation_")]
NUM_DAYS = 30
STEPS_PER_DAY = 96
LEGEND_FONTSIZE = 11


def load_preds_ensemble(model, dataset, study_prefix, country):
    pattern_pred = f"outputs/{country}/{model}_{study_prefix}run*_{dataset}_y_pred.npy"
    pattern_true = f"outputs/{country}/{model}_{study_prefix}run*_{dataset}_y_true.npy"
    files_pred = sorted(glob.glob(pattern_pred))
    files_true = sorted(glob.glob(pattern_true))
    if not files_pred or not files_true:
        return None, None
    preds = np.array([np.load(f) for f in files_pred])
    trues = np.array([np.load(f) for f in files_true])
    return np.mean(preds, axis=0), trues[0]


def compute_daily_metrics(y_true, y_pred):
    y_t = y_true[:NUM_DAYS * STEPS_PER_DAY].reshape(NUM_DAYS, STEPS_PER_DAY)
    y_p = y_pred[:NUM_DAYS * STEPS_PER_DAY].reshape(NUM_DAYS, STEPS_PER_DAY)
    daily_rmse = np.sqrt(np.mean((y_t - y_p) ** 2, axis=1))
    daily_mae = np.mean(np.abs(y_t - y_p), axis=1)
    return daily_rmse, daily_mae


hours = np.arange(STEPS_PER_DAY) / 4
colors_synth = {"intra": "steelblue", "spline": "firebrick", "lgbm": "forestgreen"}


def plot_subplot(ax, data, day_type, day_idx):
    """Helper to plot a single subplot"""
    if day_type == "best":
        title = f"Best Day {day_idx + 1} - Top-2 synth vs Real vs Actual"
    else:
        title = f"Real Best - Day {day_idx + 1} (MAE diff={data['diff_daily_mae'][day_idx]:.2f})"
    
    y_true_day = data["real_data"]["y_true"][day_idx * STEPS_PER_DAY:(day_idx + 1) * STEPS_PER_DAY]
    ax.plot(hours, y_true_day, label="Actual", color="dimgray", linewidth=2, linestyle="--")
    
    if day_type == "best":
        yr = data["real_data"]["y_pred"][day_idx * STEPS_PER_DAY:(day_idx + 1) * STEPS_PER_DAY]
        ax.plot(hours, yr, label=f"Real (MAE={data['real_daily_mae'][day_idx]:.2f})",
                color="black", linewidth=1.5)
        synth_ranking = []
        for r in data["all_results"]:
            for ds in SYNTH_DATASETS:
                if ds in r["data"]:
                    synth_ranking.append((r["data"][ds]["mean_mae"], r, ds))
        synth_ranking.sort(key=lambda x: x[0])
        top2 = synth_ranking[:2]
        top2_colors = ["steelblue", "firebrick"]
        for idx, (val, r, ds) in enumerate(top2):
            yp = r["data"][ds]["y_pred"][day_idx * STEPS_PER_DAY:(day_idx + 1) * STEPS_PER_DAY]
            dv = r["data"][ds]["daily_mae"][day_idx]
            ax.plot(hours, yp, label=f"{r['model']} + {ds} (MAE={dv:.2f})",
                    color=top2_colors[idx], linewidth=1.3, alpha=0.8)
    else:
        yr = data["real_data"]["y_pred"][day_idx * STEPS_PER_DAY:(day_idx + 1) * STEPS_PER_DAY]
        ys = data["synth_data"]["y_pred"][day_idx * STEPS_PER_DAY:(day_idx + 1) * STEPS_PER_DAY]
        ax.plot(hours, yr, label=f"Real (MAE={data['real_daily_mae'][day_idx]:.2f})",
                color="black", linewidth=1.5)
        ax.plot(hours, ys, label=f"{data['best_model']} + {data['best_synth_ds']} (MAE={data['synth_daily_mae'][day_idx]:.2f})",
                color=colors_synth.get(data['best_synth_ds'], "steelblue"), linewidth=1.5)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Price")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    
    # Add study title for first column
    if day_type == "best":
        ax.text(-0.15, 1.15, f"{data['study_name']} | {COUNTRY} | Best synth: {data['best_model']} + {data['best_synth_ds']}",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                ha="left", va="bottom")


def collect_plot_data(models, study_name, study_prefix):
    """Collect plot data for given models and study"""
    all_results = []
    for model in models:
        model_data = {}
        for ds in ALL_DATASETS:
            y_pred, y_true = load_preds_ensemble(model, ds, study_prefix, COUNTRY)
            if y_pred is None:
                continue
            daily_rmse, daily_mae = compute_daily_metrics(y_true, y_pred)
            model_data[ds] = {
                "daily_rmse": daily_rmse,
                "daily_mae": daily_mae,
                "mean_rmse": daily_rmse.mean(),
                "mean_mae": daily_mae.mean(),
                "y_pred": y_pred,
                "y_true": y_true,
            }
        if "real" in model_data:
            all_results.append({"model": model, "data": model_data})

    if not all_results:
        print(f"No data found for {study_name}.")
        return None

    # Find best synth by MAE
    best_entry = None
    best_synth_ds = None
    best_synth_val = float("inf")
    for r in all_results:
        for ds in SYNTH_DATASETS:
            if ds in r["data"]:
                if r["data"][ds]["mean_mae"] < best_synth_val:
                    best_synth_val = r["data"][ds]["mean_mae"]
                    best_entry = r
                    best_synth_ds = ds

    best_model = best_entry["model"]
    real_data = best_entry["data"]["real"]
    synth_data = best_entry["data"][best_synth_ds]

    real_daily_mae = real_data["daily_mae"]
    synth_daily_mae = synth_data["daily_mae"]
    diff_daily_mae = real_daily_mae - synth_daily_mae

    best_day = int(np.argmin(synth_daily_mae))
    worst_day = int(np.argmin(diff_daily_mae))

    print(f"[{study_name}] Best synth: {best_model} + {best_synth_ds}")
    print(f"  Best day: {best_day + 1}, Worst day: {worst_day + 1}")

    return {
        "study_name": study_name,
        "best_model": best_model,
        "best_synth_ds": best_synth_ds,
        "real_data": real_data,
        "synth_data": synth_data,
        "real_daily_mae": real_daily_mae,
        "synth_daily_mae": synth_daily_mae,
        "diff_daily_mae": diff_daily_mae,
        "best_day": best_day,
        "worst_day": worst_day,
        "all_results": all_results
    }


def generate_plots(plot_data, suffix):
    """Generate and save 2x2 and 1x4 plots"""
    # --- 2x2 Layout ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plot_subplot(axes[0, 0], plot_data[0], "best", plot_data[0]["best_day"])
    plot_subplot(axes[0, 1], plot_data[0], "worst", plot_data[0]["worst_day"])
    plot_subplot(axes[1, 0], plot_data[1], "best", plot_data[1]["best_day"])
    plot_subplot(axes[1, 1], plot_data[1], "worst", plot_data[1]["worst_day"])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"outputs/{COUNTRY}/viz_2x2_{suffix}.svg", format="svg", bbox_inches="tight")
    plt.show()

    # --- 1x4 Layout ---
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    plot_subplot(axes[0], plot_data[0], "best", plot_data[0]["best_day"])
    plot_subplot(axes[1], plot_data[0], "worst", plot_data[0]["worst_day"])
    plot_subplot(axes[2], plot_data[1], "best", plot_data[1]["best_day"])
    plot_subplot(axes[3], plot_data[1], "worst", plot_data[1]["worst_day"])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"outputs/{COUNTRY}/viz_1x4_{suffix}.svg", format="svg", bbox_inches="tight")
    plt.show()


# ============================================================================
# ALL MODELS (catboost, lightgbm, xgboost, rf, dnn)
# ============================================================================
print("=" * 60)
print("ALL MODELS (catboost, lightgbm, xgboost, rf, dnn)")
print("=" * 60)
plot_data_all = []
for study_name, study_prefix in STUDY_TYPES:
    data = collect_plot_data(MODELS_ALL, study_name, study_prefix)
    if data:
        plot_data_all.append(data)

if plot_data_all:
    generate_plots(plot_data_all, "combined")


# ============================================================================
# LSTM (DNN) ONLY
# ============================================================================
print("\n" + "=" * 60)
print("LSTM (DNN) ONLY")
print("=" * 60)
plot_data_lstm = []
for study_name, study_prefix in STUDY_TYPES:
    data = collect_plot_data(MODELS_LSTM, study_name, study_prefix)
    if data:
        plot_data_lstm.append(data)

if plot_data_lstm:
    generate_plots(plot_data_lstm, "lstm")
