import pandas as pd
import optuna
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import shap
import wandb
from Optune_simulation_env import get_best_params, walk_forward_predict_test
from utils import load_data


N_TRIALS = 30
FINAL_TEST_DAYS = 30
OPTUNA_VAL_DAYS = 30

def main():

    models = [
        "lightgbm",
        "catboost",
        "xgboost",
        "rf",
        'dnn'
    ]
    
    real_ds = load_data("real")
    synt_ds_lgbm = load_data("lgbm")
    synt_ds_spline = load_data("spline")
    synt_ds_intra = load_data("intra")

    #// Run optuna for synt
    for i in models:
        print("### Doing model: ", i, "For the Synth data only")
        run_uptuna(synt_ds_lgbm, i, "lightgbm")
        run_uptuna(real_ds, i, "real")
        run_uptuna(synt_ds_spline, i, "spliine")
        run_uptuna(synt_ds_intra, i, "intra")



def run_uptuna(ds: pd.DataFrame, model: str, synth_type :str, seed: int = 1):
    wandb.init(project="EnergyPrices_new", name=f"{model}_{synth_type}", reinit=True)
    np.random.seed(seed)
    
    FEATURES = features()
    all_days = np.array(sorted(ds["day"].unique()))
    final_test_days = all_days[-FINAL_TEST_DAYS:]
    tune_days = all_days[:-FINAL_TEST_DAYS]

    # Use the last part of tune_days as Optuna validation window (e.g., 21 days)
    
    optuna_val_days = tune_days[-OPTUNA_VAL_DAYS:]
    optuna_train_days_pool = all_days
    print("Optuna train pool:", optuna_train_days_pool[0], "→", optuna_train_days_pool[-1], len(optuna_train_days_pool))
    print("Optuna val days  :", optuna_val_days[0], "→", optuna_val_days[-1], len(optuna_val_days))
    print("Final test days  :", final_test_days[0], "→", final_test_days[-1], len(final_test_days))

    study = get_best_params(
        ds=ds,
        train_days_pool=optuna_train_days_pool,
        val_days=optuna_val_days,
        n_trials=N_TRIALS,
        FEATURE_COLS=FEATURES,
        model_type = model,
        seed=seed,  # IMPORTANT if supported
        study_name=f"HUPX_{model}_{synth_type}"
    )

    best_params = study.best_params
    test_res = walk_forward_predict_test(
                    ds=ds,
                    best_params=best_params,
                    train_days_pool=optuna_train_days_pool,        # important: allow training on ALL tune_days
                    test_days=final_test_days,
                    feature_cols=FEATURES,
                    model_type=model
                )
    y_true = test_res["y_true"]
    y_pred = test_res["y_pred"]
    rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
    smape_v = smape(y_true=y_true, y_pred=y_pred)

    print("#"*25)
    print(model)
    print("#"*25)
    print("Final test MAE:", test_res["mae"])
    print("Final test RMSE:", rmse)
    print("final test SMAPE:", smape_v)

    wandb.log({
        "final_mae": test_res["mae"],
        "final_rmse": rmse,
        "final_smape": smape_v
    })

    wandb.finish()


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


# calculate smape
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


if __name__ == "__main__":
    main()