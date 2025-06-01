import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import pmdarima as pm
import warnings
def run_arima_case(train_df, test_df, input_features, target_feature,
                   max_p=5, max_q=5, d=None, 
                   seasonal=False, m=1,
                   suppress_warnings=True):
    
    print("Preparing data for ARIMA...")
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()
    y_train = train_df[target_feature]
    y_test = test_df[target_feature]
    
    X_train = train_df[input_features] if input_features else None
    X_test = test_df[input_features] if input_features else None

    if y_train.isnull().any():
        raise ValueError(f"NaNs found in training target '{target_feature}'. ARIMA cannot handle NaNs.")
    if X_train is not None and X_train.isnull().values.any():
         raise ValueError(f"NaNs found in training exogenous features {input_features}. ARIMA cannot handle NaNs.")
    if y_test.isnull().any():
        warnings.warn(f"NaNs found in test target '{target_feature}'. Evaluation metrics might be affected.")
    if X_test is not None and X_test.isnull().values.any():
         raise ValueError(f"NaNs found in test exogenous features {input_features} needed for forecasting. Cannot proceed.")

    print(f"Running auto_arima (max_p={max_p}, max_q={max_q}, d={d}, seasonal={seasonal}, m={m})...")
    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore") 
        try:
            auto_model = pm.auto_arima(
                y=y_train,
                exogenous=X_train,
                max_p=max_p,
                max_q=max_q,
                d=d, 
                seasonal=seasonal,
                m=m,
                stepwise=True,      
                suppress_warnings=suppress_warnings,
                error_action='ignore', 
                trace=True          
            )
        except np.linalg.LinAlgError as e:
             print(f"Linear Algebra Error during auto_arima (often due to perfect multicollinearity in exogenous features): {e}")
             return {"error": f"LinAlgError in auto_arima: {e}"}
        except Exception as e:
             print(f"An unexpected error occurred during auto_arima: {e}")
             return {"error": f"Exception in auto_arima: {e}"}

    print(f"Best ARIMA order found: {auto_model.order}, Seasonal order: {auto_model.seasonal_order}")
    print("Fitting the best model...")

    n_periods_to_forecast = len(test_df)
    print(f"Generating forecasts for {n_periods_to_forecast} periods...")

    requires_exog = X_train is not None
    
    if requires_exog and X_test is None:
         raise ValueError("Model requires exogenous features for forecasting, but none were provided in the test set.")
    if requires_exog and len(X_test) != n_periods_to_forecast:
        raise ValueError(f"Number of test exogenous samples ({len(X_test)}) does not match forecast horizon ({n_periods_to_forecast}).")

    try:
        y_pred, conf_int = auto_model.predict(
            n_periods=n_periods_to_forecast,
            exogenous=X_test if requires_exog else None,
            return_conf_int=True
        )
    except Exception as e:
         print(f"Error during ARIMA prediction: {e}")
         return {
                "error": f"Prediction error: {e}",
                "arima_order": str(auto_model.order),
                "seasonal_order": str(auto_model.seasonal_order)
         }

    print("Calculating metrics...")
    try:
        mse_val = mean_squared_error(y_test, y_pred)
        rmse_val = root_mean_squared_error(y_test, y_pred)
        mae_val = mean_absolute_error(y_test, y_pred)

        print(f"ARIMA Metrics (Original Scale): MSE={mse_val:.4f}, RMSE={rmse_val:.4f}, MAE={mae_val:.4f}")
        results = {
            "mse": float(mse_val),
            "rmse": float(rmse_val),
            "mae": float(mae_val),
            "arima_order": str(auto_model.order),
            "seasonal_order": str(auto_model.seasonal_order)
        }
    except ValueError as e:
        print(f"Error calculating metrics (likely due to NaNs or length mismatch): {e}")
        results = {
            "error": f"Metric calculation error: {e}",
            "arima_order": str(auto_model.order),
            "seasonal_order": str(auto_model.seasonal_order)
        }

    return results