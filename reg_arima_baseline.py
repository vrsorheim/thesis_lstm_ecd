# reg_arima_baseline.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import warnings
import traceback

def run_regression_arima_errors_case(train_df, test_df, input_features, target_feature,
                                     regression_formula=None, 
                                     poly_degree=2, 
                                     arima_max_p=5, arima_max_q=5, arima_d=None, 
                                     seasonal=False, m=1,
                                     arima_residuals_subsample=100000,
                                     suppress_warnings=True):
    print("Preparing data for Regression with ARIMA Errors...")

    try:
        train_df = train_df.sort_index()
        test_df = test_df.sort_index()
    except Exception as e:
        print(f"Error sorting indices: {e}")
        return {"error": f"Error sorting indices: {e}"}

    required_cols = [target_feature] + input_features
    if not all(col in train_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in train_df.columns]
        print(f"Error: Missing required columns in train_df: {missing}")
        return {"error": f"Missing required columns in train_df: {missing}"}
    if not all(col in test_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in test_df.columns]
        print(f"Error: Missing required columns in test_df: {missing}")
        return {"error": f"Missing required columns in test_df: {missing}"}

    y_train = train_df[target_feature]
    y_test = test_df[target_feature]
    train_reg_df = train_df[required_cols].copy()
    test_reg_df = test_df[required_cols].copy()

    formula_terms = [] 

    if regression_formula is None:
        print(f"Generating Polynomial Features (degree={poly_degree}) for {input_features}")
        if poly_degree < 1:
             print("Warning: poly_degree is less than 1. Setting to 1.")
             poly_degree = 1

        for feat in input_features:
            formula_terms.append(feat)
            if poly_degree > 1:
                try:
                    num_unique = train_reg_df[feat].nunique()
                    if num_unique > 2: 
                        for degree in range(2, poly_degree + 1):
                            col_name = f"{feat}_pow{degree}"
                            try:
                                train_reg_df[col_name] = np.power(train_reg_df[feat].astype(float), degree)
                                test_reg_df[col_name] = np.power(test_reg_df[feat].astype(float), degree)
                                if np.isinf(train_reg_df[col_name]).any() or np.isinf(test_reg_df[col_name]).any():
                                     warnings.warn(f"Infinite values generated for {col_name}. Check input scale or reduce poly_degree.", UserWarning)
                                formula_terms.append(col_name) 
                            except OverflowError:
                                 warnings.warn(f"OverflowError generating polynomial feature {col_name}. Skipping this term.", UserWarning)
                            except Exception as e_pow:
                                 warnings.warn(f"Error generating polynomial feature {col_name}: {e_pow}. Skipping this term.", UserWarning)
                    else:
                        print(f"Skipping polynomial terms (degree > 1) for feature '{feat}' as it has {num_unique} unique values in training data.")
                except KeyError:
                     warnings.warn(f"Feature '{feat}' not found during polynomial generation. Skipping.", UserWarning)
                except Exception as e_unique:
                     warnings.warn(f"Error checking uniqueness or generating poly terms for '{feat}': {e_unique}. Skipping poly terms.", UserWarning)
        if not formula_terms:
             print("Error: No terms generated for the regression formula.")
             return {"error": "No terms generated for the regression formula."}
        regression_formula = f"{target_feature} ~ {' + '.join(formula_terms)}"
        print(f"Using generated formula: {regression_formula}")

    else:
        print(f"Using provided formula: {regression_formula}")
        try:
             parsed_terms = [term.strip() for term in regression_formula.split('~')[1].split('+')]
             required_formula_cols = [t for t in parsed_terms if t and t != '1' and not t.startswith(('np.', 'I(', 'C(')) and ':' not in t and '*' not in t]
             missing_formula_cols = [col for col in required_formula_cols if col not in train_reg_df.columns]
             if missing_formula_cols:
                 warnings.warn(f"Columns derived from provided formula might be missing in train_reg_df: {missing_formula_cols}", UserWarning)
             formula_terms = required_formula_cols 
        except Exception as e_parse:
             warnings.warn(f"Could not parse provided formula for column check: {e_parse}", UserWarning)
             formula_terms = input_features

    print("\n--- Diagnostics before OLS ---")
    print(f"Shape of train_reg_df: {train_reg_df.shape}")
    print(f"Columns available in train_reg_df: {train_reg_df.columns.tolist()}")
    print("NaN check in train_reg_df (target + all available features):")
    nan_counts = train_reg_df.isnull().sum()
    if nan_counts.sum() == 0:
        print("None")
    else:
        print(nan_counts[nan_counts > 0]) 
    print("\nInf check in train_reg_df:")
    try:
        numeric_cols = train_reg_df.select_dtypes(include=np.number).columns
        inf_counts = train_reg_df[numeric_cols].apply(lambda x: np.isinf(x).sum())
        if inf_counts.sum() == 0:
            print("None")
        else:
            print(inf_counts[inf_counts > 0]) 
    except Exception as e_inf_check:
        print(f"Could not perform Inf check: {e_inf_check}")
    print("-----------------------------\n")

    print("Fitting regression model...")
    ols_model = None
    try:
        ols_check_terms = []
        for term in formula_terms: 
             if term in train_reg_df.columns:
                  ols_check_terms.append(term)
        for feat in input_features:
             if poly_degree > 1:
                  for degree in range(2, poly_degree + 1):
                       col_name = f"{feat}_pow{degree}"
                       if col_name in train_reg_df.columns:
                           ols_check_terms.append(col_name)
        cols_for_ols_check = list(dict.fromkeys([target_feature] + ols_check_terms))
        ols_data_check = train_reg_df[cols_for_ols_check]
        if ols_data_check.isnull().values.any():
             warnings.warn("NaNs detected in data subset for OLS formula just before fitting. Using missing='drop'.", UserWarning)
             print("NaN counts per column for OLS (before drop):")
             print(ols_data_check.isnull().sum()[ols_data_check.isnull().sum() > 0])
        if np.isinf(ols_data_check.select_dtypes(include=np.number)).values.any():
             warnings.warn("Infinities detected in data subset for OLS formula just before fitting. OLS might fail.", UserWarning)
             print("Infinity counts per column for OLS:")
             numeric_ols_cols = ols_data_check.select_dtypes(include=np.number).columns
             print(ols_data_check[numeric_ols_cols].apply(lambda x: np.isinf(x).sum())[ols_data_check[numeric_ols_cols].apply(lambda x: np.isinf(x).sum()) > 0])
        with warnings.catch_warnings():
             if suppress_warnings:
                 warnings.simplefilter("ignore")
             ols_model = smf.ols(formula=regression_formula, data=train_reg_df, missing='drop').fit()
        print("\nOLS Model Summary:")
        print(ols_model.summary())
        print("-" * 30)

    except Exception as e:
        print(f"Error fitting regression model: {e}")
        traceback.print_exc() 
        return {"error": f"Regression fitting error: {e}", "formula": regression_formula}

    print("Calculating training residuals...")
    train_residuals = None
    try:
        train_regression_preds = ols_model.predict(train_reg_df)
        y_train_aligned = y_train.loc[train_regression_preds.index]
        train_residuals = y_train_aligned - train_regression_preds
        if train_residuals.isnull().any():
            num_nans = train_residuals.isnull().sum()
            warnings.warn(f"{num_nans} NaNs found unexpectedly in regression residuals after alignment. Filling with 0.", UserWarning)
            train_residuals = train_residuals.fillna(0) 
    except Exception as e:
        print(f"Error predicting regression on training data or calculating residuals: {e}")
        traceback.print_exc()
        return {"error": f"Regression prediction/residual (train) error: {e}", "formula": regression_formula}

    residuals_for_arima_fit = None
    if arima_residuals_subsample is not None and arima_residuals_subsample > 0:
        n_residuals_available = len(train_residuals)
        if n_residuals_available > arima_residuals_subsample:
            print(f"Subsampling residuals: Using last {arima_residuals_subsample} of {n_residuals_available} points for auto_arima fit.")
            residuals_for_arima_fit = train_residuals.iloc[-arima_residuals_subsample:]
        else:
            print(f"Using all {n_residuals_available} available residuals for auto_arima fit (subsample limit not reached).")
            residuals_for_arima_fit = train_residuals
    else:
        n_residuals_available = len(train_residuals)
        print(f"Using all {n_residuals_available} available residuals for auto_arima fit (subsampling disabled).")
        residuals_for_arima_fit = train_residuals

    if residuals_for_arima_fit is None or residuals_for_arima_fit.empty:
         print("Error: Residuals series for ARIMA fit is empty after potential subsampling.")
         return {"error": "Residuals series for ARIMA fit is empty."}
    if len(residuals_for_arima_fit) < 10: 
         print(f"Warning: Residuals series for ARIMA fit is very small (length={len(residuals_for_arima_fit)}). ARIMA might fail.")

    print(f"Fitting auto_arima on {len(residuals_for_arima_fit)} residuals...")
    residual_arima_model = None
    try:
        if residuals_for_arima_fit.nunique() <= 1:
            raise ValueError("Residuals series is constant or near-constant. Cannot fit ARIMA.")

        with warnings.catch_warnings():
            if suppress_warnings:
                warnings.simplefilter("ignore")

            residual_arima_model = pm.auto_arima(
                y=residuals_for_arima_fit, 
                exogenous=None,            
                max_p=arima_max_p,
                max_q=arima_max_q,
                d=arima_d,                
                seasonal=seasonal,
                m=m,
                stepwise=True,            
                suppress_warnings=suppress_warnings,
                error_action='warn',      
                trace=True                 
            )
        print(f"Best Residual ARIMA order: {residual_arima_model.order}, Seasonal order: {residual_arima_model.seasonal_order}")

    except (np.linalg.LinAlgError, MemoryError, ValueError) as e: 
        print(f"Error ({type(e).__name__}) during residual auto_arima: {e}")
        print("This often indicates issues with:")
        print("  - Memory allocation (try reducing arima_residuals_subsample)")
        print("  - Data properties (e.g., residuals being constant after differencing)")
        print("  - Numerical instability (check OLS Cond. No., consider simplifying OLS)")
        traceback.print_exc()
        return {"error": f"{type(e).__name__} in residual auto_arima: {e}", "formula": regression_formula}
    except Exception as e:
        print(f"An unexpected error occurred during residual auto_arima: {e}")
        traceback.print_exc()
        return {"error": f"Exception in residual auto_arima: {e}", "formula": regression_formula}

    print("Forecasting regression component on test data...")
    regression_forecast = None
    try:
        regression_forecast = ols_model.predict(test_reg_df)
        if regression_forecast.isnull().any():
            warnings.warn("NaNs found in regression forecast for test data. Filling with ffill/bfill.", UserWarning)
            regression_forecast = regression_forecast.fillna(method='ffill').fillna(method='bfill') 
    except Exception as e:
        print(f"Error predicting regression on test data: {e}")
        print("Columns available in test_reg_df for prediction:", test_reg_df.columns.tolist())
        print("Model exog names expected by OLS model:", getattr(ols_model.model, 'exog_names', 'N/A'))
        traceback.print_exc()
        return {"error": f"Regression prediction (test) error: {e}", "formula": regression_formula}

    print("Forecasting ARIMA residuals component...")
    arima_residuals_forecast = None
    n_periods_to_forecast = len(test_df)
    try:
        arima_residuals_forecast_np, conf_int_res = residual_arima_model.predict(
            n_periods=n_periods_to_forecast,
            exogenous=None,
            return_conf_int=True
        )
        arima_residuals_forecast = pd.Series(arima_residuals_forecast_np, index=test_df.index)
        if arima_residuals_forecast.isnull().any():
             warnings.warn("NaNs found in ARIMA residuals forecast. Filling with 0.", UserWarning)
             arima_residuals_forecast = arima_residuals_forecast.fillna(0) 

    except Exception as e:
        print(f"Error during ARIMA residual prediction: {e}")
        traceback.print_exc()
        return {
            "error": f"ARIMA Residual Prediction error: {e}",
            "regression_formula": regression_formula,
            "residual_arima_order": str(getattr(residual_arima_model, 'order', 'N/A')),
            "residual_seasonal_order": str(getattr(residual_arima_model, 'seasonal_order', 'N/A'))
        }

    print("Combining regression and residual forecasts...")
    final_forecast = None
    try:
        final_forecast = regression_forecast.add(arima_residuals_forecast, fill_value=0)
        if final_forecast.isnull().any():
             warnings.warn("NaNs found in final combined forecast after adding components. Filling with ffill/bfill.", UserWarning)
             final_forecast = final_forecast.fillna(method='ffill').fillna(method='bfill') 
    except Exception as e:
         print(f"Error combining forecasts: {e}")
         traceback.print_exc()
         return {
            "error": f"Error combining forecasts: {e}",
            "regression_formula": regression_formula,
            "residual_arima_order": str(getattr(residual_arima_model, 'order', 'N/A')),
            "residual_seasonal_order": str(getattr(residual_arima_model, 'seasonal_order', 'N/A'))
        }

    print("Calculating evaluation metrics...")
    results = {}
    try:
        y_test_eval, final_forecast_eval = y_test.align(final_forecast, join='inner') 
        mask = ~y_test_eval.isnull() & ~final_forecast_eval.isnull()
        y_test_clean = y_test_eval[mask]
        final_forecast_clean = final_forecast_eval[mask]
        if len(y_test_clean) == 0:
             print("Error: No valid overlapping data points between y_test and final_forecast after cleaning.")
             results["error"] = "Evaluation error: No valid data points after cleaning NaNs/alignment."
        else:
             mse_val = mean_squared_error(y_test_clean, final_forecast_clean)
             rmse_val = root_mean_squared_error(y_test_clean, final_forecast_clean)
             mae_val = mean_absolute_error(y_test_clean, final_forecast_clean)

             print(f"\nRegression+ARIMA Metrics (Original Scale on {len(y_test_clean)} points):")
             print(f"  MSE  = {mse_val:.6f}")
             print(f"  RMSE = {rmse_val:.6f}")
             print(f"  MAE  = {mae_val:.6f}")

             results = {
                 "mse": float(mse_val),
                 "rmse": float(rmse_val),
                 "mae": float(mae_val),
                 "num_eval_points": int(len(y_test_clean)),
                 "regression_formula": regression_formula,
                 "residual_arima_order": str(getattr(residual_arima_model, 'order', 'N/A')),
                 "residual_seasonal_order": str(getattr(residual_arima_model, 'seasonal_order', 'N/A'))
             }
    except ValueError as e:
        print(f"Error calculating metrics (likely due to NaNs or length mismatch even after cleaning): {e}")
        traceback.print_exc()
        results = {
            "error": f"Metric calculation error: {e}",
            "regression_formula": regression_formula,
            "residual_arima_order": str(getattr(residual_arima_model, 'order', 'N/A')),
            "residual_seasonal_order": str(getattr(residual_arima_model, 'seasonal_order', 'N/A'))
        }
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
        traceback.print_exc()
        results = {
            "error": f"Unexpected evaluation error: {e}",
            "regression_formula": regression_formula,
            "residual_arima_order": str(getattr(residual_arima_model, 'order', 'N/A')),
            "residual_seasonal_order": str(getattr(residual_arima_model, 'seasonal_order', 'N/A'))
        }
    print("-" * 30)
    return results