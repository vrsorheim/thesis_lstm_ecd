from utils import load_and_preprocess_data, save_results
import warnings
import os
import pandas as pd

def run_any_arima_experiment(args, results, completed_runs, results_file, model_func, model_name_prefix):
    print(f"[INFO] Setting up {model_name_prefix} baseline run...")
    try:
        df_raw, basic_columns, feature_names_str = load_and_preprocess_data(args)
        target_col = 'ECD_bot'
        cols_to_keep = list(dict.fromkeys([target_col] + args.input_features))
        missing_cols = [col for col in cols_to_keep if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for {model_name_prefix}: {missing_cols}")

        df_run = df_raw[cols_to_keep].sort_index().copy()

        n = len(df_run)
        train_prop = 0.70
        val_prop = 0.15
        train_end_idx = int(n * train_prop)
        val_end_idx = int(n * (train_prop + val_prop))
        train_df = df_run.iloc[:train_end_idx].copy()
        val_df = df_run.iloc[train_end_idx:val_end_idx].copy()
        test_df = df_run.iloc[val_end_idx:].copy()

        print(f"Manual split sizes: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

        train_fit_df = pd.concat([train_df, val_df]).sort_index()
        print(f"Combined Train+Val size for fitting: {len(train_fit_df)}")

        if train_fit_df.isnull().values.any() or test_df.isnull().values.any():
            warnings.warn("NaN values detected in train_fit_df or test_df after split. Check data preprocessing.")

    except FileNotFoundError as e:
        print(f"Error loading data for {model_name_prefix} run: {e}. Skipping.")
        return results
    except ValueError as e:
        print(f"Data processing error for {model_name_prefix} run: {e}. Skipping.")
        return results
    except Exception as e:
        print(f"Unexpected error processing data for {model_name_prefix} run: {e}. Skipping.")
        import traceback
        traceback.print_exc()
        return results
    
    middle_folder = os.path.basename(os.path.dirname(args.data_file))
    if model_name_prefix == "RegARIMA":
        if args.reg_formula:
             run_name = f"{model_name_prefix}_Formula_Feats{feature_names_str}_Data{middle_folder}"
        else:
             run_name = f"{model_name_prefix}_Poly{args.reg_poly_degree}_Feats{feature_names_str}_Data{middle_folder}"
    else: 
        run_name = f"{model_name_prefix}_Feats{feature_names_str}_Data{middle_folder}"

    if run_name in completed_runs:
        print(f"Skipping {model_name_prefix} run (already done): {run_name}")
        return results

    print(f"\n--- Starting {model_name_prefix} Run: {run_name} ---")
    print(f"    Target: {target_col}")
    print(f"    Exogenous Features: {args.input_features}")
    print(f"    Fit size: {len(train_fit_df)}, Test size: {len(test_df)}") 

    try:
        specific_args = {
            'train_df': train_fit_df,
            'test_df': test_df,
            'input_features': args.input_features,
            'target_feature': target_col
        }
        if model_name_prefix == "RegARIMA":
            specific_args['regression_formula'] = args.reg_formula
            specific_args['poly_degree'] = args.reg_poly_degree
        model_results = model_func(**specific_args)
        
        if model_results is None:
             print(f"Warning: {model_name_prefix} function returned None for run {run_name}")
             model_results = {"error": f"{model_name_prefix} function returned None"}

        results.append((run_name, args.input_features, model_results))
        save_results(results, results_file)
        if "error" not in model_results: 
             completed_runs.add(run_name)
        print(f"--- Finished {model_name_prefix} Run: {run_name} ---")

    except ImportError as e:
         print(f"{model_name_prefix} Error: Required library not found. Error: {e}")
         failure_info = {"error": f"ImportError: {e}."}
         results.append((run_name, args.input_features, failure_info))
         save_results(results, results_file)
         completed_runs.add(run_name)
    except Exception as e:
        print(f"Error during {model_name_prefix} execution for run {run_name}: {e}")
        import traceback
        traceback.print_exc()
        failure_info = {"error": str(e)}
        results.append((run_name, args.input_features, failure_info))
        save_results(results, results_file)
        completed_runs.add(run_name)

    return results
