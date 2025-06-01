import argparse
import itertools
import os
import random
import numpy as np
import torch

from train_eval import train_and_evaluate, device
from utils import (
    load_results,
    save_results,
    load_and_preprocess_data,
    split_and_scale_data,
    build_run_name
)

from arima import run_any_arima_experiment
from arima_baseline import run_arima_case
from reg_arima_baseline import run_regression_arima_errors_case
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM-based time series model for ECD prediction.")

    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=['lstm', 'encdec_lstm', 'encdec_lstm_aug', 'lstm_aug', 'encdec_lstm_tf','arimax', 'regarima'],
                        help="Which model variant to use.")
    parser.add_argument("--hidden_dim", type=int, default=32, help="LSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    parser.add_argument("--window_size", type=int, default=720, help="Number of historical steps")
    parser.add_argument("--horizon", type=int, default=360, help="Number of future steps to predict")
    parser.add_argument("--data_file", type=str, default="input_files/case_1_rot_drill_fluid/TimeSeries.out",
                        help="Path to the main time series data file (e.g., TimeSeries.out). Used for single run and hyperparam search.")
    parser.add_argument("--in_file", type=str, default="input_files/case_1_rot_drill_fluid/Case_5m_rot_operation.in",
                        help="Path to the operation/fluid input file (e.g., operation.in). Used for single run and hyperparam search.")

    parser.add_argument("--fluid", action="store_true", help="If set, enable fluid processing features.")
    parser.add_argument("--p_fluid", action="store_true", help="If set (and --fluid is set), use simple fluid percentage tracking.")
    parser.add_argument("--ds_a_fluid", action="store_true",
                        help="If set (and --fluid is set), use drillstring/annulus fluid tracking.")
    
    parser.add_argument("--reg_poly_degree", type=int, default=2,
                        help="Polynomial degree for regression part in reg_arima model.")
    parser.add_argument("--reg_formula", type=str, default=None,
                        help="Explicit R-style formula for regression part in reg_arima.")

    parser.add_argument("--input_features", nargs="+", default=["Rate_in"],
                        help="List of feature columns to use for training (single run) or default for search if not specified by --search_feature_sets.")
    parser.add_argument("--augmented_encoder", action="store_true",
                        help="If set, use historical ECD as an input to the encoder (requires compatible model type like encdec_lstm_aug).")

    parser.add_argument("--hyperparam_search", action="store_true",
                        help="If set, run a hyperparameter search instead of a single training run")

    parser.add_argument("--ecd_noise_std", type=float, default=0.0,
                        help="Std dev of Gaussian noise added to *scaled* historical ECD during training if using augmented encoder.")

    parser.add_argument("--results_file", type=str, default="hyperparameter_results.json",
                        help="File to save experiment results.")
    parser.add_argument("--model_path", type=str, default="models",
                        help="Directory to save trained model checkpoints.")

    parser.add_argument("--search_model_types", type=str, nargs='+', default=None,
                        choices=['lstm', 'encdec_lstm', 'encdec_lstm_aug', 'lstm_aug', 'encdec_lstm_tf'],
                        help="List of model_type values for hyperparameter search.")
    parser.add_argument("--search_hidden_dims", type=int, nargs='+', default=None,
                        help="List of hidden_dim values for hyperparameter search.")
    parser.add_argument("--search_num_layers", type=int, nargs='+', default=None,
                        help="List of num_layers values for hyperparameter search.")
    parser.add_argument("--search_lrs", type=float, nargs='+', default=None,
                        help="List of learning_rate values for hyperparameter search.")
    parser.add_argument("--search_window_sizes", type=int, nargs='+', default=None,
                        help="List of window_size values for hyperparameter search.")
    parser.add_argument("--search_feature_sets", type=str, nargs='+', default=None,
                        help="List of feature sets for search. Each set as a comma-separated string (e.g., 'Rate_in,dsRpm' 'Density').")
    parser.add_argument("--search_augmented_encoders", type=str, nargs='+', default=None, choices=['True', 'False'],
                        help="List of augmented_encoder states (True/False) for hyperparameter search.")
    parser.add_argument("--search_ecd_noise_stds", type=float, nargs='+', default=None,
                        help="List of ecd_noise_std values for hyperparameter search.")

    return parser.parse_args()


def run_single_experiment(args, results, completed_runs, results_file):
    try:
        df_raw, basic_columns, feature_names_str = load_and_preprocess_data(args)
        train_df, val_df, test_df, scaler = split_and_scale_data(df_raw, basic_columns)
    except FileNotFoundError as e:
        print(f"Error loading data for single run (data_file: {args.data_file}, in_file: {args.in_file}): {e}. Skipping.")
        return results
    except Exception as e:
        print(f"Error processing data for single run (data_file: {args.data_file}, in_file: {args.in_file}): {e}. Skipping.")
        return results

    middle_folder = os.path.basename(os.path.dirname(args.data_file))
    run_name = build_run_name(args.model_type, args, feature_names_str, middle_folder)
    if run_name in completed_runs:
        print(f"Skipping run (already done): {run_name}")
        return results

    print(f"\n[INFO] Running single training for: {run_name}")
    try:
        res = train_and_evaluate(
            model_type=args.model_type,
            input_features=args.input_features, 
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            lr=args.lr,
            run_name=run_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            window_size=args.window_size,
            horizon=args.horizon,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            device=device,
            scaler=scaler,
            patience=args.patience,
            all_columns=basic_columns,
            augmented_encoder=args.augmented_encoder,
            ecd_noise_std=args.ecd_noise_std,
            model_path=args.model_path
        )
        results.append((run_name, args.input_features, res))
        save_results(results, results_file)
    except Exception as e:
        print(f"Error during training/evaluation for run {run_name}: {e}")
        failure_info = {"error": str(e)}
        results.append((run_name, args.input_features, failure_info))
        save_results(results, results_file)
    return results


def run_hyperparameter_search(args, results, completed_runs, results_file):
    print("\n[INFO] Starting hyperparameter search...")
    print(f"[INFO] Using fixed data_file for search: {args.data_file}")
    print(f"[INFO] Using fixed in_file for search: {args.in_file}")

    search_space = {
        "model_types": args.search_model_types if args.search_model_types is not None else [args.model_type],
        "hidden_dims": args.search_hidden_dims if args.search_hidden_dims is not None else [args.hidden_dim],
        "num_layers_list": args.search_num_layers if args.search_num_layers is not None else [args.num_layers],
        "learning_rates": args.search_lrs if args.search_lrs is not None else [args.lr],
        "window_sizes": args.search_window_sizes if args.search_window_sizes is not None else [args.window_size],
    }

    if args.search_feature_sets is not None:
        search_space["feature_sets"] = [[f.strip() for f in fs.split(',')] for fs in args.search_feature_sets]
    else:
        search_space["feature_sets"] = [args.input_features]


    if args.search_augmented_encoders is not None:
        search_space["augmented_encoders"] = [s == 'True' for s in args.search_augmented_encoders]
    else:
        search_space["augmented_encoders"] = [args.augmented_encoder]

    if args.search_ecd_noise_stds is not None:
        search_space["ecd_noise_stds"] = args.search_ecd_noise_stds
    else:
        search_space["ecd_noise_stds"] = [args.ecd_noise_std]


    param_combinations = list(itertools.product(
        search_space["model_types"],
        search_space["feature_sets"],
        search_space["hidden_dims"],
        search_space["num_layers_list"],
        search_space["learning_rates"],
        search_space["window_sizes"],
        search_space["augmented_encoders"],
        search_space["ecd_noise_stds"]
    ))

    total_runs = len(param_combinations)
    print(f"Total combinations to check: {total_runs}")
    run_count = 0
    base_args_for_data_load = argparse.Namespace(**vars(args))
    try:
        df_raw_search, basic_columns_search, _ = load_and_preprocess_data(base_args_for_data_load)
        train_df_search, val_df_search, test_df_search, scaler_search = split_and_scale_data(df_raw_search, basic_columns_search)

    except FileNotFoundError as e:
        print(f"FATAL: Error loading data for hyperparameter search (data_file: {base_args_for_data_load.data_file}, in_file: {base_args_for_data_load.in_file}): {e}. Aborting search.")
        return results
    except Exception as e:
        print(f"FATAL: Error processing data for hyperparameter search (data_file: {base_args_for_data_load.data_file}, in_file: {base_args_for_data_load.in_file}): {e}. Aborting search.")
        return results


    for combo in param_combinations:
        run_count += 1
        (model_type, feats, hd, nl, lr, ws, aug_enc, noise_std) = combo

        missing_from_scaled_data = [f for f in feats if f not in basic_columns_search]
        if missing_from_scaled_data:
            print(f"Skipping combo {run_count}/{total_runs}: Features {missing_from_scaled_data} not found in preloaded/scaled data columns derived from basic_columns: {basic_columns_search}. "
                  f"This might be due to initial data load not covering all features from all search sets.")
            continue


        if not aug_enc and noise_std > 0:
            print(f"Skipping combo {run_count}/{total_runs}: Noise std > 0 requires augmented_encoder=True.")
            continue
        if aug_enc and model_type not in ['encdec_lstm_aug', 'lstm_aug']:
            print(f"Skipping combo {run_count}/{total_runs}: augmented_encoder=True requires encdec_lstm_aug or lstm_aug model type.")
            continue
        if not aug_enc and model_type in ['encdec_lstm_aug', 'lstm_aug']:
            print(f"Skipping combo {run_count}/{total_runs}: augmented_encoder=False requires non-augmented model type.")
            continue

        current_args = argparse.Namespace(**vars(base_args_for_data_load))
        current_args.model_type = model_type
        current_args.input_features = feats 
        current_args.hidden_dim = hd
        current_args.num_layers = nl
        current_args.lr = lr
        current_args.window_size = ws
        current_args.augmented_encoder = aug_enc
        current_args.ecd_noise_std = noise_std
    
        current_args.fluid = True if any(f_el in current_args.input_features for f_el in ['fluid_binary', 'pct_obm1', 'DS_event_pct', 'new_fluid_reached_annulus']) or base_args_for_data_load.fluid else False
        if current_args.fluid : 
            current_args.p_fluid = True if 'pct_obm1' in current_args.input_features or ('pct_Case_2a1_Fluid' in current_args.input_features and not current_args.ds_a_fluid) else False
            current_args.ds_a_fluid = True if 'DS_event_pct' in current_args.input_features or 'new_fluid_reached_annulus' in current_args.input_features else False
            if current_args.p_fluid and current_args.ds_a_fluid:
                 current_args.p_fluid = False
        else:
            current_args.p_fluid = False
            current_args.ds_a_fluid = False

        middle_folder = os.path.basename(os.path.dirname(current_args.data_file))
        feature_names_str_combo = "_".join(sorted(feats))
        if current_args.p_fluid : feature_names_str_combo += '_pctSimple'
        if current_args.ds_a_fluid : feature_names_str_combo += '_pctDSA'  

        run_name = build_run_name(model_type, current_args, feature_names_str_combo, middle_folder)

        if run_name in completed_runs:
            print(f"Skipping run {run_count}/{total_runs} (already done): {run_name}")
            continue

        print(f"\n--- Starting Run {run_count}/{total_runs}: {run_name} ---")
        print(f"    Features for this run: {feats}")
        try:
            kwargs = dict(
                model_type=model_type,
                input_features=current_args.input_features, 
                hidden_dim=hd,
                num_layers=nl,
                lr=lr,
                run_name=run_name,
                train_df=train_df_search, 
                val_df=val_df_search,     
                test_df=test_df_search,   
                window_size=ws,
                horizon=current_args.horizon,
                num_epochs=current_args.num_epochs,
                batch_size=current_args.batch_size,
                device=device,
                scaler=scaler_search,     
                patience=current_args.patience,
                all_columns=basic_columns_search, 
                augmented_encoder=aug_enc,
                ecd_noise_std=noise_std,
                model_path=current_args.model_path
            )
            res = train_and_evaluate(**kwargs)
            results.append((run_name, current_args.input_features, res))
            save_results(results, results_file)
            completed_runs.add(run_name)
        except Exception as e:
            print(f"Error during training/evaluation for run {run_name} (Combo {run_count}/{total_runs}): {e}")
            import traceback
            traceback.print_exc()
            failure_info = {"error": str(e)}
            results.append((run_name, current_args.input_features, failure_info))
            save_results(results, results_file)
            completed_runs.add(run_name)

    print("\n[INFO] Hyperparameter search finished.")
    return results

def main():
    set_seed(42)
    args = parse_arguments()
    results_file = args.results_file
    results, completed_runs_list = load_results(results_file)
    completed_runs = set(completed_runs_list)
    
    if args.model_type.lower() == 'arimax':
        run_any_arima_experiment(args, results, completed_runs, results_file,
                                model_func=run_arima_case,
                                model_name_prefix="ARIMAX")
    elif args.model_type.lower() == 'regarima':
        run_any_arima_experiment(args, results, completed_runs, results_file,
                                model_func=run_regression_arima_errors_case,
                                model_name_prefix="RegARIMA")

    if args.hyperparam_search:
        run_hyperparameter_search(args, results, completed_runs, results_file)
    
    elif args.model_type.lower() not in ['arimax', 'regarima']:
        if len(args.input_features) == 1 and ',' in args.input_features[0]:
            args.input_features = [f.strip() for f in args.input_features[0].split(',')]
        elif any(',' in f for f in args.input_features): 
             new_features = []
             for item in args.input_features:
                 new_features.extend([f.strip() for f in item.split(',')])
             args.input_features = new_features

        run_single_experiment(args, results, completed_runs, results_file)

    print("\nExecution finished.")

if __name__ == "__main__":
    main()