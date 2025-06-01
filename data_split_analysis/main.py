import sys
import os
import json
import argparse
import re 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import collections

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from utils import (
        load_results,
        load_and_preprocess_data,
        split_and_scale_data,
    )
    from architectures import create_model
    from train_eval import FutureAwareTimeSeriesDataset, AugmentedTimeSeriesDataset, device, inverse_transform_ecd
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure utils.py, architectures.py, and train_eval.py are in the Python path.")
    sys.exit(1)

class FutureAwareTimeSeriesDatasetWithFeatures(FutureAwareTimeSeriesDataset):
    def __init__(self, data: pd.DataFrame, window_size, horizon, input_features, target_feature, extra_feature_names: list):
        super().__init__(data, window_size, horizon, input_features, target_feature)
        self.extra_feature_names = extra_feature_names
        missing_extra = [f for f in self.extra_feature_names if f not in data.columns]
        if missing_extra:
            raise ValueError(f"Extra features not found in data: {missing_extra}")
        self.extra_features_data = self.data[self.extra_feature_names].values

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx) 
        start_pred_idx_in_data = idx + self.window_size
        if start_pred_idx_in_data < len(self.extra_features_data):
             extra_features_values = self.extra_features_data[start_pred_idx_in_data]
        else:
             extra_features_values = np.full(len(self.extra_feature_names), np.nan)
        return X, y, torch.tensor(extra_features_values, dtype=torch.float32)


class AugmentedTimeSeriesDatasetWithFeatures(AugmentedTimeSeriesDataset):
    def __init__(self, data: pd.DataFrame, window_size, horizon, input_features, target_feature, ecd_noise_std, extra_feature_names: list):
        super().__init__(data, window_size, horizon, input_features, target_feature, ecd_noise_std)
        self.extra_feature_names = extra_feature_names
        missing_extra = [f for f in self.extra_feature_names if f not in data.columns]
        if missing_extra:
            raise ValueError(f"Extra features not found in data: {missing_extra}")
        self.extra_features_data = self.data[self.extra_feature_names].values

    def __getitem__(self, idx):
        x_hist, x_dec, y = super().__getitem__(idx)
        start_pred_idx_in_data = idx + self.window_size
        if start_pred_idx_in_data < len(self.extra_features_data):
             extra_features_values = self.extra_features_data[start_pred_idx_in_data]
        else:
             extra_features_values = np.full(len(self.extra_feature_names), np.nan)

        return x_hist, x_dec, y, torch.tensor(extra_features_values, dtype=torch.float32)

def parse_analysis_args():
    parser = argparse.ArgumentParser(description="Analyze pre-trained ECD prediction models with derived metrics.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model .pt file.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the JSON results file containing hyperparameters for the model run.")
    parser.add_argument("--data_file", type=str, default="../input_files/case_1_rot_drill_fluid/TimeSeries.out", help="Path to the main time series data file (e.g., TimeSeries.out used for training).")
    parser.add_argument("--in_file", type=str, default="../input_files/case_1_rot_drill_fluid/Case_5m_rot_operation.in", help="Path to the operation/fluid input file (e.g., operation.in used for training).")
    parser.add_argument("--transition_feature", type=str, default="time_since_switch", help="Feature name used to define transitions.")
    parser.add_argument("--transition_threshold", type=float, default=360, help="Threshold for the transition feature (e.g., steps below this are considered 'transition'). Value relates to how many steps after the change we consider it as a transition.") # 360 steps = 1 hour if step is 10s
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    return parser.parse_args()

def analyze(args):
    print(f"--- Starting Analysis for Model: {args.model_checkpoint} ---")

    run_name = os.path.splitext(os.path.basename(args.model_checkpoint))[0]
    print(f"Extracted Run Name: {run_name}")

    print(f"Loading results from: {args.results_file}")
    results_data, _ = load_results(args.results_file)
    run_config = None
    run_hparams = None
    run_features = None
    found_run = False
    for item in results_data:
        if isinstance(item, (list, tuple)) and len(item) >= 2: 
            current_run_name = item[0]
            if current_run_name == run_name:
                print(f"Found matching run entry for: {run_name}")
                run_features = item[1] 
                try:
                    if 'ENCDEC_LSTM' in run_name.upper():
                        model_type_match = "encdec_lstm"
                    elif 'LSTM' in run_name.upper():
                        model_type_match = "lstm"
                    else:
                        print(f"Warning: Could not infer model type from run name '{run_name}'. Assuming 'encdec_lstm'.")
                        model_type_match = "encdec_lstm" 
                    hd_match = re.search(r'_HD(\d+)', run_name)
                    nl_match = re.search(r'_NL(\d+)', run_name)
                    lr_match = re.search(r'_LR([\d.eE-]+)', run_name) 
                    ws_match = re.search(r'_WS(\d+)', run_name)
                    hz_match = re.search(r'_HZ(\d+)', run_name)
                    dp_match = re.search(r'_DP([\d.]+)', run_name)     
                    lstm_dp_match = re.search(r'_LSTMDP([\d.]+)', run_name) 

                    if not all([hd_match, nl_match, lr_match, ws_match, hz_match]):
                         raise ValueError("Could not parse essential numeric hyperparameters (HD, NL, LR, WS, HZ) from run name.")

                    is_augmented = 'AUG' in model_type_match.upper() or '_augmECD' in run_name 
                    if is_augmented:
                        model_type_match += "_aug"
                    run_hparams = argparse.Namespace(
                        model_type=model_type_match.lower(),
                        hidden_dim=int(hd_match.group(1)),
                        num_layers=int(nl_match.group(1)),
                        lr=float(lr_match.group(1)),
                        window_size=int(ws_match.group(1)),
                        horizon=int(hz_match.group(1)),
                        augmented_encoder=is_augmented, 
                        input_features=run_features
                    )
                    found_run = True
                    break 
                except Exception as e:
                    print(f"Error parsing hyperparameters from run_name '{run_name}': {e}")
                    found_run = False

    if not found_run:
        print(f"Error: Could not find run '{run_name}' OR could not parse its hyperparameters from the name in {args.results_file}")
        sys.exit(1)

    if not run_hparams.input_features or not isinstance(run_hparams.input_features, list):
        print(f"Error: Input features not found or not a list for run {run_name} in {args.results_file}.")
        sys.exit(1)
    print("Loading and preprocessing data...")
    input_features_set = set(run_hparams.input_features)
    has_fluid_binary = 'fluid_binary' in input_features_set
    has_time_switch = 'time_since_switch' in input_features_set
    has_pct = any('pct' in f for f in input_features_set)
    has_ds_event = any('DS_event' in f for f in input_features_set)
    has_annulus_reach = any('reached_annulus' in f for f in input_features_set)

    fluid_flag = has_fluid_binary or has_time_switch or has_pct or has_ds_event or has_annulus_reach
    p_fluid_flag = has_pct and not has_ds_event
    ds_a_fluid_flag = has_ds_event or has_annulus_reach

    print(f"Inferred data processing flags: fluid={fluid_flag}, p_fluid={p_fluid_flag}, ds_a_fluid={ds_a_fluid_flag}")
    data_args = argparse.Namespace(
        data_file=args.data_file,
        in_file=args.in_file,
        fluid=fluid_flag,
        p_fluid=p_fluid_flag,
        ds_a_fluid=ds_a_fluid_flag,
        input_features=run_hparams.input_features 
    )
    try:
        df_raw, basic_columns, _ = load_and_preprocess_data(data_args)
        if args.transition_feature in df_raw.columns:
             if args.transition_feature not in basic_columns:
                 basic_columns.append(args.transition_feature)
        elif args.transition_feature: 
             print(f"Error: Requested transition feature '{args.transition_feature}' not found in loaded data columns: {df_raw.columns.tolist()}.")
             sys.exit(1)
        cols_needed_for_scaling = list(dict.fromkeys(run_hparams.input_features + ['ECD_bot'] + ([args.transition_feature] if args.transition_feature else [])))
        missing_in_basic = [col for col in cols_needed_for_scaling if col not in basic_columns]
        if missing_in_basic:
            print(f"Warning: Columns needed for scaling were not in basic_columns returned by preprocessing: {missing_in_basic}. Adding them.")
            basic_columns.extend(missing_in_basic)
            basic_columns = list(dict.fromkeys(basic_columns)) 

        missing_in_df = [col for col in basic_columns if col not in df_raw.columns]
        if missing_in_df:
             print(f"Error: Columns required for scaling are missing from the final preprocessed DataFrame: {missing_in_df}")
             sys.exit(1)
        df_to_split = df_raw[basic_columns]

        _, _, test_df_scaled, scaler = split_and_scale_data(df_to_split, basic_columns)
        all_columns_for_scaler = basic_columns.copy()
        print(f"Data loaded. Test set size: {len(test_df_scaled)}. Columns scaled: {all_columns_for_scaler}")

    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)
    except ValueError as e: 
        print(f"Error during data processing or scaling: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")
        raise 

    print("Creating test DataLoader...")
    extra_features_to_get = []
    if args.transition_feature and args.transition_feature in all_columns_for_scaler:
        extra_features_to_get = [args.transition_feature]
        print(f"Requesting extra feature for analysis: {extra_features_to_get}")
    elif args.transition_feature:
        print(f"Warning: Requested transition feature '{args.transition_feature}' not found in scaled columns, cannot retrieve it.")

    required_ds_cols = run_hparams.input_features + ['ECD_bot'] + extra_features_to_get
    missing_ds_cols = [c for c in required_ds_cols if c not in test_df_scaled.columns]
    if missing_ds_cols:
        print(f"Error: Columns required for dataset creation are missing from test_df_scaled: {missing_ds_cols}")
        sys.exit(1)

    if run_hparams.augmented_encoder:
         print("Using AugmentedTimeSeriesDatasetWithFeatures.")
         test_dataset = AugmentedTimeSeriesDatasetWithFeatures(
             data=test_df_scaled, window_size=run_hparams.window_size, horizon=run_hparams.horizon,
             input_features=run_hparams.input_features, target_feature='ECD_bot',
             ecd_noise_std=0.0, 
             extra_feature_names=extra_features_to_get
         )
    else: 
         print("Using FutureAwareTimeSeriesDatasetWithFeatures.")
         test_dataset = FutureAwareTimeSeriesDatasetWithFeatures(
             data=test_df_scaled, window_size=run_hparams.window_size, horizon=run_hparams.horizon,
             input_features=run_hparams.input_features, target_feature='ECD_bot',
             extra_feature_names=extra_features_to_get
         )
    if len(test_dataset) == 0:
        print("Error: Test dataset is empty! Check data splitting or window/horizon size.")
        sys.exit(1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Loading model...")
    input_dim = len(run_hparams.input_features)
    model = create_model(
        model_type=run_hparams.model_type,
        input_dim=input_dim,
        hidden_dim=run_hparams.hidden_dim,
        num_layers=run_hparams.num_layers,
        horizon=run_hparams.horizon
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        model.eval() 
        print(f"Model state loaded successfully from {args.model_checkpoint}")
    except Exception as e:
        print(f"Error loading model state dict from {args.model_checkpoint}: {e}")
        sys.exit(1)

    print("Running inference on test set...")
    all_preds_scaled = []
    all_targets_scaled = []
    all_extra_features = []
    with torch.no_grad():
        for batch in test_loader:
            if run_hparams.augmented_encoder:
                if len(extra_features_to_get) > 0:
                    x_hist, x_dec, y_scaled, extra_feats = batch
                else:
                    x_hist, x_dec, y_scaled = batch 
                    extra_feats = torch.empty(y_scaled.size(0), 0) 
                x_hist, x_dec, y_scaled = x_hist.to(device), x_dec.to(device), y_scaled.to(device)
                y_pred_scaled = model(x_hist, x_dec)
            else:
                if len(extra_features_to_get) > 0:
                    X_scaled, y_scaled, extra_feats = batch
                else:
                    X_scaled, y_scaled = batch 
                    extra_feats = torch.empty(y_scaled.size(0), 0)
                X_scaled, y_scaled = X_scaled.to(device), y_scaled.to(device)
                y_pred_scaled = model(X_scaled)
            all_preds_scaled.append(y_pred_scaled.cpu().numpy())
            all_targets_scaled.append(y_scaled.cpu().numpy())
            if len(extra_features_to_get) > 0:
                all_extra_features.append(extra_feats.cpu().numpy())

    if not all_preds_scaled:
        print("Error: No predictions were generated. Test loader might be empty or issue during inference.")
        sys.exit(1)

    all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    all_targets_scaled = np.concatenate(all_targets_scaled, axis=0)
    if len(extra_features_to_get) > 0:
        all_extra_features = np.concatenate(all_extra_features, axis=0)
        print(f"Collected extra features shape: {all_extra_features.shape}")
    else:
         all_extra_features = np.array([]) 
    print(f"Inference complete. Predictions shape: {all_preds_scaled.shape}, Targets shape: {all_targets_scaled.shape}")

    print("Inverse transforming predictions and targets...")
    try:
        all_preds_inv = inverse_transform_ecd(all_preds_scaled, scaler, all_columns=all_columns_for_scaler)
        all_targets_inv = inverse_transform_ecd(all_targets_scaled, scaler, all_columns=all_columns_for_scaler)

        extra_feature_values_inv = None 
        is_transition = None         

        if all_extra_features.size > 0 and args.transition_feature:
            try:
                extra_feature_index = all_columns_for_scaler.index(args.transition_feature)
                print(f"Found transition feature '{args.transition_feature}' at index {extra_feature_index} in scaler columns.")
                dummy_extra = np.zeros((all_extra_features.shape[0], len(all_columns_for_scaler)))
                if all_extra_features.shape[1] == 1:
                    dummy_extra[:, extra_feature_index] = all_extra_features[:, 0]
                    inv_extra_full = scaler.inverse_transform(dummy_extra)
                    extra_feature_values_inv = inv_extra_full[:, extra_feature_index] 
                    print(f"Inverse transformed '{args.transition_feature}'. Example values: {extra_feature_values_inv[:5]}")

                    is_transition = extra_feature_values_inv < args.transition_threshold
                    print(f"Using inverse-transformed '{args.transition_feature}' < {args.transition_threshold} for transition definition.")
                else:
                     print(f"Warning: Expected 1 extra feature, but got {all_extra_features.shape[1]}. Cannot reliably inverse transform transition feature.")
                     is_transition = np.zeros(all_preds_inv.shape[0], dtype=bool) 

            except ValueError:
                print(f"Warning: Transition feature '{args.transition_feature}' not found in scaler columns ({all_columns_for_scaler}). Cannot use inverse-transformed value for thresholding.")
                is_transition = np.zeros(all_preds_inv.shape[0], dtype=bool)
            except Exception as e_inv:
                 print(f"Error during inverse transform of extra feature: {e_inv}")
                 is_transition = np.zeros(all_preds_inv.shape[0], dtype=bool)
        elif args.transition_feature:
             print(f"Warning: Transition feature '{args.transition_feature}' requested but not collected. Cannot calculate transition metrics.")
             is_transition = np.zeros(all_preds_inv.shape[0], dtype=bool) 
        else:
             print("No transition feature requested. Skipping transition metrics.")
             is_transition = np.zeros(all_preds_inv.shape[0], dtype=bool)
    except Exception as e:
        print(f"Error during inverse transform of predictions/targets: {e}")
        raise
    print("\n--- Calculated Metrics ---")
    mse_overall = mean_squared_error(all_targets_inv, all_preds_inv) 
    rmse_overall = root_mean_squared_error(all_targets_inv, all_preds_inv)
    mae_overall = mean_absolute_error(all_targets_inv, all_preds_inv)
    print(f"Overall MAE : {mae_overall:.10f}")
    print(f"Overall RMSE: {rmse_overall:.10f}")
    print(f"Overall MSE : {mse_overall:.10f}") 

    if all_targets_inv.shape[1] > 1: 
         actual_ecd_diff = np.diff(all_targets_inv, axis=1)
         predicted_ecd_diff = np.diff(all_preds_inv, axis=1)
         rmse_diff = root_mean_squared_error(actual_ecd_diff, predicted_ecd_diff)
         mae_diff = mean_absolute_error(actual_ecd_diff, predicted_ecd_diff) 
         print(f"Rate of Change MAE : {mae_diff:.10f}")
         print(f"Rate of Change RMSE: {rmse_diff:.10f}")
    else:
         rmse_diff = float('nan')
         mae_diff = float('nan')
         print("Rate of Change Metrics: N/A (Horizon=1)")

    if is_transition is not None and np.any(is_transition): 
        transition_indices = np.where(is_transition)[0]
        non_transition_indices = np.where(~is_transition)[0]
        print(f"Transition definition: '{args.transition_feature}' < {args.transition_threshold}")
        if len(transition_indices) > 0:
            rmse_transition = root_mean_squared_error(
                all_targets_inv[transition_indices],
                all_preds_inv[transition_indices]
            )
            mae_transition = mean_absolute_error( 
                 all_targets_inv[transition_indices],
                 all_preds_inv[transition_indices]
            )
            print(f"Transition MAE     ({len(transition_indices)} samples): {mae_transition:.10f}")
            print(f"Transition RMSE    ({len(transition_indices)} samples): {rmse_transition:.10f}")
        else:
            rmse_transition = float('nan')
            mae_transition = float('nan')
            print(f"Transition Metrics: N/A (No samples met the transition condition)")

        if len(non_transition_indices) > 0:
            rmse_non_transition = root_mean_squared_error(
                all_targets_inv[non_transition_indices],
                all_preds_inv[non_transition_indices]
            )
            mae_non_transition = mean_absolute_error( 
                 all_targets_inv[non_transition_indices],
                 all_preds_inv[non_transition_indices]
            )
            print(f"Non-Transition MAE ({len(non_transition_indices)} samples): {mae_non_transition:.10f}")
            print(f"Non-Transition RMSE({len(non_transition_indices)} samples): {rmse_non_transition:.10f}")
        else:
            rmse_non_transition = float('nan')
            mae_non_transition = float('nan')
            print("Non-Transition Metrics: N/A (All samples were transition samples)")
    else:
         print("Transition metrics skipped (Condition not met or feature unavailable).")

    print("\n--- Plotting Combined Error vs. Fluid Change Frequency ---")
    try:
        if extra_feature_values_inv is None or extra_feature_values_inv.size == 0:
             print("Warning: Inverse-transformed transition feature ('extra_feature_values_inv') not available or empty. Skipping frequency plot.")
        else:
            interval_sizes_to_test = [1500, 2000, 2500, 3500, 4000,5000, 5500,6000,6500, 7000,8000, 9000,10000, 11000, 12000,15000,18000, 20000, 25000]
            print(f"Analyzing interval sizes: {interval_sizes_to_test} to create combined trend.")
            print(f"Using '{args.transition_feature}' == 0 (approx.) to identify switches.")
            combined_data_for_avg = collections.defaultdict(lambda: {'mse_data': [], 'rmse_data': []})
            num_samples_total = all_preds_inv.shape[0]

            for interval_size in interval_sizes_to_test:
                print(f"  Processing interval_size = {interval_size}...")
                interval_results = []
                for i in range(0, num_samples_total, interval_size):
                    start_idx = i
                    end_idx = min(i + interval_size, num_samples_total)
                    if start_idx >= end_idx: continue
                    interval_time_since_switch = extra_feature_values_inv[start_idx:end_idx]
                    interval_preds = all_preds_inv[start_idx:end_idx]
                    interval_targets = all_targets_inv[start_idx:end_idx]
                    interval_mse = 0.0; interval_rmse = 0.0
                    if interval_preds.size > 0:
                        interval_mse = mean_squared_error(interval_targets.ravel(), interval_preds.ravel())
                        interval_rmse = root_mean_squared_error(interval_targets.ravel(), interval_preds.ravel())
                    num_switches = np.sum(np.abs(interval_time_since_switch) < 1)
                    interval_results.append((num_switches, interval_mse, interval_rmse))
                grouped_errors = collections.defaultdict(lambda: {'mse_list': [], 'rmse_list': []})
                for num_switches, mse, rmse in interval_results:
                    grouped_errors[num_switches]['mse_list'].append(mse)
                    grouped_errors[num_switches]['rmse_list'].append(rmse)
                current_switch_counts = sorted(grouped_errors.keys())
                if not current_switch_counts:
                    print(f"    No switches found for interval_size {interval_size}.")
                    continue

                for k in current_switch_counts:
                    avg_mse_k = np.mean(grouped_errors[k]['mse_list'])
                    avg_rmse_k = np.mean(grouped_errors[k]['rmse_list'])
                    n_k = len(grouped_errors[k]['mse_list']) 
                    if n_k > 0: 
                        combined_data_for_avg[k]['mse_data'].append((avg_mse_k, n_k))
                        combined_data_for_avg[k]['rmse_data'].append((avg_rmse_k, n_k))

            final_combined_errors = {}
            overall_switch_counts = sorted(combined_data_for_avg.keys())

            if not overall_switch_counts:
                print("No data collected across all interval sizes. Skipping combined plot.")
            else:
                print("\nCalculating final weighted average errors across interval sizes...")
                for k in overall_switch_counts:
                    mse_points = combined_data_for_avg[k]['mse_data']
                    total_weight_mse = sum(n for _, n in mse_points)
                    weighted_avg_mse = sum(mse * n for mse, n in mse_points) / total_weight_mse if total_weight_mse > 0 else 0

                    rmse_points = combined_data_for_avg[k]['rmse_data']
                    total_weight_rmse = sum(n for _, n in rmse_points)
                    weighted_avg_rmse = sum(rmse * n for rmse, n in rmse_points) / total_weight_rmse if total_weight_rmse > 0 else 0

                    total_n_k = total_weight_mse 

                    final_combined_errors[k] = {'avg_mse': weighted_avg_mse, 'avg_rmse': weighted_avg_rmse, 'total_n': total_n_k}
                    print(f"  - {k} switches/interval: Combined Avg MSE={weighted_avg_mse:.10f}, Combined Avg RMSE={weighted_avg_rmse:.10f} (from total {total_n_k} intervals across analyses)")

                plot_switch_counts = list(final_combined_errors.keys())
                plot_avg_mse = [final_combined_errors[c]['avg_mse'] for c in plot_switch_counts]
                plot_avg_rmse = [final_combined_errors[c]['avg_rmse'] for c in plot_switch_counts]
                fig, ax1 = plt.subplots(figsize=(10, 6))

                color_mse = 'orange'
                ax1.set_xlabel('Number of Fluid Switches')
                ax1.set_ylabel('Combined Average MSE', color=color_mse)
                ax1.plot(plot_switch_counts, plot_avg_mse, color=color_mse, marker='s', linestyle='-', linewidth=2, label='Weighted Avg MSE')
                ax1.tick_params(axis='y', labelcolor=color_mse)
                ax1.set_xticks(plot_switch_counts) 

                ax2 = ax1.twinx()
                color_rmse = 'blue'
                ax2.set_ylabel('Combined Average RMSE', color=color_rmse)
                ax2.plot(plot_switch_counts, plot_avg_rmse, color=color_rmse, marker='o', linestyle='--', linewidth=2, label='Weighted Avg RMSE')
                ax2.tick_params(axis='y', labelcolor=color_rmse)

                ax1.set_ylim(bottom=0)
                ax2.set_ylim(bottom=0)

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.suptitle(f'Combined Model Error vs. Fluid Switch Frequency', fontsize=10)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

                plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                if not os.path.exists("analysis_results"):
                     os.makedirs("analysis_results")
                save_path_combined = f"analysis_results/{run_name}_error_vs_switch_freq_combined_avg.png"

                try:
                    plt.savefig(save_path_combined)
                    print(f"Combined plot saved to {save_path_combined}")
                except Exception as save_err:
                    print(f"Error saving combined plot: {save_err}")

                print("Displaying combined plot...")
                plt.show()

    except ImportError:
        print("Warning: Matplotlib not found. Install it (`pip install matplotlib`) to generate the plot.")
    except Exception as e:
        print(f"Error during plotting: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    print("--- Analysis Complete ---")

if __name__ == "__main__":
    analysis_args = parse_analysis_args()
    analyze(analysis_args)
