import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) 

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm          
import seaborn as sns   


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_and_preprocess_data, split_and_scale_data
from architectures import create_model
from train_eval import device, inverse_transform_ecd
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Find test interval with most fluid changes and run rolling forecast."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved model checkpoint (e.g., models/my_run.pt)")
    parser.add_argument("--data_file", type=str, default="../input_files/case_1_rot_drill_fluid/TimeSeries.out",
                        help="Path to the time series data file")
    parser.add_argument("--window_size", type=int, default=720, 
                        help="Historical window size (must match model training)")
    parser.add_argument("--horizon", type=int, default=360, 
                        help="Forecast horizon per segment (must match model training)")
    parser.add_argument("--model_type", type=str, default="lstm",
                        help="Model type")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of layers")
    parser.add_argument("--input_features", nargs="+", default=["Rate_in"],
                        help="List of input feature columns for the model")
    parser.add_argument("--target_feature", type=str, default="ECD_bot",
                        help="Name of the target feature (for inverse scaling)")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size (if applicable)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (if applicable)")
    parser.add_argument("--in_file", type=str, default="../input_files/case_1_rot_drill_fluid/Case_5m_rot_operation.in",
                        help="Path to the fluid/operation.in file for merging")
    parser.add_argument("--fluid", action="store_true", help="If set, run with fluid changes preprocessing")
    parser.add_argument("--p_fluid", action="store_true", help="If set, run with fluid percentages")
    parser.add_argument("--ds_a_fluid", action="store_true", help="If set, run with fluid percentages on ds and annulus too")
    parser.add_argument("--augmented_encoder", action="store_true", help="If set, model expects augmented ECD in hist input")
    parser.add_argument("--interval_length", type=int, default=10000,
                        help="Length of the interval to find and predict.")
    parser.add_argument("--fluid_change_col", type=str, default="fluid_binary",
                        help="Column name indicating fluid changes (must be in input_features or loaded).")
    parser.add_argument("--output_suffix", type=str, default="_fluid_change_interval",
                        help="Suffix for output json and plot files.")

    return parser.parse_args()

def find_most_active_interval(data, interval_length, change_col_name):
    if len(data) < interval_length:
        print(f"Error: Data length ({len(data)}) is less than interval length ({interval_length}). Cannot find interval.")
        return None, 0

    try:
        numeric_data = pd.to_numeric(data, errors='coerce')
        if numeric_data.isnull().any():
            print(f"Warning: Non-numeric values found in {change_col_name} after coercion. Changes might be inaccurate.")
    except Exception as e:
        print(f"Warning: Could not convert {change_col_name} to numeric. Error: {e}. Change detection may fail.")
        numeric_data = data 

    changes = numeric_data.diff().abs()
    changes = changes.fillna(0)
    change_points = changes > 1e-6 
    max_changes = -1
    best_start_index = None

    if interval_length <= 0: return None, 0

    rolling_sum_changes = change_points.rolling(window=interval_length, min_periods=1).sum() 

    if not rolling_sum_changes.empty and rolling_sum_changes.max() > 0:
        end_index = rolling_sum_changes.idxmax()
        max_changes = int(rolling_sum_changes.loc[end_index])
        best_start_index = end_index - interval_length + 1
        best_start_index = max(0, best_start_index) 
    else:
         print(f"Warning: Could not find any intervals with changes in '{change_col_name}'. Defaulting to start.")
         best_start_index = 0
         max_changes = 0

    if best_start_index is not None and best_start_index + interval_length > len(data):
        print(f"Warning: Calculated best interval [{best_start_index}, {best_start_index+interval_length-1}] exceeds data length {len(data)}. Adjusting.")
        best_start_index = max(0, len(data) - interval_length)
        max_changes = int(change_points.iloc[best_start_index : best_start_index + interval_length].sum())

    if best_start_index is None:
        print("Warning: No valid interval start index determined. Defaulting to 0.")
        best_start_index = 0
        max_changes = 0

    return best_start_index, max_changes


def main():
    args = parse_arguments()

    metrics_folder = "test_metrics"
    plots_folder = "test_plots"
    os.makedirs(metrics_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    print(f"Using model: {model_name}")

    print("Loading and preprocessing data...")
    try:
        df, basic_columns, _ = load_and_preprocess_data(args)
        _, _, test_scaled_df, scaler = split_and_scale_data(df, basic_columns)
        print(f"Test data shape: {test_scaled_df.shape}")
    except Exception as e:
        print(f"Error during data loading/processing: {e}")
        sys.exit(1)

    if args.fluid_change_col not in test_scaled_df.columns:
         print(f"Error: Fluid change column '{args.fluid_change_col}' not found in scaled test data.")
         print(f"Available columns: {test_scaled_df.columns.tolist()}")
         sys.exit(1)

    print(f"Finding {args.interval_length} timestep interval with most '{args.fluid_change_col}' changes within the test set, ensuring space for window_size...")
    test_fluid_col_data = test_scaled_df[args.fluid_change_col].reset_index(drop=True)
    min_start_index_for_valid_history = args.window_size

    if len(test_fluid_col_data) < min_start_index_for_valid_history + args.interval_length:
        print(f"Error: Test data (length {len(test_fluid_col_data)}) is too short...") 
        sys.exit(1)

    search_data = test_fluid_col_data[min_start_index_for_valid_history:]
    interval_start_relative_to_search, num_changes = find_most_active_interval(
        search_data.reset_index(drop=True),
        args.interval_length,
        args.fluid_change_col
    )

    if interval_start_relative_to_search is None:
         print("Could not identify a suitable interval within the valid search range. Exiting.")
         sys.exit(1)

    interval_start_in_test = min_start_index_for_valid_history + interval_start_relative_to_search
    print(f"Found interval starting at test set index: {interval_start_in_test} with {num_changes} changes.")

    required_data_length_for_pred = interval_start_in_test + args.interval_length
    if len(test_scaled_df) < required_data_length_for_pred:
         print(f"Error: Test data length ({len(test_scaled_df)}) is insufficient...") 
         sys.exit(1)

    x_data_full_test = test_scaled_df[args.input_features].values
    y_data_full_test = test_scaled_df[args.target_feature].values

    print(f"Identifying fluid switches within interval [{interval_start_in_test}, {interval_start_in_test + args.interval_length - 1}]...")
    switch_indices_in_interval = []
    try:
        interval_fluid_data = test_scaled_df[args.fluid_change_col].iloc[
            interval_start_in_test : interval_start_in_test + args.interval_length
        ].reset_index(drop=True) 

        changes = interval_fluid_data.diff().abs()
        switch_points = changes > 1e-6
        switch_indices_in_interval = switch_points[switch_points].index.tolist()

        print(f"Found {len(switch_indices_in_interval)} switch points within the interval at indices (relative to interval start): {switch_indices_in_interval[:10]}...") 
    except Exception as e:
        print(f"Warning: Could not identify fluid switch points for plotting. Error: {e}")
        switch_indices_in_interval = [] 

    print("Setting up the model...")
    input_dim = len(args.input_features)
    model = create_model(args.model_type, input_dim, args.hidden_dim, args.num_layers, args.horizon)
    model.to(device)
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    except FileNotFoundError: sys.exit(1)
    except Exception as e: print(f"Error loading model state_dict: {e}"); sys.exit(1)

    print(f"Performing sequential forecast on interval: [{interval_start_in_test}, {interval_start_in_test + args.interval_length - 1}] (relative to test set start)")
    all_preds_interval = []
    all_targets_interval = []
    num_segments_interval = (args.interval_length + args.horizon - 1) // args.horizon
    print(f"Total timesteps to predict: {args.interval_length}, Segments: {num_segments_interval}")

    for seg in range(num_segments_interval):
        current_pred_start_abs_idx = interval_start_in_test + seg * args.horizon
        remaining_interval = args.interval_length - (seg * args.horizon)
        current_horizon = min(args.horizon, remaining_interval)
        if current_horizon <= 0: break
        hist_data_start_abs_idx = current_pred_start_abs_idx - args.window_size
        future_data_end_abs_idx = current_pred_start_abs_idx + current_horizon

        if future_data_end_abs_idx > len(x_data_full_test):
             print(f"\nWarning: Ran out of test data for segment {seg}...") 
             current_horizon = len(x_data_full_test) - current_pred_start_abs_idx
             future_data_end_abs_idx = len(x_data_full_test)
             if current_horizon <= 0: break

        print(f"Segment {seg + 1}/{num_segments_interval}: Predicting {current_horizon} steps...") 

        try: 
            if args.model_type in ["lstm_aug", "encdec_lstm_aug"]:
                 x_hist_features = x_data_full_test[hist_data_start_abs_idx : current_pred_start_abs_idx]; x_hist_target = y_data_full_test[hist_data_start_abs_idx : current_pred_start_abs_idx]; x_hist = np.concatenate([x_hist_features, x_hist_target.reshape(-1, 1)], axis=1); x_dec = x_data_full_test[current_pred_start_abs_idx : future_data_end_abs_idx]; x_hist_tensor = torch.tensor(x_hist, dtype=torch.float32).unsqueeze(0).to(device); x_dec_tensor = torch.tensor(x_dec, dtype=torch.float32).unsqueeze(0).to(device)
                 with torch.no_grad(): pred = model(x_hist_tensor, x_dec_tensor)
            else: 
                 sample_x = x_data_full_test[hist_data_start_abs_idx : future_data_end_abs_idx]; x_tensor = torch.tensor(sample_x, dtype=torch.float32).unsqueeze(0).to(device)
                 with torch.no_grad(): full_pred = model(x_tensor)
                 if full_pred.dim() == 2: pred = full_pred[:, -current_horizon:]
                 elif full_pred.dim() == 3: pred = full_pred[:, -current_horizon:, 0]
                 else: raise ValueError(f"Unexpected model output dimension: {full_pred.dim()}...")

            pred = pred.cpu().numpy().flatten()
            if len(pred) > current_horizon: pred = pred[:current_horizon]

        except Exception as e: print(f"\nError during model prediction for segment {seg}: {e}"); sys.exit(1)

        sample_y = y_data_full_test[current_pred_start_abs_idx : future_data_end_abs_idx]
        all_preds_interval.append(pred)
        all_targets_interval.append(sample_y)

    if not all_preds_interval: print("No predictions generated."); sys.exit(1)
    all_preds = np.concatenate(all_preds_interval, axis=0)[:args.interval_length]
    all_targets = np.concatenate(all_targets_interval, axis=0)[:args.interval_length]
    print(f"Generated {len(all_preds)} predictions for the interval.")

    all_preds_inv = inverse_transform_ecd(all_preds.reshape(1, -1), scaler, basic_columns)
    all_targets_inv = inverse_transform_ecd(all_targets.reshape(1, -1), scaler, basic_columns)

    mse_val = mean_squared_error(all_targets_inv.flatten(), all_preds_inv.flatten())
    rmse_val = root_mean_squared_error(all_targets_inv.flatten(), all_preds_inv.flatten())
    mae_val = mean_absolute_error(all_targets_inv.flatten(), all_preds_inv.flatten())

    metrics = { 
        "model_name": model_name, "interval_analyzed": True, "interval_start_in_test": interval_start_in_test,
        "interval_length": args.interval_length, "interval_fluid_changes": num_changes,
        "interval_switches_plotted": len(switch_indices_in_interval), 
        "mse": mse_val, "rmse": rmse_val, "mae": mae_val
    }
    json_out_filename = f"{model_name}{args.output_suffix}.json"
    json_out_path = os.path.join(metrics_folder, json_out_filename)
    results_dict = {model_name + "_interval": metrics}
    try:
        with open(json_out_path, "w") as f: json.dump(results_dict, f, indent=4)
        print("\nEvaluation Metrics:")
        print(json.dumps(metrics, indent=4))
        print(f"Saved metrics to: {json_out_path}")
    except Exception as e: print(f"Error saving metrics to JSON: {e}")

    plot_out_filename = f"{model_name}{args.output_suffix}.png"
    plot_out_path = os.path.join(plots_folder, plot_out_filename)
    timesteps = np.arange(len(all_targets_inv.flatten()))
    
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })

    plt.figure(figsize=(12, 8))

    plt.plot(timesteps, all_targets_inv.flatten(), label="Actual ECD", marker=".", linestyle='-', linewidth=2, markersize=5)
    plt.plot(timesteps, all_preds_inv.flatten(), label="Predicted ECD", linestyle='-', linewidth=2, alpha=0.8)

    added_switch_label = False
    if switch_indices_in_interval:
        print(f"Adding {len(switch_indices_in_interval)} vertical lines for fluid switches.")
        for switch_idx in switch_indices_in_interval:
            if 0 <= switch_idx < len(timesteps):
                label = ""
                if not added_switch_label:
                    label = 'Fluid Switch'
                    added_switch_label = True
                plt.axvline(x=switch_idx, color='black', linestyle='--', linewidth=1.1, label=label, alpha=0.9)
            else:
                print(f"Warning: Switch index {switch_idx} out of bounds for plotting [0, {len(timesteps)-1}]")

    plt.xlabel(f"Time Step within Interval (starting at test index {interval_start_in_test})")
    plt.ylabel("ECD")
    plt.title(f"ECD Forecast on High Fluid Change Interval ({args.interval_length} steps, {len(switch_indices_in_interval)} switches in interval)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(plot_out_path, dpi=300, bbox_inches='tight')
        print(f"Saved forecast plot to: {plot_out_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()
    
    cross_plots_folder     = "cross_plots"
    residual_plots_folder  = "residual_plots"
    os.makedirs(cross_plots_folder,    exist_ok=True)
    os.makedirs(residual_plots_folder, exist_ok=True)

    base_fname = f"{model_name}{args.output_suffix}"
    cross_plot_path = os.path.join(cross_plots_folder, f"{base_fname}_cross.png")

    plt.figure(figsize=(8, 8))
    plt.scatter(
        all_targets_inv.flatten(), 
        all_preds_inv.flatten(),
        s=8, alpha=0.6, edgecolor="none"
    )

    lo = min(all_targets_inv.min(), all_preds_inv.min())
    hi = max(all_targets_inv.max(), all_preds_inv.max())
    plt.plot([lo, hi], [lo, hi], "--k", linewidth=1.0)

    plt.xlabel("Actual ECD")
    plt.ylabel("Predicted ECD")
    plt.title("Cross-plot: Predicted vs. Actual ECD")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    try:
        plt.savefig(cross_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved cross-plot to: {cross_plot_path}")
    except Exception as e:
        print(f"Error saving cross-plot: {e}")
    plt.close()

    residual_plot_path = os.path.join(residual_plots_folder, f"{base_fname}_residuals.png")
    residuals = all_targets_inv.flatten() - all_preds_inv.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, density=True, alpha=0.7)

    import seaborn as sns
    sns.kdeplot(residuals, bw_adjust=1.2, linewidth=2)
    plt.legend(["KDE", "Histogram"])

    plt.xlabel("Residual (Actual – Predicted)")
    plt.ylabel("Density")
    plt.title("Residuals Distribution")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    try:
        plt.savefig(residual_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved residual-distribution plot to: {residual_plot_path}")
    except Exception as e:
        print(f"Error saving residual plot: {e}")
    plt.close()

    hexbin_path = os.path.join(cross_plots_folder, f"{base_fname}_cross_hexbin.png")

    plt.figure(figsize=(8, 8))
    hb = plt.hexbin(
        all_targets_inv.flatten(),
        all_preds_inv.flatten(),
        gridsize=80,          
        norm=LogNorm(),       
        cmap="viridis"
    )

    plt.plot([lo, hi], [lo, hi], "--k", linewidth=1.0)

    cb = plt.colorbar(hb)
    cb.set_label("Point count (log scale)")

    plt.xlabel("Actual ECD")
    plt.ylabel("Predicted ECD")
    plt.title("Cross-plot (hexbin density)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(hexbin_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved hexbin cross-plot to: {hexbin_path}")
    
    resid_v_actual_path = os.path.join(residual_plots_folder,
                                   f"{base_fname}_resid_vs_actual.png")

    plt.figure(figsize=(10,6))
    plt.scatter(all_targets_inv.flatten(),
                residuals,
                alpha=0.05, s=4)

    plt.axhline(0, color="k", linewidth=1)
    plt.xlabel("Actual ECD")
    plt.ylabel("Residual (Actual – Predicted)")
    plt.title("Residuals vs. Actual ECD")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(resid_v_actual_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residual-vs-actual plot to: {resid_v_actual_path}")
                       

    diagnostic_plots_folder = "diagnostic_plots_fluid"
    os.makedirs(diagnostic_plots_folder, exist_ok=True)

    combo_path = os.path.join(
        diagnostic_plots_folder, f"{base_fname}_cross_and_residuals.png"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_hex, ax_hist = axes


    lo = min(all_targets_inv.min(), all_preds_inv.min())
    hi = max(all_targets_inv.max(), all_preds_inv.max())

    hb = ax_hex.hexbin(
        all_targets_inv.flatten(),
        all_preds_inv.flatten(),
        gridsize=80,
        norm=LogNorm(),
        cmap="viridis"
    )
    ax_hex.plot([lo, hi], [lo, hi], "--k", linewidth=1)

    cb = fig.colorbar(hb, ax=ax_hex)
    cb.set_label("Point count (log scale)")

    ax_hex.set_xlabel("Actual ECD")
    ax_hex.set_ylabel("Predicted ECD")
    ax_hex.set_title("Predicted vs. Actual (hexbin)")
    ax_hex.grid(True, linestyle=":", alpha=0.5)

    ax_hist.hist(residuals, bins=50, density=True, alpha=0.7, label="Histogram")
    sns.kdeplot(residuals, bw_adjust=1.2, linewidth=2, ax=ax_hist, label="KDE")

    ax_hist.set_xlabel("Residual (Actual – Predicted)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Residuals Distribution")
    ax_hist.grid(True, linestyle=":", alpha=0.6)
    ax_hist.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])   

    try:
        fig.savefig(combo_path, dpi=300, bbox_inches="tight")
        print(f"Saved combined diagnostic plot to: {combo_path}")
    except Exception as e:
        print(f"Error saving combined diagnostic plot: {e}")
    plt.close(fig)

if __name__ == "__main__":
    main()