import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.colors import LogNorm   
import seaborn as sns                    
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_and_preprocess_data, split_and_scale_data
from architectures import create_model
from train_eval import device, inverse_transform_ecd
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run sequential forecast with a variable number of timesteps. "
                    "Specify how many timesteps to test or use the whole test dataset."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved model checkpoint (e.g., models/my_run.pt)")
    parser.add_argument("--data_file", type=str, default="..\input_files\case_1_rot_drill_fluid\TimeSeries.out",
                        help="Path to the time series data file")
    parser.add_argument("--window_size", type=int, default=720,
                        help="Historical window size")
    parser.add_argument("--horizon", type=int, default=360,
                        help="Forecast horizon per segment")
    parser.add_argument("--segments", type=int, default=14,
                        help="Number of forecast segments to roll out (used if --test_timesteps is not provided and --test_all is not set)")
    parser.add_argument("--test_timesteps", type=int, default=None,
                        help="Total number of timesteps to forecast. Overrides segments*horizon if provided.")
    parser.add_argument("--test_all", action="store_true",
                        help="If set, test the entire test dataset (after the window) as forecast.")
    parser.add_argument("--model_type", type=str, default="lstm",
                        help="Model type (e.g., lstm, cnn, lstm_aug, encdec_lstm_aug, etc.)")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of layers")
    parser.add_argument("--input_features", nargs="+", default=["Rate_in"],
                        help="List of input feature columns")
    parser.add_argument("--target_feature", type=str, default="ECD_bot",
                        help="Name of the target feature (for inverse scaling)")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="Kernel size (if applicable)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (if applicable)")
    parser.add_argument("--json_out", type=str, default="rolling_forecast_metrics.json",
                        help="Filename to save computed metrics as JSON")
    parser.add_argument("--in_file", type=str, default="..\input_files/case_1_rot_drill_fluid/Case_5m_rot_operation.in",
                        help="Path to the fluid/operation.in file for merging")
    
    parser.add_argument("--fluid", action="store_true", help="If set, run with fluid changes")
    parser.add_argument("--p_fluid", action="store_true", help="If set, run with fluid percentages")
    parser.add_argument("--ds_a_fluid", action="store_true",
                        help="If set, run with fluid percentages on ds and annulus too")
    
    parser.add_argument("--augmented_encoder", action="store_true", help="If sert, run with augmented ECD in hist input")
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    metrics_folder = "test_metrics"
    plots_folder = "test_plots"
    os.makedirs(metrics_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    df, basic_columns, _= load_and_preprocess_data(args)
    _, _, test_scaled, scaler = split_and_scale_data(df, basic_columns)
    x_data = test_scaled[args.input_features].values
    y_data = test_scaled[args.target_feature].values
    total_data = len(x_data)
    if args.test_all:
        test_forecast_timesteps = total_data - args.window_size
    elif args.test_timesteps is not None:
        test_forecast_timesteps = args.test_timesteps
    else:
        test_forecast_timesteps = args.segments * args.horizon

    if args.test_all:
        full_segments = test_forecast_timesteps // args.horizon
        remainder = test_forecast_timesteps % args.horizon
        segments = full_segments + (1 if remainder > 0 else 0)
    else:
        segments = (test_forecast_timesteps + args.horizon - 1) // args.horizon

    plot_out_path = os.path.join(plots_folder, f"{model_name}_{test_forecast_timesteps}ts_forecast.png")
    json_out_path = os.path.join(metrics_folder, args.json_out)
    if os.path.exists(json_out_path):
        with open(json_out_path, "r") as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    if model_name in results_dict and results_dict[model_name].get("tested_timesteps") == test_forecast_timesteps:
        print(f"Model '{model_name}' has already been tested for {test_forecast_timesteps} timesteps. Skipping this test.")
        sys.exit(0)

    required_data = args.window_size + (segments - 1) * args.horizon
    if total_data < required_data:
        raise ValueError(f"Test data length ({total_data}) is less than required ({required_data}) for the rolling forecast.")

    input_dim = len(args.input_features)
    extra_kwargs = {"kernel_size": args.kernel_size, "dropout": args.dropout}
    model = create_model(args.model_type, input_dim, args.hidden_dim, args.num_layers, args.horizon, **extra_kwargs)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    all_preds_segments = []
    all_targets_segments = []
    for seg in range(segments):
        start_idx = seg * args.horizon
        if args.test_all and (seg == segments - 1) and ((total_data - args.window_size) % args.horizon != 0):
            forecast_horizon = (total_data - args.window_size) % args.horizon
        else:
            forecast_horizon = args.horizon

        sample_start = start_idx
        sample_end = start_idx + args.window_size + forecast_horizon
        if sample_end > total_data:
            break

        if args.model_type in ["lstm_aug", "encdec_lstm_aug"]:
            x_hist_features = x_data[sample_start : sample_start + args.window_size]
            x_hist_target = y_data[sample_start : sample_start + args.window_size]
            x_hist = np.concatenate([x_hist_features, x_hist_target.reshape(-1, 1)], axis=1)  
            x_dec = x_data[sample_start + args.window_size : sample_end]  
            x_hist_tensor = torch.tensor(x_hist, dtype=torch.float32).unsqueeze(0).to(device)
            x_dec_tensor = torch.tensor(x_dec, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_hist_tensor, x_dec_tensor)
            pred = pred.cpu().numpy().squeeze(0)
        else:
            sample_x = x_data[sample_start:sample_end]
            x_tensor = torch.tensor(sample_x, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_tensor)
            pred = pred.cpu().numpy().squeeze(0)[:forecast_horizon]

        sample_y = y_data[sample_start + args.window_size : sample_end]
        all_preds_segments.append(pred)
        all_targets_segments.append(sample_y)

    all_preds = np.concatenate(all_preds_segments, axis=0).reshape(1, -1)[:, :test_forecast_timesteps]
    all_targets = np.concatenate(all_targets_segments, axis=0).reshape(1, -1)[:, :test_forecast_timesteps]

    all_preds_inv = inverse_transform_ecd(all_preds, scaler, basic_columns)
    all_targets_inv = inverse_transform_ecd(all_targets, scaler, basic_columns)

    mse_val = mean_squared_error(all_targets_inv.flatten(), all_preds_inv.flatten())
    rmse_val = root_mean_squared_error(all_targets_inv.flatten(), all_preds_inv.flatten())
    mae_val = mean_absolute_error(all_targets_inv.flatten(), all_preds_inv.flatten())
    metrics = {
        "model_name": model_name,
        "tested_timesteps": test_forecast_timesteps,
        "segments_used": segments,
        "mse": mse_val,
        "rmse": rmse_val,
        "mae": mae_val
    }
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

    results_dict[model_name] = metrics
    with open(json_out_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Evaluation Metrics:")
    print(metrics)
    print(f"Saved metrics to: {json_out_path}")

    timesteps = np.arange(1, test_forecast_timesteps + 1)
    plt.figure(figsize=(14, 6))
    plt.plot(timesteps, all_targets_inv.flatten(), label="Actual ECD", marker=".", linestyle='-', linewidth=2, markersize=5)
    plt.plot(timesteps, all_preds_inv.flatten(), label="Predicted ECD", linestyle='-', linewidth=2, alpha=0.8)
    plt.xlabel("Time Step")
    plt.ylabel("ECD")
    plt.title(f"Continuous Forecast ({test_forecast_timesteps} timesteps) via {segments} sequential evaluations")
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

    base_fname = f"{model_name}_{test_forecast_timesteps}ts"  

    hexbin_path = os.path.join(
        cross_plots_folder, f"{base_fname}_cross_hexbin.png"
    )

    plt.figure(figsize=(8, 8))

    lo = min(all_targets_inv.min(), all_preds_inv.min())
    hi = max(all_targets_inv.max(), all_preds_inv.max())

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
    plt.title("Cross-plot: Predicted vs. Actual ECD")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    try:
        plt.savefig(hexbin_path, dpi=300, bbox_inches="tight")
        print(f"Saved hexbin cross-plot to: {hexbin_path}")
    except Exception as e:
        print(f"Error saving hexbin cross-plot: {e}")
    plt.close()

    residuals = all_targets_inv.flatten() - all_preds_inv.flatten()
    hist_path = os.path.join(
        residual_plots_folder, f"{base_fname}_residuals_hist.png"
    )

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
        plt.savefig(hist_path, dpi=300, bbox_inches="tight")
        print(f"Saved residual-distribution plot to: {hist_path}")
    except Exception as e:
        print(f"Error saving residual-distribution plot: {e}")
    plt.close()

    diagnostic_plots_folder = "diagnostic_plots"
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
