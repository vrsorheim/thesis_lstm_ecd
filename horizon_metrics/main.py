import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_eval import (
    AugmentedTimeSeriesDataset,
    FutureAwareTimeSeriesDataset,
    device,
    inverse_transform_ecd
)
from utils import (
    load_and_preprocess_data,
    split_and_scale_data
)
from architectures import create_model 

def plot_horizon_metrics(metrics_dict, horizon, plot_folder):
    os.makedirs(plot_folder, exist_ok=True)
    mse_per_horizon = metrics_dict["mse_per_horizon"]  
    mae_per_horizon = metrics_dict["mae_per_horizon"]   
    rmse_per_horizon = metrics_dict["rmse_per_horizon"] 

    steps = range(1, horizon + 1)

    plt.figure()
    plt.plot(steps, mse_per_horizon)
    plt.xlabel("Timesteps")
    plt.ylabel("MSE (real units)")
    plt.title("Horizon-wise MSE")
    plt.grid(False)
    plt.tight_layout()  
    mse_path = os.path.join(plot_folder, "horizon_wise_mse.png")
    plt.savefig(mse_path)
    plt.close()

    plt.figure()
    plt.plot(steps, mae_per_horizon)
    plt.xlabel("Timesteps")
    plt.ylabel("MAE (real units)")
    plt.title("Horizon-wise MAE")
    plt.grid(False)
    plt.tight_layout()  
    mae_path = os.path.join(plot_folder, "horizon_wise_mae.png")
    plt.savefig(mae_path)
    plt.close()

    plt.figure()
    plt.plot(steps, rmse_per_horizon)
    plt.xlabel("Timesteps")
    plt.ylabel("RMSE (real units)")
    plt.title("Horizon-wise RMSE")
    plt.grid(False)
    plt.tight_layout()  
    rmse_path = os.path.join(plot_folder, "horizon_wise_rmse.png")
    plt.savefig(rmse_path)
    plt.close()

    plt.figure()
    plt.plot(steps, mse_per_horizon, label="MSE")
    plt.plot(steps, mae_per_horizon, label="MAE")
    plt.plot(steps, rmse_per_horizon, label="RMSE")
    plt.xlabel("Timesteps")
    plt.ylabel("Metric Value (real units)")
    plt.title("Horizon-wise MSE, MAE, RMSE (Combined)")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()  
    combined_path = os.path.join(plot_folder, "horizon_wise_metrics_combined.png")
    plt.savefig(combined_path)
    plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute horizon-wise metrics for a previously trained model."
    )
    parser.add_argument("--run_name", type=str, required=True,
                        help="Directory where trained model checkpoints are stored.")
    parser.add_argument("--model_type", type=str, default="lstm",
                        help="Which model type was used")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension (e.g., LSTM hidden size)")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of LSTM/transformer/CNN layers")
    parser.add_argument("--horizon", type=int, default=360,
                        help="Number of future steps predicted.")
    parser.add_argument("--window_size", type=int, default=720,
                        help="Number of historical steps.")
    parser.add_argument("--ecd_noise_std", type=float, default=0.0,
                        help="Gaussian noise standard deviation used for augmented encoders.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for test loading.")
    parser.add_argument("--input_features", nargs="+", default=["Rate_in"],
                        help="List of feature columns used during training.")
    parser.add_argument("--augmented_encoder", action="store_true",
                        help="Whether the model used the augmented encoder approach.")
    parser.add_argument("--data_file", type=str, default="../input_files/case_1_rot_drill_fluid/TimeSeries.out",
                        help="The same data file used in training.")
    
    parser.add_argument("--in_file", type=str, default="../input_files/case_1_rot_drill_fluid/Case_5m_rot_operation.in",
                        help="Path to the fluid/operation.in file for merging")
    
    parser.add_argument("--fluid", action="store_true", help="If set, run with fluid changes")
    parser.add_argument("--p_fluid", action="store_true", help="If set, run with fluid percentages")
    parser.add_argument("--ds_a_fluid", action="store_true",
                        help="If set, run with fluid percentages on ds and annulus too")
    
    parser.add_argument("--plot_folder", type=str, default="plots")
    return parser.parse_args()

def compute_horizon_metrics_for_run(run_name, args):

    df, basic_columns, _= load_and_preprocess_data(args)
    _, _, test_df, scaler = split_and_scale_data(df, basic_columns)

    if args.augmented_encoder:
        test_dataset = AugmentedTimeSeriesDataset(
            data=test_df,
            window_size=args.window_size,
            horizon=args.horizon,
            input_features=args.input_features,
            target_feature='ECD_bot',
            ecd_noise_std=args.ecd_noise_std
        )
    else:
        test_dataset = FutureAwareTimeSeriesDataset(
            data=test_df,
            window_size=args.window_size,
            horizon=args.horizon,
            input_features=args.input_features,
            target_feature='ECD_bot'
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    input_dim = len(args.input_features)
    model = create_model(
        model_type=args.model_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        horizon=args.horizon
    )
    model.to(device)

    checkpoint_path = args.run_name
    print(f"Loading model weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    metrics_dict = compute_horizon_wise_metrics(
        model_type=args.model_type,
        model=model,
        loader=test_loader,
        device=device,
        scaler=scaler,
        all_columns=basic_columns,
        augmented_encoder=args.augmented_encoder
    )
    plot_horizon_metrics(
        metrics_dict=metrics_dict,
        horizon=args.horizon,
        plot_folder=args.plot_folder
    )

def compute_horizon_wise_metrics(
    model_type,
    model,
    loader,
    device,
    scaler,
    all_columns,
    augmented_encoder=False
):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if augmented_encoder:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    if model_type in ['lstm', 'tcn']:
                        x, y = batch
                        x, y = x.to(device), y.to(device)
                        y_pred = model(x)
                        y_true = y
                    else:
                        x_enc, x_dec, y = batch
                        x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                        y_pred = model(x_enc, x_dec)
                        y_true = y
                else:
                    raise ValueError("Unexpected batch format in augmented encoder case.")

            else:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    y_true = y
                elif isinstance(batch, (list, tuple)) and len(batch) == 4:
                    x_hist, x_dec, y, lengths = batch
                    x_hist, x_dec, y = x_hist.to(device), x_dec.to(device), y.to(device)
                    y_pred = model(x_hist, x_dec, lengths)
                    y_true = y
                else:
                    raise ValueError("Unexpected batch format in standard case.")
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    N = all_preds.shape[0]
    H = all_preds.shape[1]
    print(f"Number of test samples: {N}")
    print(f"Horizon length: {H}")

    all_preds_inv = inverse_transform_ecd(all_preds, scaler, all_columns)
    all_targets_inv = inverse_transform_ecd(all_targets, scaler, all_columns)

    mse_per_horizon = []
    mae_per_horizon = []
    rmse_per_horizon = []

    for h in range(H):
        pred_h = all_preds_inv[:, h]
        true_h = all_targets_inv[:, h]
        
        mse_h = mean_squared_error(true_h, pred_h)
        mae_h = mean_absolute_error(true_h, pred_h)
        rmse_h = root_mean_squared_error(true_h, pred_h)

        mse_per_horizon.append(mse_h)
        mae_per_horizon.append(mae_h)
        rmse_per_horizon.append(rmse_h)

    metrics_dict = {
        "mse_per_horizon": mse_per_horizon,
        "mae_per_horizon": mae_per_horizon,
        "rmse_per_horizon": rmse_per_horizon
    }

    return metrics_dict

def main():
    args = parse_arguments()
    compute_horizon_metrics_for_run(args.run_name, args)

if __name__ == "__main__":
    main()
