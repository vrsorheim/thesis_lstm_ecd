import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error

from architectures import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FutureAwareTimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, window_size, horizon, input_features, target_feature):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.input_features = input_features
        self.target_feature = target_feature

        self.inputs = self.data[input_features].values
        self.targets = self.data[target_feature].values

    def __len__(self):
        num_sequences = len(self.data) - self.window_size - self.horizon + 1
        return max(0, num_sequences)


    def __getitem__(self, idx):
        X_hist = self.inputs[idx : idx+self.window_size]
        X_fut = self.inputs[idx+self.window_size : idx+self.window_size+self.horizon]
        X = np.concatenate([X_hist, X_fut], axis=0)

        y = self.targets[idx+self.window_size : idx+self.window_size+self.horizon]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class AugmentedTimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, window_size, horizon, input_features, target_feature, ecd_noise_std):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.input_features = input_features
        self.target_feature = target_feature
        self.ecd_noise_std = ecd_noise_std

        self.inputs = self.data[input_features].values
        self.targets = self.data[target_feature].values

    def __len__(self):
        num_sequences = len(self.data) - self.window_size - self.horizon + 1
        return max(0, num_sequences)

    def __getitem__(self, idx):
        hist_slice = slice(idx, idx + self.window_size)
        fut_slice  = slice(idx + self.window_size, idx + self.window_size + self.horizon)

        x_hist_features = self.inputs[hist_slice]
        x_hist_target = self.targets[hist_slice]

        if self.ecd_noise_std > 0:
            noise = np.random.randn(self.window_size) * self.ecd_noise_std
            x_hist_target = x_hist_target + noise

        x_hist_target_reshaped = x_hist_target.reshape(-1, 1)

        if x_hist_features.shape[0] != x_hist_target_reshaped.shape[0]:
             raise ValueError(f"Shape mismatch: Features {x_hist_features.shape}, Target {x_hist_target_reshaped.shape}")

        x_hist = np.concatenate([x_hist_features, x_hist_target_reshaped], axis=1)

        x_dec = self.inputs[fut_slice]
        y = self.targets[fut_slice]

        return (
            torch.tensor(x_hist, dtype=torch.float32),
            torch.tensor(x_dec, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


def train_model(model_type, model, train_loader, val_loader, optimizer, criterion, num_epochs, device, patience, save_path, augmented_encoder):
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if augmented_encoder:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x_enc, x_dec, y = batch
                    x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                    y_pred = model(x_enc, x_dec)
                else:
                    raise ValueError(f"Unexpected batch format in augmented encoder case. Expected 3 items, got {len(batch)}")
            else: 
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                else:
                     raise ValueError(f"Unexpected batch format in standard case. Expected 2 items, got {len(batch)}")

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * y.size(0)
        epoch_train_loss /= len(train_loader.dataset)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if augmented_encoder:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        x_enc, x_dec, y_val = batch
                        x_enc, x_dec, y_val = x_enc.to(device), x_dec.to(device), y_val.to(device)
                        y_pred_val = model(x_enc, x_dec)
                    else:
                         raise ValueError(f"Unexpected batch format in augmented encoder validation. Expected 3 items, got {len(batch)}")
                else:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        X_val, y_val = batch
                        X_val, y_val = X_val.to(device), y_val.to(device)
                        y_pred_val = model(X_val)
                    else:
                        raise ValueError(f"Unexpected batch format in standard validation. Expected 2 items, got {len(batch)}")

                loss_val = criterion(y_pred_val, y_val)
                epoch_val_loss += loss_val.item() * y_val.size(0)
        epoch_val_loss /= len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"Model improved. Saving checkpoint to {save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                break

    return train_losses, val_losses


def evaluate_model(model_type, model, loader, criterion, device, augmented_encoder):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if augmented_encoder:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x_enc, x_dec, y_test = batch
                    x_enc, x_dec, y_test = x_enc.to(device), x_dec.to(device), y_test.to(device)
                    y_pred_test = model(x_enc, x_dec)
                else:
                    raise ValueError(f"Unexpected batch format in augmented encoder evaluation. Expected 3 items, got {len(batch)}")
            else:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    X_test, y_test = batch
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    y_pred_test = model(X_test)
                else:
                    raise ValueError(f"Unexpected batch format in standard evaluation. Expected 2 items, got {len(batch)}")

            loss_test = criterion(y_pred_test, y_test)
            total_loss += loss_test.item() * y_test.size(0) 
    total_loss /= len(loader.dataset)
    return total_loss

def plot_train_val_losses(train_losses, val_losses, run_name, feature_set_str): 
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{run_name}: Loss Curve (Features={feature_set_str})') 
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    os.makedirs("graphs_losses", exist_ok=True)
    plot_filename = os.path.join("graphs_losses", f"{run_name}_loss.png")
    plt.savefig(plot_filename)
    plt.close()


def train_and_evaluate(
        model_type,
        input_features,
        hidden_dim,
        num_layers,
        lr,
        run_name,
        train_df,
        val_df,
        test_df,
        window_size,
        horizon,
        num_epochs,
        batch_size,
        device,
        scaler,
        patience,
        all_columns,
        augmented_encoder,
        ecd_noise_std,
        model_path,
        **model_kwargs
    ):
    if augmented_encoder:
        train_dataset = AugmentedTimeSeriesDataset(
            data=train_df, window_size=window_size, horizon=horizon,
            input_features=input_features, target_feature='ECD_bot',
            ecd_noise_std=ecd_noise_std
        )
        val_dataset = AugmentedTimeSeriesDataset(
            data=val_df, window_size=window_size, horizon=horizon,
            input_features=input_features, target_feature='ECD_bot',
            ecd_noise_std=ecd_noise_std
        )
        test_dataset = AugmentedTimeSeriesDataset(
            data=test_df, window_size=window_size, horizon=horizon,
            input_features=input_features, target_feature='ECD_bot',
            ecd_noise_std=ecd_noise_std
        )
    else: 
        train_dataset = FutureAwareTimeSeriesDataset(
            data=train_df, window_size=window_size, horizon=horizon,
            input_features=input_features, target_feature='ECD_bot'
        )
        val_dataset   = FutureAwareTimeSeriesDataset(
            data=val_df, window_size=window_size, horizon=horizon,
            input_features=input_features, target_feature='ECD_bot'
        )
        test_dataset  = FutureAwareTimeSeriesDataset(
            data=test_df, window_size=window_size, horizon=horizon,
            input_features=input_features, target_feature='ECD_bot'
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4
    )

    input_dim = len(input_features)
    model = create_model(model_type, input_dim, hidden_dim, num_layers, horizon, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\n=== Starting run: {run_name} ===")
    print(f"Model: {model_type}, Features: {input_features}, Hidden: {hidden_dim}, "
          f"Layers: {num_layers}, LR: {lr}, Augmented: {augmented_encoder}")

    os.makedirs(model_path, exist_ok=True)
    model_filename = os.path.join(model_path, f"{run_name}.pt")
    train_losses, val_losses = train_model(
        model_type=model_type,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        patience=patience,
        save_path=model_filename,
        augmented_encoder=augmented_encoder
    )

    feature_set_str = ','.join(input_features) if isinstance(input_features, list) else str(input_features)
    plot_train_val_losses(train_losses, val_losses, run_name, feature_set_str)

    try:
        model.load_state_dict(torch.load(model_filename, map_location=device))
    except FileNotFoundError:
        print(f"Warning: Best model checkpoint {model_filename} not found. Evaluating model with last state.")

    test_loss = evaluate_model(model_type, model, test_loader, criterion, device, augmented_encoder)
    print(f"{run_name} - Test MSE Loss (scaled): {test_loss:.4f}")

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            if augmented_encoder:
                x_hist, x_dec, y_scaled = batch
                x_hist = x_hist.to(device)
                x_dec  = x_dec.to(device)
                y_scaled = y_scaled.to(device)
                y_pred_scaled = model(x_hist, x_dec)
            else: 
                X_scaled, y_scaled = batch
                X_scaled = X_scaled.to(device)
                y_scaled = y_scaled.to(device)
                y_pred_scaled = model(X_scaled)

            all_preds.append(y_pred_scaled.cpu().numpy())
            all_targets.append(y_scaled.cpu().numpy())

    if not all_preds:
         print(f"Warning: No predictions generated for run {run_name}")
         return {
             "train_losses": [float(x) for x in train_losses],
             "val_losses":   [float(x) for x in val_losses],
             "test_loss":    float('nan'),
             "mse":          float('nan'),
             "rmse":         float('nan'),
             "mae":          float('nan'),
         }


    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if scaler is None or all_columns is None:
        raise ValueError("Scaler or all_columns not provided for inverse transformation.")

    try:
        all_preds_inv = inverse_transform_ecd(all_preds, scaler, all_columns=all_columns)
        all_targets_inv = inverse_transform_ecd(all_targets, scaler, all_columns=all_columns)
    except Exception as e:
        print(f"Error during inverse transform for run {run_name}: {e}")
        mse_val, rmse_val, mae_val = float('nan'), float('nan'), float('nan')
    else:
        mse_val = mean_squared_error(all_targets_inv, all_preds_inv)
        rmse_val = root_mean_squared_error(all_targets_inv, all_preds_inv)
        mae_val = mean_absolute_error(all_targets_inv, all_preds_inv)
        print(f"{run_name} Metrics (Original Scale): MSE={mse_val:.4f}, RMSE={rmse_val:.4f}, MAE={mae_val:.4f}")
    return {
        "train_losses": [float(x) for x in train_losses],
        "val_losses":   [float(x) for x in val_losses],
        "test_loss":    float(test_loss),
        "mse":          float(mse_val),
        "rmse":         float(rmse_val),
        "mae":          float(mae_val),
    }


def inverse_transform_ecd(scaled_ecd_values, scaler, all_columns):
    try:
        ecd_index = all_columns.index("ECD_bot")
    except ValueError:
        raise ValueError(f"'ECD_bot' not found in provided all_columns: {all_columns}")

    num_cols = len(all_columns)
    if scaler.n_features_in_ != num_cols:
         raise ValueError(f"Scaler was fitted on {scaler.n_features_in_} features, but all_columns has {num_cols} features.")


    if scaled_ecd_values.ndim == 1:
        scaled_ecd_values = scaled_ecd_values.reshape(-1, 1)

    n_samples, n_horizon = scaled_ecd_values.shape
    dummy_array_shape = (n_samples * n_horizon, num_cols)
    dummy_flat = np.zeros(dummy_array_shape)

    scaled_midpoint = 0.5
    for col_idx in range(num_cols):
         if col_idx != ecd_index:
              dummy_flat[:, col_idx] = scaled_midpoint

    dummy_flat[:, ecd_index] = scaled_ecd_values.flatten()
    inv_flat = scaler.inverse_transform(dummy_flat)
    
    original_ecd_flat = inv_flat[:, ecd_index]
    original_ecd = original_ecd_flat.reshape(n_samples, n_horizon)

    return original_ecd