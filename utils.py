from collections import deque
import pandas as pd
import os
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def add_time_since_switch(df: pd.DataFrame, fluid_col: str, new_col: str) -> pd.DataFrame:
    if fluid_col not in df.columns:
        raise ValueError(f"Column '{fluid_col}' not found in DataFrame.")
    if df.empty:
        df[new_col] = 0 
        return df
    switch = df[fluid_col].shift() != df[fluid_col]
    switch.iloc[0] = False 
    group_ids = switch.cumsum()
    df[new_col] = df.groupby(group_ids).cumcount()
    return df


def load_and_merge_data(timeseries_file, fluid_file):
    try:
        df_time = pd.read_csv(timeseries_file, sep=r"\s+")
    except FileNotFoundError:
         raise FileNotFoundError(f"TimeSeries file not found: {timeseries_file}")

    basic_columns_time = ['Time', 'Rate_in', 'SPP', 'ECD_bot', 'dsRpm', 'totVolInj']
    missing_cols_time = [col for col in basic_columns_time if col not in df_time.columns]
    if missing_cols_time:
        raise ValueError(f"Missing columns in {timeseries_file}: {missing_cols_time}")
    df_time = df_time[basic_columns_time]

    try:
        df_fluid = pd.read_csv(fluid_file, sep=r'\s+', skiprows=1, header=None) 
        expected_fluid_cols = 13 
        if df_fluid.shape[1] < expected_fluid_cols:
             raise ValueError(f"Fluid file {fluid_file} has fewer columns ({df_fluid.shape[1]}) than expected ({expected_fluid_cols}).")
        column_headers = [
            "Time_op", "Volume", "Pump rate", "Fluid name", "Density", "ROP", "BOP open",
            "Bound val 1", "Bound val 2", "Bound type 1", "Bound type 2", "DS rot", "TemperatIn"
        ]
        df_fluid.columns = column_headers[:df_fluid.shape[1]] 

    except FileNotFoundError:
         raise FileNotFoundError(f"Operation/Fluid file not found: {fluid_file}")


    basic_columns_fluid = ["Fluid name", "Density"]
    missing_cols_fluid = [col for col in basic_columns_fluid if col not in df_fluid.columns]
    if missing_cols_fluid:
        raise ValueError(f"Missing columns in {fluid_file}: {missing_cols_fluid}")

    df_fluid = df_fluid[basic_columns_fluid]

    df_fluid.rename(columns={"Fluid name": "drilling_fluid"}, inplace=True)
    
    df_time.reset_index(drop=True, inplace=True)
    df_fluid.reset_index(drop=True, inplace=True)

    if len(df_time) != len(df_fluid):
        print(f"Warning: Row count mismatch! TimeSeries file has {len(df_time)} rows, Fluid file has {len(df_fluid)} rows. Merging might be incorrect.")
        min_len = min(len(df_time), len(df_fluid))
        df_time = df_time.iloc[:min_len]
        df_fluid = df_fluid.iloc[:min_len]
    
    df_merged = pd.concat([df_time, df_fluid], axis=1)

    return df_merged

def process_fluid_data(args):
    df = load_and_merge_data(args.data_file, args.in_file)
    df = add_time_since_switch(df, fluid_col="drilling_fluid", new_col="time_since_switch")

    df = encode_binary_fluid(df, column="drilling_fluid")
    df = df[:-1]

    basic_columns = ['Rate_in', 'SPP', 'ECD_bot', 'dsRpm', 'Density', 'fluid_binary'] 
    feature_names = '_'.join(args.input_features) if isinstance(args.input_features, list) else str(args.input_features)

    if args.p_fluid and args.ds_a_fluid:
         print("Warning: Both --p_fluid and --ds_a_fluid flags set. Prioritizing --ds_a_fluid.")
         args.p_fluid = False 

    if args.p_fluid:
        df_pct = pct_fluid_tracking(df)
        df = pd.merge(df, df_pct, on='Time', how='left')
        feature_names += '_pctSimple'
        basic_columns.extend(['pct_obm1', 'pct_Case_2a1_Fluid'])
    elif args.ds_a_fluid:
        required_cols_ds = ['Time', 'drilling_fluid', 'totVolInj']
        if not all(col in df.columns for col in required_cols_ds):
             raise ValueError(f"Missing required columns for ds_a_fluid tracking: Need {required_cols_ds}")

        df_pct = track_whole_well_and_annulus_reach(df)
        if 'drilling_fluid' in df_pct.columns:
             df_pct = df_pct.drop(columns=['drilling_fluid'])
        df = pd.merge(df, df_pct, on='Time', how='left') 
        feature_names += '_pctDSA'
        basic_columns.extend(['pct_obm1', 'pct_Case_2a1_Fluid', 'DS_event_pct', 'new_fluid_reached_annulus'])

    basic_columns = list(dict.fromkeys(basic_columns))
    final_missing_cols = [col for col in basic_columns if col not in df.columns]
    if final_missing_cols:
         print(f"Warning: Columns specified in basic_columns are missing from the final DataFrame after processing: {final_missing_cols}")
         basic_columns = [col for col in basic_columns if col in df.columns]
    return df, basic_columns, feature_names


def load_results(results_file):
    results = []
    completed_runs = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            if isinstance(results, list):
                completed_runs = [r[0] for r in results if isinstance(r, (list, tuple)) and len(r) > 0]
            else:
                print(f"Warning: Results file {results_file} does not contain a valid list. Resetting results.")
                results = [] 
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {results_file}. Starting with empty results.")
            results = []
        except Exception as e:
            print(f"Error loading results file {results_file}: {e}. Starting with empty results.")
            results = [] 
    return results, completed_runs


def save_results(results, results_file):
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, allow_nan=True)
    except Exception as e:
        print(f"Error saving results to {results_file}: {type(e).__name__}: {e}")


def encode_binary_fluid(df, column="drilling_fluid"):
    if column not in df.columns:
         raise ValueError(f"Column '{column}' not found in DataFrame for encoding.")
    df["fluid_binary"] = df[column].apply(lambda x: 1 if x=="obm1" else 0)
    return df


def pct_fluid_tracking(df, total_well_volume=194.89,
                            time_col="Time", fluid_col="drilling_fluid",
                            vol_col="totVolInj"):
    required_cols = [time_col, fluid_col, vol_col]
    if not all(col in df.columns for col in required_cols):
         raise ValueError(f"Missing required columns for pct_fluid_tracking: Need {required_cols}")
    if df.empty:
         print("Warning: Input DataFrame for pct_fluid_tracking is empty.")
         return pd.DataFrame(columns=[time_col, "pct_obm1", "pct_Case_2a1_Fluid"])

    df = df.sort_values(by=time_col).reset_index(drop=True).copy()

    segments = deque() 
    fluid_types = df[fluid_col].unique()
    current_volumes = {ftype: 0.0 for ftype in fluid_types}
    if "obm1" not in current_volumes: current_volumes["obm1"] = 0.0
    if "Case_2a1_Fluid" not in current_volumes: current_volumes["Case_2a1_Fluid"] = 0.0

    results = []
    prev_vol = df.at[0, vol_col]
    current_fluid = df.at[0, fluid_col]
    current_fluid_start_vol = prev_vol

    for idx, row in df.iterrows():
        current_vol = row[vol_col]
        new_fluid = row[fluid_col]
        if new_fluid not in current_volumes:
             current_volumes[new_fluid] = 0.0

        if new_fluid != current_fluid:
            segment_vol = current_vol - current_fluid_start_vol
            if segment_vol > 1e-9:
                segments.append((current_fluid, segment_vol))
                current_volumes[current_fluid] += segment_vol
            current_fluid = new_fluid
            current_fluid_start_vol = current_vol

        new_vol_step = max(current_vol - prev_vol, 0)

        if new_vol_step > 1e-9: 
            segments.append((current_fluid, new_vol_step))
            current_volumes[current_fluid] += new_vol_step
            total_in_well = sum(current_volumes.values())
            while total_in_well > total_well_volume and segments:
                try:
                    oldest_fluid, oldest_vol = segments.popleft()
                except IndexError: 
                     break
                removable = min(oldest_vol, total_in_well - total_well_volume)

                current_volumes[oldest_fluid] -= removable
                total_in_well -= removable

                if oldest_vol > removable + 1e-9:
                    segments.appendleft((oldest_fluid, oldest_vol - removable))

        total_actual = sum(current_volumes.values())
        if total_actual < 1e-9:  
             pct_obm1 = 0.0
             pct_case = 0.0
        else:
            pct_obm1 = (current_volumes.get("obm1", 0.0) / total_actual) * 100.0
            pct_case = (current_volumes.get("Case_2a1_Fluid", 0.0) / total_actual) * 100.0

        results.append({
            time_col: row[time_col],
            "pct_obm1": round(pct_obm1, 3),
            "pct_Case_2a1_Fluid": round(pct_case, 3)
        })

        prev_vol = current_vol

    return pd.DataFrame(results)

def track_whole_well_and_annulus_reach(df, well_capacity=171.42, ds_capacity=23.47,
                                       time_col="Time", fluid_col="drilling_fluid",
                                       vol_col="totVolInj"):
   
    required_cols = [time_col, fluid_col, vol_col]
    if not all(col in df.columns for col in required_cols):
         raise ValueError(f"Missing required columns for track_whole_well: Need {required_cols}")
    if df.empty:
         print("Warning: Input DataFrame for track_whole_well is empty.")
         return pd.DataFrame(columns=[time_col, fluid_col, "pct_obm1", "pct_Case_2a1_Fluid", "DS_event_pct", "new_fluid_reached_annulus"])

    df = df.sort_values(by=time_col).reset_index(drop=True).copy()

    segments = deque() 
    fluid_types = df[fluid_col].unique()
    current_volumes = {ftype: 0.0 for ftype in fluid_types}
    if "obm1" not in current_volumes: current_volumes["obm1"] = 0.0
    if "Case_2a1_Fluid" not in current_volumes: current_volumes["Case_2a1_Fluid"] = 0.0
    total_in_well = 0.0 

    current_event_vol = 0.0
    current_event_fluid = df.at[0, fluid_col]  

    results = []
    prev_vol = df.at[0, vol_col] 

    for idx, row in df.iterrows():
        current_cum_vol = row[vol_col]
        fluid = row[fluid_col]
        if fluid not in current_volumes: 
             current_volumes[fluid] = 0.0

        if fluid != current_event_fluid:
            current_event_fluid = fluid
            current_event_vol = 0.0 
            
        new_vol_step = max(current_cum_vol - prev_vol, 0)
        current_event_vol += new_vol_step

        if new_vol_step > 1e-9:
            segments.append((fluid, new_vol_step))
            current_volumes[fluid] += new_vol_step
            total_in_well += new_vol_step

            while total_in_well > well_capacity and segments:
                try:
                     oldest_fluid, oldest_vol = segments.popleft()
                except IndexError:
                     break
                removable = min(oldest_vol, total_in_well - well_capacity)
                current_volumes[oldest_fluid] -= removable
                total_in_well -= removable
                if oldest_vol > removable + 1e-9: 
                    segments.appendleft((oldest_fluid, oldest_vol - removable))

        effective_total_vol = min(total_in_well, well_capacity)
        pct_obm1 = (current_volumes.get("obm1", 0.0) / effective_total_vol) * 100.0 if effective_total_vol > 1e-9 else 0.0
        pct_case = (current_volumes.get("Case_2a1_Fluid", 0.0) / effective_total_vol) * 100.0 if effective_total_vol > 1e-9 else 0.0

        DS_event_pct = (current_event_vol / ds_capacity) * 100.0 if ds_capacity > 1e-9 else 0.0
        DS_event_pct = min(DS_event_pct, 100.0)
        reached_annulus = 1 if current_event_vol >= ds_capacity - 1e-9 else 0

        results.append({
            time_col: row[time_col],
            "pct_obm1": round(pct_obm1, 6),
            "pct_Case_2a1_Fluid": round(pct_case, 6),
            "DS_event_pct": round(DS_event_pct, 6),
            "new_fluid_reached_annulus": reached_annulus
        })

        prev_vol = current_cum_vol

    return pd.DataFrame(results)

def load_and_preprocess_data(args):
    if not args.fluid:
        try:
             df = pd.read_csv(args.data_file, sep=r'\s+')
        except FileNotFoundError:
             raise FileNotFoundError(f"Data file not found: {args.data_file}")

       
        basic_columns = ['Rate_in', 'SPP', 'ECD_bot', 'dsRpm'] 
        all_needed_cols = list(set(basic_columns + args.input_features))
        missing_cols = [col for col in all_needed_cols if col not in df.columns]
        if missing_cols:
             raise ValueError(f"Missing columns in {args.data_file} required for specified features: {missing_cols}")
         
        final_cols_to_keep = list(dict.fromkeys(basic_columns + args.input_features))
        df = df[final_cols_to_keep]
        basic_columns = final_cols_to_keep 
        feature_names = '_'.join(args.input_features) if isinstance(args.input_features, list) else str(args.input_features)

    else: 
        df, basic_columns, feature_names = process_fluid_data(args)
        actual_cols = df.columns.tolist()
        basic_columns = [col for col in basic_columns if col in actual_cols]
        missing_input_features = [feat for feat in args.input_features if feat not in actual_cols]
        if missing_input_features:
             raise ValueError(f"Specified input_features missing after fluid processing: {missing_input_features}")
    if 'ECD_bot' not in df.columns:
         raise ValueError("'ECD_bot' target column is missing from the DataFrame before splitting/scaling.")


    return df, basic_columns, feature_names

def split_and_scale_data(df, basic_columns):
    if not all(col in df.columns for col in basic_columns):
         missing = [col for col in basic_columns if col not in df.columns]
         raise ValueError(f"DataFrame passed to split_and_scale is missing columns: {missing}")
    if len(df.columns) != len(basic_columns):
         extra = [col for col in df.columns if col not in basic_columns]
         print(f"Warning: DataFrame passed to split_and_scale has extra columns: {extra}. Keeping only basic_columns.")
         df = df[basic_columns]


    train_df, temp_df = train_test_split(
        df,
        train_size=0.7,
        shuffle=False
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        shuffle=False
    )
     
    if train_df.empty or val_df.empty or test_df.empty:
         raise ValueError(f"Train/Val/Test split resulted in empty DataFrames. Train:{len(train_df)}, Val:{len(val_df)}, Test:{len(test_df)}")

    scaler = MinMaxScaler()
    scaler.fit(train_df)

    train_scaled = pd.DataFrame(scaler.transform(train_df), columns=basic_columns, index=train_df.index)
    val_scaled   = pd.DataFrame(scaler.transform(val_df),   columns=basic_columns, index=val_df.index)
    test_scaled  = pd.DataFrame(scaler.transform(test_df),  columns=basic_columns, index=test_df.index)

    return train_scaled, val_scaled, test_scaled, scaler

def build_run_name(model_type, args, feature_names, middle_folder):
    if args.augmented_encoder:
        noise=args.ecd_noise_std
        noise=f"AugNoise_{noise}"
    else:
        noise = ""
    return (f"{model_type.upper()}_Feats{feature_names}_HD{args.hidden_dim}_NL{args.num_layers}"
            f"_LR{args.lr}_Data{middle_folder}_WS{args.window_size}_HZ{args.horizon}_BS{args.batch_size}_{noise}")