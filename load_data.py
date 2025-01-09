#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality to load preprocessed data including labels and higher features
of a selected dataset, also provides function for loading the dataframes of each dataset.

# =============================================================================
# this file uses the dataset
# Weiss,Gary. (2019). WISDM Smartphone and Smartwatch Activity and Biometrics Dataset .
# UCI Machine Learning Repository. https://doi.org/10.24432/C5HK59.
#
# =============================================================================
"""

import os
script_dir = os.getcwd()+'/'
import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
# Import modules
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
import preprocessings as prep


def load_df(ds, directory, file_suffix=""):
    """Load data from parquet file containing meta data into dataframe"""
    if ds == '1':
        parquet = pq.ParquetFile(directory + "study24.parquet")
        
        table = parquet.read()
        df = table.to_pandas()
        
    elif ds == '2':
        parquet_1 = pq.ParquetFile(directory + "sessions_selfreport.parquet")
        parquet_2 = pq.ParquetFile(directory + "sessions_sensory" + file_suffix + ".parquet")
        
        table_1 = parquet_1.read()
        df_1 = table_1.to_pandas()
        
        table_2 = parquet_2.read()
        df_2 = table_2.to_pandas()
        
        df = (df_1, df_2)
        
    elif ds == '3':
        parquet = pq.ParquetFile(directory + "wisdm_tables/wisdm_merged" + file_suffix + ".parquet")
            
        table = parquet.read()
        df = table.to_pandas()
    
    return df

def get_inputs(ds, type, sensor_names=["acc_p", "acc_w", "gyr_p", "gyr_w"],
                 chosen_activities='all', 
                 dictionary={},
                 global_padding=None, 
                 sliding_window=True,
                 window_seconds=10, 
                 overlap_perc=0., 
                 interpolate=False, 
                 upsample=True,
                 upper_cut=False,
                 desired_lens=None,
                 delete_till_sec=None):
    
    """Load data, sequences (type '1') or load spectrograms 
    from sequences (type '2'), and apply preprocessing to them, 
    as well as labels, a list of users and activity IDs,
    and higher features array and return them.´
    
    Arguments:
    ds -- '1' for HA24, '2': for CXT22, '3' for WISDM
    type -- '1' for sequences, '2' for spectrograms
    sensor_names -- list of sensor types as strings (default: ["acc_p", "acc_w", "gyr_p", "gyr_w"])
        for ds = '2' additionally 'mag_p' and 'mag_w' are available
    chosen_activities -- list of label names that should be included (default: all classes in dataset)
    dictionary -- dictionary containing different names for selected label names 
        of the chosen dataset, converts the selected labels in the returned labels array
    
    global_padding -- 'full' padding over all sensor types to the same (max) length, 
        'semi' padding over watch/phone sequences max lengths,
        None padding over each sensors sequences max lengths (default None)
    
    sliding_window -- divide sequences into subsequences (windows) 
        whole sequences (default: True)
    window_seconds -- if sliding_window: the number of seconds that represent one window (default: 10)
    overlap_perc -- proportion of overlap between windows (default: 0.0 (no overlap))
    
    interpolate -- sequence length is interpolated by downsampling (default: False)
    upsample -- if interpolate is True, sequence length is interpolated by down- 
        and upsampling (default: True)
    upper_cut -- cut off sequences at desired lengths (default False)
    desired_lens -- given a list with len(sensor_names) lengths are assigned directly,
        given a single integer value length is assigned to all sensors,
        'avg': desired_lens are the mean lengths of each sensor,
        'avgstd': desired_lens are the mean + std lengths of each sensor,
        None: if interpolate or upper_cut, lengths will be standard lengths
    
    directory -- directory containing the dataset (default: script_dir)
    """
    
    # Full recording times in seconds
    recording_time = {'1': 150, '2': 150, '3': 180}
    # Data directories
    data_dir = {'1': '/HA24/', '2':'/CXT22/', '3': '/WISDM/'}
    data_path = os.getcwd() + data_dir[ds]
    seq_dir = {'1': 'study24_seq/', '2': 'sessions_seq/', '3': 'wisdm_seq/'}
    # Name of the label columns in each dataset
    label_col_name = {'1': 'hand_activity', '2': 'hand_activity', '3': 'activity'}
    # Seconds deleted from each sequence, before window splitting
    delete_till_dic = {'1': 0, '2': 0, '3': 10}
    # Standard sequence length, used when no desired_lens received but interpolate True
    len_ds = {'1': 15000, '2': 15000,'3': 3600}
    
    ###########################################################################
    # Load data, sort sensor names list
    df = load_df(ds, directory=data_path)
    sensor_names = sorted(sensor_names)
    num_sensors = len(sensor_names)
    # Apply default value for specific dataset when no value received for delete_till_sec
    if delete_till_sec == None: 
        delete_till_sec = delete_till_dic[ds]
    
    ###########################################################################
    # CXT22 dataset specifics
    if ds == '2': 
        df_1, df_2 = df
        
        # Delete rows without hand activity /label
        df_1 = df_1[df_1[label_col_name[ds]].notna()]
        
        # Collect ids available for each seq sensor and delete null vals
        avail_ids = {key: [] for key in sensor_names}
        
        for key in avail_ids:
            avail_ids[key] = df_2[(df_2['seq_len_'+key]) > 0]['id']
            
        # Delete null vals for seq sensors
        for i in range(num_sensors):
            df_2 = df_2[df_2['seq_len_'+sensor_names[i]].notna()]
        
        # Delete unavailable rows also from selfreport table
        avail_ids_sensors = df_2['id']
        df_1 = df_1[df_1['id'].isin(avail_ids_sensors)]
        
        # Inner join on the two tables
        df = pd.merge(df_1, df_2, on=['id', 'user_id', 'session_id', 'interval_index'])
        
        # Remove all rows labeled unsure
        df = df[df[label_col_name[ds]] != "unsicher"]

    ###########################################################################
    # WISDM dataset specifics
    if ds == '3': 
        avail_ids_sensors = df['id']
        
    ###########################################################################
    if chosen_activities != 'all':
        # Delete rows which are not in chosen activities
        # -> only keep rows with chosen activities 
        keep_rows = df[label_col_name[ds]].isin(chosen_activities)
        matched_ids = df.loc[keep_rows, 'id'].tolist()
        
        df = df[df['id'].isin(matched_ids)]
    # All remaining unique activities
    chosen_activities = np.unique(df[label_col_name[ds]])
    
    # Save labels in np array
    this_labels = np.array(df[label_col_name[ds]]) 
    # Translate label name into desired label name as in dictionary
    this_labels = np.array(
        [dictionary[label] if label in dictionary else label for label in this_labels])
    # Get user column
    users = np.array(df['user_id'])
    # Get activity ID column
    activity_ids = np.array(df['id'])
    
    ###########################################################################
    # Maximum found lengths of all seq sensors
    max_lens = [int(max(df['seq_len_'+sensor_names[i]])) for i in range(num_sensors)]

    
    if isinstance(desired_lens, list) and len(desired_lens) == num_sensors: 
        # List is complete / matched to sensors
        pass
    
    elif isinstance(desired_lens, int):
        # Expand the desired length to apply to all sensors
        desired_lens = [desired_lens] * num_sensors
        
    elif desired_lens == "avg":
        # Mean of all lengths of sensor i
        desired_lens = [round(np.mean(df['seq_len_'+sensor_names[i]])) for i in range(num_sensors)]
    
    elif desired_lens == "avgstd":
        desired_lens =  [round(np.mean(df['seq_len_'+sensor_names[i]]) \
                          + np.std(df['seq_len_'+sensor_names[i]])) for i in range(num_sensors)]
    
    elif desired_lens == None:
        if (interpolate or upper_cut): 
            # If interpolation is True, will interpolate to standard length
            desired_lens = [len_ds[ds]] * num_sensors
    else:
       AssertionError()
          
       
    if upper_cut:
        # For cutting all arrays at desired length
        max_lens = desired_lens

    # Raise Error if desired length is less than maximum length when there is no downsampling
    if desired_lens != None and not upper_cut and not interpolate:
        if any(max_len > desired_len for max_len, desired_len in zip(max_lens, desired_lens)):
            raise AssertionError(" The chosen 'desired_lens' is not compatible for applying padding. Choose 'interpolate' or 'upper_cut' to be true to force desired lengths.")

    # If global_padding is set True, the max_lens will be set to the
    # maximum found length of all sensors ('full'), 
    # or watch/phone sensors ('semi'), 
    # so every sequence (of the same device) will be having the same 
    # length after applying padding

    if global_padding == 'full':
        max_lens = [max(max_lens)] * num_sensors
        
    elif global_padding == 'semi':
        max_len = {'p': 0, 'w': 0}
        for i in range(num_sensors):
            device = sensor_names[i][-1]
            max_len[device] = max(max_len[device], max_lens[i])
        max_lens = [max_len['w'] if sensor_names[i][-1] == 'w' else max_len['p']
                    for i in range(len(max_lens))]

    ###########################################################################
    # Initialize empty lists to store sequences from each file
    sequences_x = [[] for i in range(num_sensors)]
    sequences_y = [[] for i in range(num_sensors)]
    sequences_z = [[] for i in range(num_sensors)]

    for i in range(num_sensors):
        directory_folder = data_path + seq_dir[ds] + sensor_names[i] + "/"
        grouped_parquet_files = sorted([file for file in os.listdir(directory_folder) \
                                        if file.endswith('.parquet')])
        # Read data from Parquet files
        for file_name in grouped_parquet_files:
            add_sample = True
            if ds == '1':
                file_activity = (file_name.replace('.parquet', '')).split('_')[4:]
                file_activity = ('_').join(file_activity)
            elif ds == '2':
                file_activity = file_name.split('_')[5:]
                file_activity = "_".join(file_activity).replace(".parquet", '')
                
                file_id = int(file_name.split('_')[0])
                # Add only if data is available for all sensors 
                if not (file_id in list(avail_ids_sensors)): 
                    add_sample = False
                    
            elif ds == '3':
                file_activity = str(file_name).split('_')[-1][0]
                file_id = int(file_name.split('_')[0])

                # Add only if data is available for all sensors 
                if not (file_id in list(avail_ids_sensors)): 
                    add_sample = False
                    
            # If ID is available for all sensors and check if it is a chosen activity
            if (file_activity in chosen_activities) and add_sample:

                # Read the Parquet file
                df_temp = pd.read_parquet(os.path.join(directory_folder, file_name))
                
                # Extract sequences from the DataFrame columns   
                xs = df_temp['x'].values
                ys = df_temp['y'].values
                zs = df_temp['z'].values

                if upper_cut:
                    # Cut off part longer than defined length for that sensor
                    xs = xs[:max_lens[i]]
                    ys = ys[:max_lens[i]]
                    zs = zs[:max_lens[i]]

                sequences_x[i].append(xs)
                sequences_y[i].append(ys)
                sequences_z[i].append(zs)

    for i in range(num_sensors): 
        
        # Change length with interpolation to desired length
        if interpolate:
            # Interpolate each dimension 
            sequences_x[i] = prep.interpolate_arr_lst(sequences_x[i], desired_lens[i]-1, upsample=upsample)
            sequences_y[i] = prep.interpolate_arr_lst(sequences_y[i], desired_lens[i]-1, upsample=upsample)
            sequences_z[i] = prep.interpolate_arr_lst(sequences_z[i], desired_lens[i]-1, upsample=upsample)
            # Update max value length for padding            
            #max_lens[i] = max([len(seq) for seq in sequences_x[i] + sequences_y[i] + sequences_z[i]])
            max_lens[i] = max(max([len(seq) for seq in sequences_x[i] + sequences_y[i] + sequences_z[i]]), desired_lens[i])

        # Pad data with last value
        sequences_x[i] = [np.concatenate((seq, np.full(int(max_lens[i] - len(seq)), seq[-1], dtype=seq.dtype)), axis=0) for seq in sequences_x[i]]
        sequences_y[i] = [np.concatenate((seq, np.full(int(max_lens[i] - len(seq)), seq[-1], dtype=seq.dtype)), axis=0) for seq in sequences_y[i]]
        sequences_z[i] = [np.concatenate((seq, np.full(int(max_lens[i] - len(seq)), seq[-1], dtype=seq.dtype)), axis=0) for seq in sequences_z[i]]

    # Save all np arrays  in one list
    data = []
    frequencies = [0] * num_sensors

    # Convert lists of sequences to numpy arrays, each shaped (3, num_recordings, seq_length)
    for i in range(num_sensors):
        data.append(np.array([sequences_x[i], sequences_y[i], sequences_z[i]]))
        # Divide length by record time in seconds to get frequency
        # In the case of a lot of padding, can lead to distorted / higher values
        frequencies[i] = data[i].shape[2] / recording_time[ds]
        
    
    # In case of sliding_window:
    # split sequences into window chunks with length window_size and a stepsize
    if sliding_window:
        data, this_labels, additional_list = prep.apply_sliding_window_by_seconds(
            data, this_labels, frequencies, additional_list=[users, activity_ids], 
            window_seconds=window_seconds, percentage_overlap=overlap_perc, 
            delete_till_sec=delete_till_sec)
    else:
        additional_list = [users, activity_ids]
        
    # Reshape -> (num_recordings, seq_length, 3)
    for i in range(len(data)):
        data[i] = data[i].transpose(1, 2, 0)

    ####################################################################
    # Get some higher features for each sensor and dimension
    higher_features = prep.get_higher_features(data)
    
    # Create spectrograms
    if type == '2':
        data = prep.get_spect_data(data, this_labels, additional_list, sensor_names, 
                              frequencies, recording_time[ds])
        
    return data, this_labels, additional_list, higher_features
    


