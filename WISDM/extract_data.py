#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# this file uses the dataset
# Weiss,Gary. (2019). WISDM Smartphone and Smartwatch Activity and Biometrics Dataset .
# UCI Machine Learning Repository. https://doi.org/10.24432/C5HK59.
# 
# =============================================================================

""" This file extracts meta data (id, user_id, activity label, timestamp at start, 
timestamp at end, sequence length) about the sequences and stores them in one 
parquet file and create one parquet file per sequence into a folder 
'wisdm_seq<file_suffix>/', both for every devices x sensors combination. 
The name addition depends on the conducted preprocessings of the sequences. 

For preprocessing directly when loading the data:
Preprocessings treat the frequency inconsistency (different sequence lengths).
Level 1 alignment <>: No Preprocessing (or Decimation if possible); divide sequences with multiples of expected frequency
Level 2 alignment <_i>: Linear interpolation only with downsampling
Level 3 alignment <_ic>: (complete) Linear interpolation with down- and upsampling
"""

# Import libraries
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Import modules
script_dir = os.getcwd()+'/'
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
import preprocessings as prep
pd.set_option('display.max_rows', 901)

all_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
              'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']

def write_to_parquet(table_writings, sensor, device, filenum, user_id, activity, 
                     seq, schema, starttime, endtime, interpolation, 
                     folder_dir, upsample=False):
    """First write a file for this sequence. Then write/save a line to write 
    for this specific sensor device file. Returns the extended lines for the 
    sensor device file.
    Downsampling can be conducted with the methods decimation or interpolation.
    """
    
    ######### Write this sequence
    filenum_formatted = str(filenum).zfill(3) 
    output_name = f'{filenum_formatted}_{sensor[:3]}_{device[0]}_{user_id}_{activity}.parquet'
    
    path = folder_dir + output_name
    
    table = pa.Table.from_pandas(pd.DataFrame(columns=['x', 'y', 'z'], dtype=float), schema=schema)
    pq.write_table(table, path)
    
    ### Downsampling 
    if not interpolation:
        if len(seq) >= (2 * 3600):
            seq = prep.decimate_seq(seq, desired_len=3600)
    else:
        # If frequency is more than desired length 3600 (20Hz), compress this sequence
        if len(seq) != 3600:
            seq = prep.interpolate_seq(seq, desired_len=3600, allow_upsampling=upsample)
            
            
    structured_array = np.core.records.fromarrays(seq.T, names='x, y, z', formats='float64, float64, float64')
   
    # For every timestamp write one line
    with pq.ParquetWriter(path, schema) as writer:
        table = pa.Table.from_pandas(pd.DataFrame(structured_array))
        writer.write_table(table)
    
    # Write one line to table for specific sensor and device
    line = {'id': [np.int16(filenum)], 'user_id': [user_id], 'activity': [activity], \
            'seq_len': [np.int16(len(seq))], 'starttime': [starttime], \
            'endtime': [endtime]}
    table_add = pa.Table.from_pandas(pd.DataFrame(line))

    # Concatenate the new data to the existing table
    table_writings = pa.concat_tables([table_writings, table_add])
    return table_writings
    

def create_merge_table(file_suffix, directory):
    import shutil

    datapath = directory + "wisdm_tables/"

    sensors = ['acc_p', 'acc_w', 'gyr_p', 'gyr_w']
    
    df_lst = []
    for s in sensors:
        parquet_file = pq.ParquetFile(datapath + "wisdm_" + s + file_suffix + ".parquet")
        table = parquet_file.read()
        df_lst.append(table.to_pandas())
    
    # Delete duplicates, are not expected to exist as they were eliminated during data reading
    for i in range(len(df_lst)):
        df_lst[i] = df_lst[i].drop_duplicates(subset=['user_id', 'starttime'], keep='first')

    ###########################################################################
    # Merge / perform inner join
    df_merged_w = pd.merge(df_lst[1], df_lst[3], suffixes= ['_acc_w', '_gyr_w'], on=['id', 'user_id', 'activity'])
    df_merged_p = pd.merge(df_lst[0], df_lst[2], suffixes= ['_acc_p', '_gyr_p'], on=['id', 'user_id', 'activity'])
    df_merged = pd.merge(df_merged_w, df_merged_p, on=['id', 'user_id', 'activity'])
 

    columns_arranged = ['id', 'user_id', 'activity'] + \
            ['starttime_' + s for s in sensors] +\
            ['endtime_' + s for s in sensors] + ['seq_len_' + s for s in sensors]
    df_merged = df_merged[columns_arranged]

    df_merged.to_parquet(datapath + 'wisdm_merged' + file_suffix + '.parquet')


def main():
    
    # XXX Defining a downsampling method will change/downsample some sequences which have 
    # been presumably recorded at a higher/lower frequency:
    # Linear Interpolation using Downsampling
    interpolation = False

    # XXX Linear Interpolation Down- and Upsampling
    upsample = True  # will only be taken into account when interpolation=True

    file_suffix = ""
    if interpolation == True:
        file_suffix = "_i"
        if upsample: file_suffix += "c"

   
    # Folder for meta data tables (inside one file per sensor x device combination)
    folder_table = script_dir + "wisdm_tables/"
    if not os.path.exists(folder_table): os.makedirs(folder_table)

    devices = ['phone', 'watch']
    sensors = ['accel', 'gyro']

    schema = pa.schema([
        ('x', pa.float64()),
        ('y', pa.float64()),
        ('z', pa.float64())])

    schema_meta = pa.schema([
        ('id', pa.int16()),
        ('user_id', pa.int16()),
        ('activity', pa.string()),
        ('seq_len', pa.int16()),
        ('starttime', pa.float64()),
        ('endtime', pa.float64())])
    
    ###########################################################################
    c = 0
    #id_c = 0
    num_files = 0
    
    # Get number of all files
    for sensor in sensors:
        for device in devices:
            num_files += len(sorted(Path(script_dir + "wisdm-dataset/raw/"+device+"/"+sensor).rglob('*.txt')))
    
    for sensor in sensors:
        for device in devices:
            filenums_collected = []
            num_user = 0
            user_id_old = 0
            id_c = 0

            # Folder for sequences
            folder_dir = script_dir + "wisdm_seq"+file_suffix+"/"
            folder_dir += sensor[:3]+"_"+device[0]+"/"
            if not os.path.exists(folder_dir): os.makedirs(folder_dir)

            output_name_meta = "wisdm_"+sensor[:3]+"_"+device[0]+file_suffix
            output_name_meta += ".parquet"
            path_meta = folder_table + output_name_meta
            
            
            # Create initial parquet file
            table_add = pa.Table.from_pandas(pd.DataFrame(columns=['id', 'user_id', 'activity', 'seq_len', 'starttime', 'endtime'], dtype=float), schema=schema_meta)
            pq.write_table(table_add, path_meta)
            table_writings = pq.read_table(path_meta)
           
            sorted_raw_files = sorted(Path(script_dir + "wisdm-dataset/raw/"+device+"/"+sensor).rglob('*.txt'))
            for file in sorted_raw_files:
                df = pd.read_csv(file, lineterminator=';', header=None, names=['user', 'activity', 'timestamp', 'x', 'y', 'z'],
                                       low_memory=False)
                df = df.iloc[:-1]
                df['activity'] = df['activity'].astype(str)
                df['user'] = df['user'].astype('int16')
                
                c += 1
                print(f'Data from {file}, file number {c}/{num_files}, is extracted')
               
                user_id = df['user'][0]
                
                if id_c > 0:
                    if user_id != user_id_old:
                        num_user += 1
                        
                import datetime
                datetime_obj_isave = datetime.datetime.fromtimestamp(0)
                seq_index_start = 0
                act_old = df['activity'][0]
                
                for index, row in df.iterrows():
                    act_i = row['activity']
                    timestamp = row['timestamp']
                    
                    # Dividing by 1,000,000,000 roughly makes 3 minute intervals, so I assume this calculation is valid
                    timestamp_seconds = timestamp / 1000000000
                    datetime_obj_i = datetime.datetime.fromtimestamp(timestamp_seconds)
                    # Save first timestamp of dataset
                    if index == 0: datetime_obj_isave = datetime_obj_i
                    # Calc time difference
                    time_diff = (datetime_obj_i - datetime_obj_isave)
                    time_diff_min = round((time_diff.days * 24 * 60 + time_diff.seconds) / 60, 2)
                    
                    # Save if new activity arrives or end of data set is reached
                    if (((time_diff_min > 3.5 or act_old != act_i) and not (time_diff_min < 6 and act_old == act_i)) and index >= 1) or index == (len(df) - 1):
                        
                        seq_index_end = index #first index of next activity
                        # Extract whole sequence of last activity
                        sequence = (df[['x', 'y', 'z']].values)[seq_index_start:seq_index_end]
                        # Extract activity class/name
                        activity = df['activity'][seq_index_start]
                        
                        filenum = num_user * len(all_labels) + all_labels.index(activity)
                        if filenum not in filenums_collected:
                            table_writings = write_to_parquet(table_writings, sensor, \
                                             device, filenum, user_id, activity, \
                                             sequence, schema, \
                                             df['timestamp'][seq_index_start], \
                                             df['timestamp'][seq_index_end-1], \
                                             interpolation=interpolation,
                                             folder_dir=folder_dir,
                                             upsample=upsample)
                            filenums_collected.append(filenum)
                                     
                        # Start count for next activity
                        seq_index_start = seq_index_end
                        # Save first timestamp of new activity
                        datetime_obj_isave = datetime_obj_i
                        # Count up
                        id_c += 1
                        
                    # Save last activity for comparison
                    act_old = act_i
                    # Save last user for comparision
                    user_id_old = user_id
        
            # Write parquet non sequential data
            pq.write_table(table_writings, path_meta)
    
    ###########################################################################
    
    create_merge_table(file_suffix, script_dir)

if __name__ == "__main__":
    main()