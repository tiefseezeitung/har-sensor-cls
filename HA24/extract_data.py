#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract data from raw JSON files and save data into Parquet files.
Preprocessings can be conducted here already (will add suffix to file) but 
is not recommended as it can also be applied from 'load_data' module when loading 
data into the memory.

For preprocessing directly when loading the data:
Preprocessings treat the frequency inconsistency (different sequence lengths).
Level 1 alignment <>: No Preprocessing (or Decimation if possible); divide sequences with multiples of expected frequency
Level 2 alignment <_i>: Linear interpolation only with downsampling
Level 3 alignment <_ic>: (complete) Linear interpolation with down- and upsampling
"""

# Import libraries
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Import modules
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
import preprocessings as prep
import auxiliary as aux

def write_seq_to_parquet(seq, sens_type, id_c, user, activity, desired_seq_len, 
                         schema_seq, directory, interpolation=True, upsample=False, file_suffix=""):
    """ Writes a sequence into one parquet file. """
    
    # Do not save when seq is recorded but no activity available
    if activity == None: return 0
    # Fill space with 0 if only one digit id
    id_c = f"{int(id_c):03}"

    output_name = f'{id_c}_{user}_{sens_type}_{activity}.parquet'
    folder_path = os.path.join(directory, f"study24_seq{file_suffix}", sens_type)
    writing_path = os.path.join(folder_path, output_name)
    os.makedirs(folder_path, exist_ok=True)
    
    table = pa.Table.from_pandas(pd.DataFrame(columns=['x', 'y', 'z'], dtype=float), schema=schema_seq)
    pq.write_table(table, writing_path)
    
    ### Interpolation 
    if not interpolation:
        if len(seq) >= (2 * desired_seq_len):
            seq = prep.decimate_seq(seq, desired_len=desired_seq_len)
    else:
        # If frequency is more than defined/expected tolerance 15050/15620, downsample
        # assumption 100-104 Hz for 2.5 mins        
        if len(seq) != desired_seq_len:
                seq = prep.interpolate_seq(seq, desired_len=desired_seq_len, allow_upsampling=upsample)
            
    # Transpose so all values can be written with in one step
    structured_array = np.core.records.fromarrays(seq.T, names='x, y, z', formats='float64, float64, float64')
    
    with pq.ParquetWriter(writing_path, schema_seq) as writer:
        table = pa.Table.from_pandas(pd.DataFrame(structured_array))
        writer.write_table(table)
        
    return len(structured_array)


def add_to_parquet(table, my_dict, schema):
    """ Save/add the information as row of data by combining with existing table. """
        
    row = my_dict
    table_add = pa.Table.from_pandas(pd.DataFrame(row), schema=schema)

    # Concatenate the new data to the existing table
    table = pa.concat_tables([table, table_add])
    return table

def write_to_parquet(table, path): 
    """ Writes table to parquet file at path"""
    pq.write_table(table, path)

def main():
    script_dir = os.getcwd()+'/'
    
    # XXX Defining a downsampling method will change/downsample some sequences which have 
    # been presumably recorded at a higher/lower frequency:
    # Linear Interpolation using Downsampling
    interpolation = False 

    # XXX Linear Interpolation Down- and Upsampling
    upsample = True  # will only be taken into account when interpolation=True

    # XXX Categories are saved in English using a dictionary from auxiliary module
    english_categories = True
        
    ###########################################################################
    schema_xyz = pa.schema([
        ('x', pa.float64()),
        ('y', pa.float64()),
        ('z', pa.float64())
        ])
    
    schema_main = pa.schema([
        ('id', pa.int16()),
        ('user_id', pa.int16()),
        ('act_id_user', pa.int16()),
        ('hand_activity', pa.string()),
        ('timestamp_from', pa.string()), #pa.timestamp('s') or pa.float()
        ('timestamp_to', pa.string()),
        ('sec_elapsed', pa.int16()),
        ('seq_len_acc_p', pa.int32()),
        ('seq_len_acc_w', pa.int32()),
        ('seq_len_gyr_p', pa.int32()),
        ('seq_len_gyr_w', pa.int32())
        ])       
    ###########################################################################
    # Create files
    
    # Add suffix to mark aligned data path
    file_suffix = ""
    if interpolation:
        file_suffix = "_i"
        if upsample: file_suffix += "c"
        
    # Create sensory table
    output_name_all = "study24"+file_suffix+".parquet"
    path_all = script_dir + output_name_all
    table_add = pa.Table.from_pandas(pd.DataFrame(columns=schema_main.names, dtype=float), schema=schema_main)
    pq.write_table(table_add, path_all)
    table_writings = pq.read_table(path_all)
    
    # Directory of json files
    data_directory = script_dir +"data/"
    
    json_files = sorted(Path(data_directory).rglob('*.json'))
    num_files = len(json_files)
    
    for file, i  in zip(json_files, range(num_files)):
        filepath = file
        # Load one session from json file to python dict
        json_file = open(filepath, "r", encoding="utf-8")
        open_dict = json.load(json_file)
        json_file.close()
        filename = str(filepath).split("/")[-1]
        
        print(f'Data from {filename}, \nfile number {i+1}/{num_files}, is extracted\n')
        
        # Initialize dict for writing a line
        id_count = i
        user_id = open_dict['user'] #user zuordnen
        act_id_user = open_dict['id']
        category = open_dict['category']
        
        
        line_dict = [{field.name: None for field in schema_main}\
                        for i in range(num_files)]
            
        # Add to dictionary
        line_dict[i]['id'] = np.int16(id_count)
        line_dict[i]['user_id'] = np.int16(user_id)
        line_dict[i]['act_id_user'] = np.int16(act_id_user)
        # Get english name and/or replace whitespace with underscores
        if english_categories:
            category = aux.english_activity_names[category].replace(' ', '_')
        else:
            category = category.replace(' ', '_')
        line_dict[i]['hand_activity'] = category
    
        line_dict[i]['timestamp_from'] = open_dict['start_time'] #this is a string
        line_dict[i]['timestamp_to'] = open_dict['end_time'] #this is a string
        line_dict[i]['sec_elapsed'] = np.int16(open_dict['sec_elapsed'])
    
         
        #######################################################################
        # Extract sensors
        
        for sensor in ['acc_w', 'acc_p', 'gyr_w', 'gyr_p']:
            # Add original sequence lengths to main table
            line_dict[i]['seq_len_'+sensor] = pa.array([np.int32(open_dict['len_'+sensor])], type=schema_main.field('seq_len_'+sensor).type)
            
            sensor_array = np.array(open_dict[sensor])
            if sensor[-1] == 'w': desired_len = aux.seq_length['w']
            else: desired_len = aux.seq_length['p']
            
            # Write this sensor sequence into one file
            _ = write_seq_to_parquet(sensor_array, sensor, i,  user_id, 
                            category, desired_len, schema_xyz, directory=script_dir,
                            interpolation=interpolation, upsample=upsample, 
                            file_suffix=file_suffix)
        
        # Add line to meta data table
        table_writings = add_to_parquet(table_writings, line_dict[i], schema_main)
          
    
    # Write parquet non sequential data to file
    write_to_parquet(table_writings, path_all)

if __name__ == "__main__":
    main()
