#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This module contains functions for extracting data from the raw JSON
files and saves data to parquet files in a structured way.
Selfreport data of all samples is saved into one file, sensory data (excluding 
the time series data of Acc./Gyr./Mag.) is saved into another file.
The sequence data is saved separately for every sample and sensor type.

#####################
For preprocessing directly when loading the data:
Preprocessings treat the frequency inconsistency (different sequence lengths).
Level 1 alignment <>: No Preprocessing (or Decimation if possible); divide sequences with multiples of expected frequency
Level 2 alignment <_i>: Linear interpolation only with downsampling
Level 3 alignment <_ic>: (complete) Linear interpolation with down- and upsampling
#####################

The program asks the user for an input, when a custom answer is detected:
The user will be prompted to either assign the custom answer to an available category 
(by typing corresponding number of category) or type in another name for it or
adopt the given answer (by pressing Enter).

For recreating the uploaded files, use these answers:
'Surfen am Computer' was adopted as it is, as a activity,
'Fußball kommt' was assigned to a new category 'Termin'.
"""

# Import libraries
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import pyarrow as pa
import pyarrow.parquet as pq

# Import modules
script_dir = os.getcwd()+'/'
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
import preprocessings as prep
import mappings
import schemas as sc

def extract(s_t, d_s, sensor, schema_names):
    """ Extracts a sensor data entry and returns its value and sensor name.
    s_t -- Service_type.
    d_s -- Datasource number.
    sensor -- Dictionary containing the data.
    schema_names -- List of all sensor names, except ['acc', 'gyr', 'mag'].
    """
    
    all_out_names = schema_names + ['acc', 'gyr', 'mag']

    message_sensor_notfound = f'\nQuery for found data of service type {s_t}, device {d_s} with sensor type {sensor["sensor_type"]} missing!\n'
    
    if s_t == 0: 
        if sensor['sensor_type'] == -4: #gps, skip (no information)
            out = ''
            out_name = ''
        elif sensor['sensor_type'] == -2: #wifi/mac adresses
            #wifi_len / num detected wifi networks
            out = len(sensor['sensor_events']) 
            out_name = all_out_names[0]
        else: print(message_sensor_notfound)
        
    elif s_t == 1: 
        if sensor['sensor_type'] == 5: #lux
            #lux_median
            out = np.average([sensor['sensor_events'][i]['value']['values'][0] for i in range(len(sensor['sensor_events']))]) 
            out_name = all_out_names[1]
        elif sensor['sensor_type'] == -5: #noise, max amp
            #noise_median
            out = np.average([sensor['sensor_events'][i]['value']['values'][0] for i in range(len(sensor['sensor_events']))])
            out_name = all_out_names[2]
        elif sensor['sensor_type'] == 13: #degree
            #temp_median
            out = np.median([sensor['sensor_events'][i]['value']['values'][0] for i in range(len(sensor['sensor_events']))])
            out_name = all_out_names[3]
        else: print(message_sensor_notfound)
        
    elif s_t == 2:
        if sensor['sensor_type'] == 21: 
            #heartbeat
            out = np.average([sensor['sensor_events'][i]['value']['values'][0] for i in range(len(sensor['sensor_events']))])
            out_name = all_out_names[4]
        elif sensor['sensor_type'] == 69640: #percentage #should only be 1 number
            #percentage_avg
            out = np.average([sensor['sensor_events'][i]['value']['values'][0] for i in range(len(sensor['sensor_events']))])
            out_name = all_out_names[5]
        elif sensor['sensor_type'] == 69641: #ms
            #ms_median, " (standard deviation of NN intervals)
            out = np.median([sensor['sensor_events'][i]['value']['values'][0] for i in range(len(sensor['sensor_events']))])
            out_name = all_out_names[6]
        else: print(message_sensor_notfound)
        
    elif s_t == 4: #step counts
        if sensor['sensor_type'] == 19: #"N/A"
            out = sensor['sensor_events'][0]['value']['values'][0]
            out_name = all_out_names[7]
        elif sensor['sensor_type'] == 69633: #"N/A"
            out = sensor['sensor_events'][0]['value']['values'][0]
            out_name = all_out_names[8]
        elif sensor['sensor_type'] == 26: #unit: Tilt
            out = sensor['sensor_events'][0]['value']['values'][0]
            out_name = all_out_names[9]
        elif sensor['sensor_type'] == 18: #unit: Step
             out = np.average([sensor['sensor_events'][i]['value']['values'][0] for i in range(len(sensor['sensor_events']))])
             out_name = all_out_names[10]
        else: print(message_sensor_notfound)
        
    # Accelerometer, Gyroscope and Magnetometer
    elif s_t == 3: 
        if sensor['sensor_type'] == 1: #acc in m/s^2
            out_name = all_out_names[11]
        elif sensor['sensor_type'] == 4: #gyroscope in rad/s
            out_name = all_out_names[12]
        elif sensor['sensor_type'] == 2: #magnetometer in T
            out_name = all_out_names[13]
        else: print(message_sensor_notfound)
        out = np.array([sensor['sensor_events'][i]['value']['values'] for i in range(len(sensor['sensor_events']))])
        
        
    if out_name != '': out_name = out_name + '_' + mappings.datasource_decoded[d_s][0]
    
    return out, out_name

def write_seq_to_parquet(seq, sens_type, id_c, session_id, device, user, 
                         activity, schema_seq, interpolation=True, 
                         upsample=False, file_suffix=""):
    """ Writes a sequence seq into one parquet file. """
    # Do not save when seq is recorded but no activity available
    if activity == None: return 0
    # Fill space with 0 if ID is one digit 
    #if len(str(id_c)) < 2: id_c = '0'+str(id_c)
    id_c = f"{int(id_c):02}"

    output_name = f'{id_c}_{session_id}_{user}_{sens_type}_{activity[0]}.parquet'
    folder_path = os.path.join(script_dir, f"sessions_seq{file_suffix}", sens_type)
    writing_path = os.path.join(folder_path, output_name)
    
    os.makedirs(folder_path, exist_ok=True)
    
    table = pa.Table.from_pandas(pd.DataFrame(columns=['x', 'y', 'z'], dtype=float), schema=schema_seq)
    pq.write_table(table, writing_path)
    
    ### Downsampling 
    desired_len = 15000
    if not interpolation:
        if len(seq) >= (2 * desired_len):
            seq = prep.decimate_seq(seq, desired_len=desired_len)
    else:
        # If frequency is more than defined/expected tolerance 15000, downsample (and upsample)
        # assumption 100 Hz for 2.5 mins/150 seconds
        
        if len(seq) != desired_len:
                seq = prep.interpolate_seq(seq, desired_len=desired_len, allow_upsampling=upsample)
            
    # Transpose so all values can be written with in one step
    structured_array = np.core.records.fromarrays(seq.T, names='x, y, z', formats='float64, float64, float64')

    with pq.ParquetWriter(writing_path, schema_seq) as writer:
        table = pa.Table.from_pandas(pd.DataFrame(structured_array))
        writer.write_table(table)
        
    return len(seq)

def write_usage_to_parquet(app_list, session_id, device, user, schema_usage, 
                           name='usage', file_suffix=""):
    """ Write list of used apps with columns [app_name, total_time_foreground] of one session to parquet file. """
    
    output_name = f'{session_id}_{user}_{device}_{name}.parquet'
    folder_path = os.path.join(script_dir, f"sessions_seq{file_suffix}", "usage")
    path = os.path.join(folder_path, output_name)
    os.makedirs(folder_path, exist_ok=True)

    table = pa.Table.from_pandas(pd.DataFrame(columns=schema_usage.names, dtype=float), schema=schema_usage)
    pq.write_table(table, path)
    
    cols = len(schema_usage.names)
    with pq.ParquetWriter(path, schema_usage) as writer:
        table = pa.Table.from_arrays([pa.array([row[i] for row in app_list]) for i in range(cols)], names=schema_usage.names)
        
        writer.write_table(table)
    
def add_to_parquet(table, my_dict, schema):
    """ Saves/adds the information as row of data by combining with existing table. """
        
    row = my_dict
    table_add = pa.Table.from_pandas(pd.DataFrame(row), schema=schema)

    # Concatenate the new data to the existing table
    table = pa.concat_tables([table, table_add])
    return table

def write_to_parquet(table, path): 
    """ Writes table to parquet file at path. """
    pq.write_table(table, path)

def def_schema(fields, dtypes):
    """ Creates a new pyarrow schema with list of fields and dtypes."""
    if len(fields) != len(dtypes):
        raise IndexError(f"Lengths of lists must match. Length of fields {len(fields)}, length of dtypes {len(dtypes)}.")
    schema = pa.schema((fields[i], dtypes[i]) for i in range(len(fields)))
    return schema

def main():        
    ###########################################################################
    # Create files
    
    # XXX Defining a downsampling method will change/downsample some sequences which have 
    # been presumably recorded at a higher/lower frequency:
    # Linear Interpolation using Downsampling
    interpolation = False
    
    # XXX Linear Interpolation Down- and Upsampling
    upsample = True  # will only be taken into account when interpolation=True
    
    # Add suffix to mark aligned data path
    file_suffix = ""
    
    if interpolation:
        file_suffix = "_i"
        if upsample: file_suffix += "c"
        
    # Create selfreport table
    output_name_all = "sessions_selfreport.parquet"
    path_all = script_dir + output_name_all
    table_add = pa.Table.from_pandas(pd.DataFrame(columns=sc.schema_selfrep.names, dtype=float), schema=sc.schema_selfrep)
    pq.write_table(table_add, path_all)
    table_writings = pq.read_table(path_all)
    
    
    # Create sensory table
    output_name_all = "sessions_sensory" + file_suffix + ".parquet"
    path_all_sensors = script_dir + output_name_all
    table_add = pa.Table.from_pandas(pd.DataFrame(columns=sc.schema_sensor.names, dtype=float), schema=sc.schema_sensor)
    pq.write_table(table_add, path_all_sensors)
    table_writings_sensors = pq.read_table(path_all_sensors)
    
    # Create notification table
    output_name_all = "sessions_notifs.parquet"
    path_all_notifs = script_dir + output_name_all
    table_add = pa.Table.from_pandas(pd.DataFrame(columns=sc.schema_notifs.names, dtype=float), schema=sc.schema_notifs)
    pq.write_table(table_add, path_all_notifs)
    table_writings_notifs = pq.read_table(path_all_notifs)
    
    # Directory of json files
    sessions_directory = script_dir + "Sessions/"
    
    c = 0
    id_c = 0 # ID count for written lines
    num_files = len(sorted(Path(sessions_directory).rglob('*.json')))
    
    for path in Path(sessions_directory).rglob('*.json'):
        filename = path
        # Load one session from json file to python dict
        json_file = open(filename, "r", encoding="utf-8")
        session= json.load(json_file)
        json_file.close()
        
        c += 1
        print(f'Data from {filename}, file number {c}/{num_files}, is extracted')
        
        user_id = session['participant']['user']['id'] #assign to user
        session_id = session['id']
        
        num_intervals = len(session["record_questionnaires"]) - 1
        num_questionnaires = len(session["record_questionnaires"])
        
        if num_intervals >= 1:
            # Init dict for writing a line
            line_dict = [{field.name: None for field in sc.schema_selfrep}\
                         for i in range(num_questionnaires)]
            # Define temp variables
            in_quest_available = False #In Session Questionnaire availale?
            post_quest_available = False #Post Session Questionnaire availale?
            delete_indices = [] #empty rows/questionnaires to be deleted
            
            for i in range(0, num_questionnaires):
                extract_answers = True #temp variable
                # Add to dictionary
                line_dict[i]['user_id'] = np.int16(user_id)
                line_dict[i]['session_id'] = np.int16(session_id)
                
                # Check whether this is a duplicate (second) pre-session questionnaire
                if session['record_questionnaires'][i]['questionnaire']['id'] == 1 and i !=0 : 
                    num_intervals = num_intervals-1
                    extract_answers = False
                    delete_indices.append(i) #dictionary with that index won't be written to file later on
             
                    
                # Check whether this is a post-session-questionnaire
                if session['record_questionnaires'][i]['questionnaire']['id'] == 3: post_quest_available = True
                else: post_quest_available = False
                
                # Check whether at least one In-session questionnaire was recorded
                if session['record_questionnaires'][i]['questionnaire']['id'] == 2: in_quest_available = True
             
                # When it's no duplicate pre-session questionnaire save questionnaire answers
                if extract_answers:
                    answers_dict = session['record_questionnaires'][i]['record_questions']
                    
                    # Add time information
                    timestamp = session['record_questionnaires'][i]['timestamp_asked']['timestamp']
                    line_dict[i]['timestamp_from'] = timestamp #this is a string
                   
                    for question in answers_dict:
                        # If post_questionnaire skip first question (it's no real question!)
                        # If post_quest_available and question == answers_dict[0]: continue
                        if post_quest_available and answers_dict.index(question) == 0: continue
                        else:
                            # Answer is custom entry
                            if question['record_answer'][0]['custom_answer_given']:
                                string_answer = question['record_answer'][0]['custom_answer']
                            # Non custom entry
                            else: string_answer = question['record_answer'][0]['answer']['answer']
                            
                            # Get name of question and get abbreviated form
                            description = question['question']['question']
                            description = mappings.questions_abbrev[description]
                            
                            # If available inspect custom answer and correct it casewise
                            if description in mappings.string_categories.keys():
                                if string_answer not in mappings.string_categories[description]:
                                    string_answer = string_answer.rstrip()
                                    # Print original question, custom answer and all non custom possible answers/categories
                                    print('\n'+question['question']['question'])
                                    print('custom answer: '+string_answer)
                                    print('available categories:')
                                    for index, cat in enumerate(mappings.string_categories[description]):
                                        print(f"{index}: {cat}")
                                    # Ask user for either index or string or enter and remove whitespace
                                    input_i_str = (input("Type in index of suitable category OR type in appropriate string OR press Enter to continue with custom answer: ")).rstrip()
                                    # If empty keep custom string
                                    if input_i_str.rstrip() == '': pass
                                    # Else if input_i_str can be converted to integer get string of category
                                    else:
                                        try:
                                            input_i_str = int(input_i_str)
                                            string_answer = mappings.string_categories[description][input_i_str]
                                        except ValueError:
                                            string_answer = input_i_str
                                    print('Saved answer: '+string_answer)
                                    
                            # Convert likert scale values accordingly
                            if string_answer in mappings.likert and description != 'cause_non_relevant_learning' and \
                                description != 'interruptions' and description != 'cause_interruption': 
                                write_answer = np.int16(mappings.likert[string_answer])
                                
                                # Adjust name for filed name in dictionary
                                if description in ['fatigue', 'boredom', 'motivation', 'concentration']: 
                                    description = 'pre_' + description
                            # Get integer mapping of num interruptions
                            elif description == 'interruptions':
                                write_answer = mappings.get_int_interruption(string_answer)
                            
                            # Convert Ja/Nein to boolean
                            elif description == 'group_learning': write_answer = mappings.boolean[string_answer] 
                            else: 
                                # Remove whitespace at the end and insert underscore between words
                                write_answer = string_answer.rstrip().replace(" ", "_")
    
                            # Convert to defined schema datatype and add to dictionary
                            line_dict[i][description] = pa.array([write_answer], type=sc.schema_selfrep.field(description).type) 
            
            if len(delete_indices) > 0: 
                for i in delete_indices: line_dict.pop(i)
                
            # Add dictionary values from the pre session quest to all saved dictionaries 
            for i in range(len(line_dict)-1):
                line_dict[i] = {key: (line_dict[0][key] if key in sc.schema_pre.names else line_dict[i][key]) for key in line_dict[i].keys()}
        
            # Add dictionary values from the post session quest to all saved dictionaries 
            if post_quest_available: #!! if not should I add None?
                # Add for all except last post quest dictionary
                for i in range(len(line_dict)-1):
                    line_dict[i] = {key: (line_dict[-1][key] if key in sc.schema_post.names else line_dict[i][key]) for key in line_dict[i].keys()}
            
            for i in range(num_intervals):
                line_dict[i]['interval_index'] = np.int16(i)
                line_dict[i]['id'] = np.int16(id_c)
                id_c += 1
                
                if in_quest_available and not (post_quest_available and num_intervals-1 == i):
                    line_dict[i]['timestamp_to'] = line_dict[i]['timestamp_from']
                    line_dict[i]['minutes_elapsed'] = np.float64("%.2f" %  \
                        (pd.Timedelta(pd.Timestamp(line_dict[i+1]['timestamp_from']) - \
                                      pd.Timestamp(line_dict[i]['timestamp_from'])).seconds / 60))
                    # Get values from next in session quest
                    line_dict[i] = {key: (line_dict[i+1][key] if key in sc.schema_in.names else line_dict[i][key]) for key in line_dict[i].keys()}
                    
                    # Get values from post mood fields
                    for field in ['fatigue', 'boredom', 'motivation', 'concentration']: 
                        line_dict[i]['post_'+field] = line_dict[i+1]['pre_'+field]
                # Last row
                else:
                    line_dict[i]['timestamp_to'] = line_dict[-1]['timestamp_from']
                    line_dict[i]['minutes_elapsed'] = np.float64("%.2f" %  \
                        (pd.Timedelta(pd.Timestamp(line_dict[-1]['timestamp_from']) - \
                                      pd.Timestamp(line_dict[i]['timestamp_from'])).seconds / 60))
                    # Get values from next in session quest
                    line_dict[i] = {key: (None if key in sc.schema_in.names else line_dict[i][key]) for key in line_dict[i].keys()}
                    
                
                table_writings = add_to_parquet(table_writings, line_dict[i], sc.schema_selfrep)
                   
            #######################################
            # Extract from sensors
            
            current_id_cs = [line_dict[i]['id'] for i in range(num_intervals)]
            current_activities = [line_dict[i]['hand_activity'] for i in range(num_intervals)]
            current_intervals = [line_dict[i]['interval_index'] for i in range(num_intervals)]
            
            dict_notif = {field.name: None for field in sc.schema_notifs}
            dict_sensory = [{field.name: None for field in sc.schema_sensor}\
                         for i in range(num_intervals)]
            
            sensor_writings = 0 #number of extracted sensory requests from "CXTDIS SR 01 Sensors", represents interval index
            num_usage = 0
            
            for i in range(len(session['sensor_requests'])):
                
                # Usage data
                if session['sensor_requests'][i]['sensor_request']['id'] == 2: #name "CXXTDIS SR 02_usage" 
                    if num_usage == 0: #if not yet extracted any usage data
                        num_usage += 1
                        
                        for sensor_type in session['sensor_requests'][i]['sensor_services']:  
                            if sensor_type['sensor_service']['service_type'] == 5:
                                # datasource:  #0: usage watch, 1: usage phone
                                ds_usage = sensor_type['sensor_service']['datasource']
                            
                                for sensor in sensor_type['sensors']: 
                                    
                                        if sensor['sensor_type'] == -6:      
                                            used_apps = []
                                            for app in sensor['sensor_events'][0]['value']:
                                                    # App was used
                                                    if (app['total_time_in_foreground'] > 0):
                                                    #\ or (app['total_time_visible'] > 0):  
                                                    
                                                        app_name = app['package_name']
                                                        # Skip if app is recording app for example 'de.dipf.edutec.edutex.androidclient'
                                                        if ('edutec' or 'edutex') in app_name:
                                                            pass
                                                        elif ('com.google.android.wearable' or 'com.google.android.apps.wearable' or
                                                        'com.mobvoi.wear' or 'com.google.android.packageinstaller' or
                                                        'app.launcher') in app_name:
                                                            pass
                                                        # 'com.google.android.wearable.app'
                                                        # 'com.google.android.apps.wearable.phone'
                                                        # 'com.google.android.apps.wearable.settings'
                                                        # 'com.mobvoi.wear.privacy.aw'
                                                        # 'com.google.android.packageinstaller'
                                                        # 'com.sec.android.app.launcher'
                                                       
                                                        else:
                                                            used_apps.append([app_name, np.int32(app['total_time_in_foreground'])])
                                                            
                                            if len(used_apps[ds_usage]) > 0:          
                                                # Sort so most used apps are up front 
                                                used_apps = sorted(used_apps, key=lambda x: x[1], reverse=True)
                                                
                                                if used_apps == []: used_apps = None
                                                
                                                write_usage_to_parquet(used_apps, session_id, mappings.datasource_decoded[ds_usage], user_id, sc.schema_usage1, name = 'usage-6', file_suffix=file_suffix)
                                        
                                        # List of launched apps
                                        if sensor['sensor_type'] == -10: 
                                            used_apps = []
                                            for app in sensor['sensor_events'][0]['value']:
                                                    # App was used
                                                    if (app['time_in_foreground'] > 0):
                                                    #\ or (app['total_time_visible'] > 0):  
                                                    
                                                        app_name = app['package_name']
                                                        # Skip if app is recording app for example 'de.dipf.edutec.edutex.androidclient'
                                                        if ('edutec' or 'edutex') in app_name:
                                                            pass
                                                        elif ('com.google.android.wearable' or 'com.google.android.apps.wearable' or
                                                        'com.mobvoi.wear' or 'com.google.android.packageinstaller' or
                                                        'app.launcher') in app_name:
                                                            pass
                                                        # 'com.google.android.wearable.app'
                                                        # 'com.google.android.apps.wearable.phone'
                                                        # 'com.google.android.apps.wearable.settings'
                                                        # 'com.mobvoi.wear.privacy.aw'
                                                        # 'com.google.android.packageinstaller'
                                                        # 'com.sec.android.app.launcher'
                                                       
                                                        else:
                                                            used_apps.append([app_name, np.int32(app['time_in_foreground']), np.int32(app['launch_count'])])
                                                            
                                            if len(used_apps) > 0:        
                                                # Sort so most used apps are up front 
                                                used_apps = sorted(used_apps, key=lambda x: x[1], reverse=True)
                                                                                            
                                                write_usage_to_parquet(used_apps, session_id, mappings.datasource_decoded[ds_usage], user_id, sc.schema_usage2, name='usage-10', file_suffix=file_suffix)
                                        
                                        # Notifications, only exists for datasource phone
                                        elif sensor['sensor_type'] == -7: 
                                            num_notifs = [0,0] 
                                            num_muted = [0,0]
                                            for notif in sensor['sensor_events']:
                                                    #app was used
                                                    if (notif['value']['notification_posted']):
    
                                                        app_name = notif['value']['package_name']
                                                        #skip if app is recording app for example 'de.dipf.edutec.edutex.androidclient'
                                                        if ('edutec' or 'edutex') in app_name:
                                                            pass
                                                        elif ('com.google.android.wearable' or 'com.google.android.apps.wearable' or
                                                        'com.mobvoi.wear' or 'com.google.android.packageinstaller' or
                                                        'app.launcher') in app_name:
                                                            pass
                                                        # 'com.google.android.wearable.app'
                                                        # 'com.google.android.apps.wearable.phone'
                                                        # 'com.google.android.apps.wearable.settings'
                                                        # 'com.mobvoi.wear.privacy.aw'
                                                        # 'com.google.android.packageinstaller'
                                                        # 'com.sec.android.app.launcher'
                                                       
                                                        # Add notification
                                                        else:
                                                            num_notifs[ds_usage] += 1
                                                            #used_apps[ds_usage] = np.append(used_apps[ds_usage], [app_name, np.float64(app['total_time_in_foreground'])], axis=0)
                                                            if "vibrate=null" in notif['value']['notification'] and\
                                                               "sound=null" in notif['value']['notification']:
                                                                   num_muted[ds_usage] += 1
                                                    
                                            if num_notifs[ds_usage] > 0:    
                                                # write/save one row
                                                dict_notif['user_id'] = pa.array([user_id], type=sc.schema_notifs.field('user_id').type)
                                                dict_notif['session_id'] = pa.array([session_id], type=sc.schema_notifs.field('session_id').type)
                                                dict_notif['num_notifications'] = np.int32(num_notifs[ds_usage])
                                                dict_notif['num_muted_notifs'] = np.int32(num_muted[ds_usage])
    
                                                table_writings_notifs = add_to_parquet(table_writings_notifs, dict_notif, sc.schema_notifs)
                                                
                                            
                    # If already written, just count up and pass 
                    else: num_usage += 1
                
                # From sensors measured every 30 minutes
                elif session['sensor_requests'][i]['sensor_request']['id'] == 1: #name "CXTDIS SR 01 Sensors"
                    
                    for sensor_type in session['sensor_requests'][i]['sensor_services']:
                        service_type = sensor_type['sensor_service']['service_type']
                        datasource = sensor_type['sensor_service']['datasource']
                        
                        for sensor in sensor_type['sensors']:
                            #sensor_type_id = sensor['sensor_type']
                        
                            out, out_name = extract(service_type, datasource, sensor, sc.schema_sensor_names.names)
                
                            if out_name.startswith(('acc', 'gyr', 'mag')):
                                # Write this one sequence into one file
                                seq_length = write_seq_to_parquet(out, out_name, current_id_cs[sensor_writings], 
                                                                  session_id,  mappings.datasource_decoded[datasource], 
                                                                  user_id, current_activities[sensor_writings],
                                                                  sc.schema_xyz, interpolation=interpolation, 
                                                                  upsample=upsample, file_suffix=file_suffix)
                                # Write/save len of this sequence
                                dict_sensory[i-num_usage]['seq_len_'+out_name] = pa.array([seq_length], type=sc.schema_sensor.field('seq_len_'+out_name).type)
                            
                            elif out_name !='':
                                dict_sensory[i-num_usage][out_name] = pa.array([out], type=sc.schema_sensor.field(out_name).type)
                                
                    # Get basic information / identifiers for that row
                    dict_sensory[i-num_usage]['id'] = pa.array([current_id_cs[sensor_writings]], type=sc.schema_sensor.field('id').type)
                    dict_sensory[i-num_usage]['user_id'] = pa.array([user_id], type=sc.schema_sensor.field('user_id').type)
                    dict_sensory[i-num_usage]['session_id'] = pa.array([session_id], type=sc.schema_sensor.field('session_id').type)
                    dict_sensory[i-num_usage]['interval_index'] = pa.array([current_intervals[sensor_writings]], type=sc.schema_sensor.field('interval_index').type)
                    # Write/save one row
                    table_writings_sensors = add_to_parquet(table_writings_sensors, dict_sensory[i-num_usage], sc.schema_sensor)
                    
                    # Count up
                    sensor_writings += 1
                    
                # It could be that there are more sensory data elevations than place to write
                if sensor_writings == (num_intervals-num_usage): break 
                
            # If there are not as many intervals as sensor recordings, something must have gone wrong
            # I subtract num_usage and this num should be one but sometimes it is mistakenly two
            if not (num_intervals >= (len(session['sensor_requests'])-num_usage)): 
                warnings.warn(f"\n Number intervals: {num_intervals}, Sensory recordings (w/o usage): {(len(session['sensor_requests'])-num_usage)}\n",  UserWarning)
    
    
    # Write parquet non sequential data
    # Write to selfreport file
    write_to_parquet(table_writings, path_all)
    # Write to notifs file
    write_to_parquet(table_writings_notifs, path_all_notifs)
    
    # Before saving sensor data, delete all empty columns
    columns_to_drop = [] #temp array for collecting empty columns
    for i, column in enumerate(table_writings_sensors.columns):
        # Check whether number of null values matches number of all possible rows
        if column.null_count == table_writings_sensors.num_rows:
            columns_to_drop.append(table_writings_sensors.schema.names[i])
    # Delete collected columns
    table_writings_sensors = table_writings_sensors.drop(columns_to_drop)
    
    # Write parquet
    write_to_parquet(table_writings_sensors, path_all_sensors)

if __name__ == "__main__":
    main()