#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimenting with Random Forest to assess performances of features.
"""

# Import libraries
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Import modules
import eval_utils  
import load_data as ld

def main(): 
    
    # Settings for loading data and labels 
    # XXX Define sensors to load data from
    seq_sensors = ["acc_p", "acc_w", "gyr_p", "gyr_w"]
    #seq_sensors = ["acc_w", "gyr_w"]
    #seq_sensors = ["acc_p", "gyr_p"]
    #seq_sensors = ["acc_w"]
    
    chosen_activities = 'all'
    dictionary = {}
    
    windows = True
    # XXX Define window length defined in seconds
    seconds = 10
    # XXX '1' for sequences, '2' for spectrograms
    load_type = '1'
    # XXX dataset
    ds = '1'
    
    if ds == '1':
        interpolate = False
        
    elif ds == '2':
        # activities matching HA24 categories:
        #chosen_activities = ['stille Hände', 'Tippen am Smartphone', 'Tippen am Tablet',
        #                  'Tippen am Computer', 'Schreiben mit Stift', 'Zappeln']
        interpolate = True  
    
    elif ds == '3': 
        ds3_dir = os.getcwd()+'/WISDM/'
        sys.path.insert(0, ds3_dir)
        from WISDM import wisdm_activities as wa
        # Activity lists from wa: wa.learn_hand_activities /  wa.hand_activities
        
        interpolate = True
        dictionary = wa.activities_dict # string names
    
    
    _, labels, _, feats = ld.get_inputs(ds, load_type, 
                     sensor_names=seq_sensors,
                     chosen_activities=chosen_activities, 
                     dictionary = dictionary,
                     sliding_window=windows,
                     window_seconds=seconds, 
                     interpolate=interpolate)
    
    ######
    # Map each category to integer number 
    all_labels_this = sorted(np.unique(labels)) #sorted unique label list
    label_mapping = {label: i for i, label in enumerate(all_labels_this)} 
    int_labels = np.array([label_mapping[str_label] for str_label in labels])
    
    # Split features and label data into training and validation set
    
    num_samples = len(labels)
    
    # Shuffle data
    seed = 46
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(num_samples)
    
    feats_shuffled = []
    for i in range(len(feats)):
        feats_shuffled.append(feats[i].iloc[shuffle_indices])
    labels_int_shuffled = int_labels[shuffle_indices]
    #labels_shuffled = labels[shuffle_indices]   
    
    # XXX index of chosen sensor corresponding to seq_sensors list
    sensor_idx = 3
    
    # Random Forest
    num_train = int(0.8 * num_samples)
    train_input = feats_shuffled[sensor_idx][:num_train]
    train_y = labels_int_shuffled[:num_train]
    val_input = feats_shuffled[sensor_idx][num_train:]
    val_y = labels_int_shuffled[num_train:]
    
    ###########################################################################
    # Test with Random Forest, inspect most contributing features
    
    model = RandomForestRegressor(n_estimators=30, random_state=22, verbose=0)
    model.fit(train_input, train_y)
    y_pred = model.predict(val_input)
    
    # Print MSE and MAE results
    _, _ = eval_utils.calc_regress_evaluation(val_y, y_pred, model_name='Random Forest')
    
    # Inspect which features have most influence 
    importances = model.feature_importances_
    indices_sorted = np.argsort(importances)[::-1]
    
    # Print feature importances
    print(f"Feature Importances for sensor {seq_sensors[sensor_idx]}:")
    for i in indices_sorted:
        print(f"{format(importances[i],'.4f')} - {feats_shuffled[1].columns[i]} (Index {i})")

if __name__ == "__main__":
    main()

"""
sensors ranked, based on val loss: 1. acc_w, 2. acc_p, 3. gyr_w, 4. gyr_p

Results
Feature Importances for sensor acc_p:
0.0876 - FFT1_Y (Index 31)
0.0648 - Mean_X (Index 0)
0.0628 - FFT1_X (Index 21)
0.0567 - Median_X (Index 3)
0.0528 - Cos_XY (Index 15)
0.0521 - Cos_XZ (Index 16)
0.0434 - Max_X (Index 9)
0.0417 - Cos_YZ (Index 17)
0.0367 - Mean_Y (Index 1)
0.0349 - Absoldev_Z (Index 14)
0.0301 - FFT1_Z (Index 41)
0.0283 - Median_Y (Index 4)
0.0273 - Mean_Z (Index 2)
0.0242 - Corr_XZ (Index 19)
0.0209 - Max_Y (Index 10)
0.0205 - Absoldev_X (Index 12)
0.0201 - Std_Z (Index 8)
0.0196 - Absoldev_Y (Index 13)
0.0187 - Corr_YZ (Index 20)
0.0184 - Max_Z (Index 11)
0.0176 - Std_X (Index 6)
0.0157 - FFT2_X (Index 22)
0.0144 - Corr_XY (Index 18)
0.0123 - Std_Y (Index 7)
0.0122 - Median_Z (Index 5)
0.0104 - FFT9_X (Index 29)
0.0096 - FFT10_X (Index 30)
0.0085 - FFT2_Y (Index 32)
0.0080 - FFT3_X (Index 23)
0.0077 - FFT7_X (Index 27)
0.0071 - FFT4_Y (Index 34)
0.0070 - FFT2_Z (Index 42)
0.0068 - FFT4_X (Index 24)
0.0067 - FFT5_Z (Index 45)
0.0066 - FFT5_Y (Index 35)
0.0065 - FFT6_Y (Index 36)
0.0064 - FFT9_Z (Index 49)
0.0062 - FFT8_Z (Index 48)
0.0061 - FFT4_Z (Index 44)
0.0061 - FFT10_Z (Index 50)
0.0061 - FFT6_Z (Index 46)
0.0058 - FFT3_Z (Index 43)
0.0055 - FFT3_Y (Index 33)
0.0053 - FFT8_X (Index 28)
0.0052 - FFT9_Y (Index 39)
0.0051 - FFT5_X (Index 25)
0.0050 - FFT10_Y (Index 40)
0.0048 - FFT7_Y (Index 37)
0.0047 - FFT7_Z (Index 47)
0.0046 - FFT6_X (Index 26)
0.0042 - FFT8_Y (Index 38)

Feature Importances for sensor acc_w:
0.1548 - FFT2_Y (Index 32)
0.1038 - FFT2_X (Index 22)
0.0699 - FFT1_X (Index 21)
0.0623 - Std_Y (Index 7)
0.0607 - Corr_XY (Index 18)
0.0311 - Absoldev_Y (Index 13)
0.0297 - Cos_XZ (Index 16)
0.0293 - FFT3_Y (Index 33)
0.0273 - Median_X (Index 3)
0.0259 - Max_Z (Index 11)
0.0257 - Cos_XY (Index 15)
0.0252 - Absoldev_Z (Index 14)
0.0215 - Corr_YZ (Index 20)
0.0204 - Std_Z (Index 8)
0.0191 - Max_Y (Index 10)
0.0185 - Mean_Z (Index 2)
0.0180 - FFT1_Z (Index 41)
0.0179 - Median_Y (Index 4)
0.0175 - FFT1_Y (Index 31)
0.0162 - Cos_YZ (Index 17)
0.0159 - Mean_X (Index 0)
0.0146 - Max_X (Index 9)
0.0134 - Mean_Y (Index 1)
0.0110 - Median_Z (Index 5)
0.0106 - Corr_XZ (Index 19)
0.0094 - FFT3_X (Index 23)
0.0077 - Absoldev_X (Index 12)
0.0073 - FFT6_X (Index 26)
0.0072 - FFT4_Z (Index 44)
0.0071 - FFT5_Y (Index 35)
0.0068 - FFT10_X (Index 30)
0.0064 - FFT4_Y (Index 34)
0.0063 - FFT2_Z (Index 42)
0.0061 - FFT7_X (Index 27)
0.0057 - FFT7_Y (Index 37)
0.0056 - FFT9_Z (Index 49)
0.0053 - FFT8_Z (Index 48)
0.0052 - FFT9_Y (Index 39)
0.0051 - Std_X (Index 6)
0.0049 - FFT9_X (Index 29)
0.0047 - FFT3_Z (Index 43)
0.0046 - FFT10_Z (Index 50)
0.0045 - FFT10_Y (Index 40)
0.0044 - FFT5_X (Index 25)
0.0042 - FFT6_Y (Index 36)
0.0040 - FFT7_Z (Index 47)
0.0040 - FFT8_X (Index 28)
0.0037 - FFT5_Z (Index 45)
0.0033 - FFT6_Z (Index 46)
0.0032 - FFT4_X (Index 24)
0.0030 - FFT8_Y (Index 38)
    
Feature Importances for sensor gyr_p:
0.0600 - Absoldev_Y (Index 13)
0.0350 - Absoldev_X (Index 12)
0.0331 - FFT2_Z (Index 42)
0.0322 - Max_Y (Index 10)
0.0316 - FFT1_Z (Index 41)
0.0300 - FFT1_X (Index 21)
0.0273 - Std_Y (Index 7)
0.0236 - Max_X (Index 9)
0.0230 - Std_Z (Index 8)
0.0220 - FFT8_Y (Index 38)
0.0219 - FFT9_Z (Index 49)
0.0212 - FFT9_X (Index 29)
0.0205 - Cos_XY (Index 15)
0.0201 - FFT4_Y (Index 34)
0.0199 - Std_X (Index 6)
0.0199 - FFT2_Y (Index 32)
0.0198 - FFT7_Y (Index 37)
0.0197 - FFT1_Y (Index 31)
0.0196 - Absoldev_Z (Index 14)
0.0190 - Corr_XY (Index 18)
0.0185 - Corr_XZ (Index 19)
0.0183 - FFT5_Y (Index 35)
0.0180 - FFT4_X (Index 24)
0.0178 - Mean_Y (Index 1)
0.0178 - FFT2_X (Index 22)
0.0176 - Max_Z (Index 11)
0.0176 - FFT3_Z (Index 43)
0.0171 - Mean_X (Index 0)
0.0171 - Mean_Z (Index 2)
0.0169 - Median_Y (Index 4)
0.0168 - FFT7_Z (Index 47)
0.0168 - FFT6_Z (Index 46)
0.0167 - FFT5_Z (Index 45)
0.0163 - FFT8_Z (Index 48)
0.0162 - FFT3_X (Index 23)
0.0160 - FFT8_X (Index 28)
0.0156 - FFT6_X (Index 26)
0.0153 - FFT10_Y (Index 40)
0.0153 - FFT9_Y (Index 39)
0.0150 - Cos_YZ (Index 17)
0.0149 - Cos_XZ (Index 16)
0.0148 - Corr_YZ (Index 20)
0.0146 - FFT3_Y (Index 33)
0.0145 - FFT10_X (Index 30)
0.0140 - FFT4_Z (Index 44)
0.0140 - FFT10_Z (Index 50)
0.0140 - FFT5_X (Index 25)
0.0137 - FFT7_X (Index 27)
0.0116 - Median_Z (Index 5)
0.0109 - FFT6_Y (Index 36)
0.0072 - Median_X (Index 3)

Feature Importances for sensor gyr_w:
0.2193 - FFT2_X (Index 22)
0.1060 - FFT2_Z (Index 42)
0.0875 - Absoldev_X (Index 12)
0.0484 - Cos_XY (Index 15)
0.0400 - Std_X (Index 6)
0.0265 - Corr_XY (Index 18)
0.0247 - FFT3_X (Index 23)
0.0234 - Absoldev_Y (Index 13)
0.0175 - FFT1_X (Index 21)
0.0174 - Absoldev_Z (Index 14)
0.0164 - FFT4_X (Index 24)
0.0159 - Corr_YZ (Index 20)
0.0155 - Cos_XZ (Index 16)
0.0153 - FFT1_Y (Index 31)
0.0146 - Mean_X (Index 0)
0.0130 - Max_X (Index 9)
0.0127 - Max_Z (Index 11)
0.0126 - Cos_YZ (Index 17)
0.0125 - Corr_XZ (Index 19)
0.0114 - FFT3_Z (Index 43)
0.0111 - Max_Y (Index 10)
0.0108 - FFT1_Z (Index 41)
0.0103 - FFT7_X (Index 27)
0.0103 - FFT5_X (Index 25)
0.0100 - Mean_Y (Index 1)
0.0098 - Mean_Z (Index 2)
0.0097 - FFT5_Z (Index 45)
0.0096 - Median_X (Index 3)
0.0094 - FFT2_Y (Index 32)
0.0090 - FFT6_X (Index 26)
0.0086 - FFT7_Z (Index 47)
0.0084 - Median_Y (Index 4)
0.0084 - Std_Y (Index 7)
0.0083 - FFT9_X (Index 29)
0.0082 - Std_Z (Index 8)
0.0079 - FFT10_Y (Index 40)
0.0075 - FFT10_Z (Index 50)
0.0074 - FFT4_Z (Index 44)
0.0072 - Median_Z (Index 5)
0.0070 - FFT6_Y (Index 36)
0.0069 - FFT10_X (Index 30)
0.0068 - FFT4_Y (Index 34)
0.0068 - FFT8_Z (Index 48)
0.0067 - FFT8_X (Index 28)
0.0067 - FFT6_Z (Index 46)
0.0066 - FFT9_Z (Index 49)
0.0064 - FFT7_Y (Index 37)
0.0062 - FFT5_Y (Index 35)
0.0060 - FFT9_Y (Index 39)
0.0060 - FFT8_Y (Index 38)
0.0054 - FFT3_Y (Index 33)
"""


