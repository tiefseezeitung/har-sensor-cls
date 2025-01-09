#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load CXT22/WISDM dataset, predict whole set with chosen model, 
trained on HA24 dataset, and evaluate it.
"""

# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
# Import modules
import eval_utils
import load_data as ld

def main():
    script_dir = os.getcwd()+'/'
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.insert(0, parent_dir)
    
    seq_sensors = ["acc_p", "acc_w", "gyr_p", "gyr_w"]
    #seq_sensors = ["acc_w", "gyr_w"]
    #seq_sensors = ["acc_p", "gyr_p"]
    #seq_sensors = ["acc_w"]
    
    # dictionary for matching integer labels
    label_mapping = {'drinking': 0,
                     'eating': 1,
                     'fidgeting': 2,
                     'idle_hands': 3,
                     'making_a_phone_call': 4,
                     'reading_a_book': 5,
                     'scratching': 6,
                     'scrolling_on_a_smartphone': 7,
                     'scrolling_on_a_tablet': 8,
                     'typing_on_a_smartphone': 9,
                     'typing_on_a_tablet': 10,
                     'typing_on_the_keyboard': 11,
                     'using_the_computer_mouse': 12,
                     'using_the_touchpad': 13,
                     'writing_with_a_pen': 14}
    
    all_labels_wspace = [label.replace('_', ' ') for label in list(label_mapping.keys())]
    
    windows = True
    seconds = 10
    # XXX
    target_lens = [15239, 15629, 15213, 15628]
    
    ds_name = {'2': 'CXT22', '3': 'WISDM'}
    # XXX
    dataset = '2'
    load_type = '2'
    
    interpolate = True
    
    # CXT22
    if dataset == '2':
    
        dictionary = {'stille_Hände': 'idle_hands', 
                    'Tippen_am_Smartphone': 'typing_on_a_smartphone', 
                    'Tippen_am_Tablet': 'typing_on_a_tablet',
                    'Tippen_am_Computer': 'typing_on_the_keyboard', 
                    'Schreiben_mit_Stift': 'writing_with_a_pen', 
                    'Zappeln': 'fidgeting'}
        
        chosen_activities = list(dictionary.keys())
        ds2_dir = os.getcwd()+'/CXT22/'
        sys.path.insert(0, ds2_dir)
    
    # WISDM
    elif dataset=='3':
        import wisdm_activities as wa
        
        # Adjust lengths for the different recording time (150->180 seconds)
        target_lens = [x * 1.2 for x in target_lens]
        wisdm_activities_matching_st24 = np.unique(list(wa.mapping_to_ha24.values()))
        chosen_activities = list(wa.mapping_to_ha24.keys())
        dictionary = wa.mapping_to_ha24
        
        ds3_dir = os.getcwd()+'/WISDM/'
        sys.path.insert(0, ds3_dir)
    
    X_data_extra, labels_extra, add_lst_extra, feats_extra = ld.get_inputs(
                     dataset, load_type, 
                     sensor_names=seq_sensors,
                     chosen_activities=chosen_activities, 
                     dictionary = dictionary,
                     sliding_window=windows,
                     window_seconds=seconds, 
                     interpolate=interpolate,
                     desired_lens=target_lens)
    
    for i in range(len(X_data_extra)):
        print('Shape of array '+seq_sensors[i]+':', X_data_extra[i].shape)
    
    ###############################################################################
    # Save additional (meta) data
    
    all_labels_this_extra = sorted(np.unique(labels_extra))
    all_labels_this_wspace_extra = [label.replace('_', ' ') for label in all_labels_this_extra]
    
    int_labels_extra = np.array([label_mapping[str_label] for str_label in labels_extra])
    
    # Print label names and the num samples 
    print('\n{:25s} {}'.format('Label name', 'Num samples'))
    print('-------------------------------------------')
    for (label_str, num_samples) in zip(np.unique(labels_extra, return_counts=True)[0], np.unique(labels_extra, return_counts=True)[1]):
        #print(label_str, '\t', num_samples)
        print('{:25s} {:3d}'.format(label_str, num_samples))
    print()
    
    
    # Load model from keras file
    from keras.models import load_model
    models_directory = script_dir + 'trained_models/'
    name = models_directory + 'ds1-ViT1-actsep-8b-no'
    model = load_model(name+'.keras')
    
    loss, accuracy, top3 = model.evaluate(X_data_extra, int_labels_extra, verbose=0)
    print('Results for Dataset '+ds_name[dataset]+':')
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Top 3 Accuracy: {top3:.4f}\n")
    
    # Predict class probabilities using the model
    y_pred_probs = model.predict(X_data_extra)
    
    # Macro precision, recall, f1 scores and confusion matrix cf
    precision, recall, f1, cf = eval_utils.print_extended_evaluation(
        y_pred_probs, int_labels_extra, labels_extra, all_labels_this_extra, label_mapping
        )
    
    # Plot confusion matrix
    sns.set(rc={'figure.figsize':(20,14)})
    fig, ax = plt.subplots()
    sns.heatmap(cf, ax=ax, annot=True, cmap='coolwarm',fmt='g', annot_kws={"fontsize":22},
                xticklabels=all_labels_wspace, yticklabels=all_labels_wspace, vmax=sum(cf[0]))
    plt.xticks(rotation=45, ha="right",fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Predicted', fontsize=30)
    plt.ylabel('Actual', fontsize=30)
    plt.title('Confusion Matrix for '+ds_name[dataset], fontsize=35)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=22)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
##################################################################
# Outputs:
    
#Results for Dataset CXT22:
#Loss: 13.3323
#Accuracy: 0.1062
#Top 3 Accuracy: 0.1679

# WISDM (first 10 sec are removed from each sequence)
#Results for Dataset WISDM:
#Loss: 19.1863
#Accuracy: 0.1739
#Top 3 Accuracy: 0.5673


