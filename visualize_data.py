#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Sequences:
Plot 3 dimensions of the sequences/windows or patches of the spectrogram images.
"""

# Import libraries#
import os
import sys
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
from math import ceil
# Import modules
import load_data as ld
script_dir = os.getcwd()+'/'


sensor_name = {'acc_p': 'accelerometer phone', 'acc_w': 'accelerometer watch',
               'gyr_p': 'gyroscope phone', 'gyr_w': 'gyroscope watch',
               'mag_p': 'magnetometer phone', 'mag_w': 'magnetometer watch'}

sensor_name_short = {'acc_p': 'Acc. Phone', 'acc_w': 'Acc. Watch',
               'gyr_p': 'Gyr. Phone', 'gyr_w': 'Gyr. Watch',
               'mag_p': 'Mag. Phone', 'mag_w': 'Mag. Watch'}

def plot_seq(seq, seconds, label, user, type_name, data_index, record_index, 
             boundary_vals, extra_text="", save_file=False):
    fig, ax1 = plt.subplots()
    ax1.plot(seq[:,0])
    ax1.plot(seq[:,1])
    ax1.plot(seq[:,2])
    ax1.set_ylabel('value')
    lim_min, lim_max = boundary_vals[type_name]
    ax1.set_ylim(lim_min, lim_max)
    ax1.set_xlabel('sequence length')
    ax1.set_xlim(0, len(seq[:,0]))
    ax1.legend(['X', 'Y', 'Z'], loc='center left', bbox_to_anchor=(1, 0.89))

    ax2 = ax1.twiny()
    ax2.set_xlabel('time [seconds]')
    ax2.set_xlim(0, seconds)
    ax2.set_xticks(range(0, seconds+1, seconds//5))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    # Adjust the position of bottom axis as needed
    ax2.spines['bottom'].set_position(('outward', 36))

    plt.title(sensor_name_short[type_name]+' - Activity "'+label.replace("_", " ") +
              '" - User '+str(user)+extra_text, pad=13)
    if save_file: 
        plt.savefig(script_dir+str(type_name)+"_activity_"+str(label)+"_user_"+\
                    str(user)+'_i_'+str(data_index)+'_rec_'+str(record_index)+'.png')

    plt.show()
    
def plot_fig_grid(seqs, seconds, label, user, type_names, data_index, record_index, 
                  boundary_vals, extra_text="", save_file=False):
    mpl.rcParams['font.size'] = 14
    assert (len(seqs) > 2)

    # calculate width and height of figure
    if len(seqs) % 2 == 0: fig_width = len(seqs) // 2 
    else: fig_width = (len(seqs)+1) // 2 
    fig_height = ceil(len(seqs) / fig_width)
   
    # calculate remainder for number of plots to be plotted in last row 
    if len(seqs) % fig_width == 0: last_row_plots = fig_width 
    else: last_row_plots = len(seqs) % fig_width
    
    # initialize figure grid
    fig, axs = plt.subplots(fig_width, fig_height, figsize=(12, 10))
    
    c = 0
    for i in range(len(axs)):
        if (i == len(axs)-1):
            num_plots_row_i = last_row_plots
        else: num_plots_row_i = len(axs[i])
        for j in range(num_plots_row_i):
            axs[i,j].plot(seqs[c][:,0])
            axs[i,j].plot(seqs[c][:,1])
            axs[i,j].plot(seqs[c][:,2])
            axs[i,j].set_ylabel('value', fontsize=12)
            
            lim_min, lim_max = boundary_vals[type_names[c]]
            axs[i,j].set_ylim(lim_min, lim_max)
            axs[i,j].set_xlabel('sequence length', fontsize=12)
            axs[i,j].set_xlim(0, len(seqs[c][:,0]))
            axs[i,j].tick_params(axis='x', labelsize=12)
            axs[i,j].tick_params(axis='y', labelsize=12)

            axs0 = axs[i,j].twiny()
            axs0.set_xlabel('time [seconds]', fontsize=12)
            axs0.set_xlim(0, seconds)
            axs0.set_xticks(range(0, seconds+1, seconds//5))
            axs0.tick_params(axis='x', labelsize=12)
            axs0.xaxis.set_ticks_position('bottom')
            axs0.xaxis.set_label_position('bottom')
            # Adjust the position as needed
            axs0.spines['bottom'].set_position(('outward', 42))

            plt.title(sensor_name_short[type_names[c]], pad=13,
                      fontsize=15, horizontalalignment='center')
            c += 1
        if (i == len(axs)-1):
            # call all remaining axes without plots off for the figure
            for j in range(len(axs[i])-last_row_plots):
                axs[i, j+last_row_plots].axis('off')
        
    fig.legend(['X', 'Y', 'Z'], loc='center left', bbox_to_anchor=(0.98, 0.855), fontsize=14)
    fig.suptitle('Activity "'+ label.replace("_", " ")  +
              '" - User '+str(user)+extra_text,
              fontsize=18, horizontalalignment='center')
    plt.tight_layout()
    if save_file: 
        plt.savefig(script_dir+str("-".join(type_names))+"_activity_"+str(label)+\
                    "_user_"+str(user)+'_i_'+str(data_index)+'_rec_'+str(record_index)+'.png')

    plt.show()
    
def plot_fig_row(seqs, seconds, label, user, type_names, data_index, record_index, 
                 boundary_vals, extra_text="", save_file=False):
    mpl.rcParams['font.size'] = 12 

    fig, axs = plt.subplots(1, len(seqs), figsize=(18, 6))

    for i in range(len(axs)):
        axs[i].plot(seqs[i][:,0])
        axs[i].plot(seqs[i][:,1])
        axs[i].plot(seqs[i][:,2])
        axs[i].set_ylabel('value')
        lim_min, lim_max = boundary_vals[type_names[i]]
        axs[i].set_ylim(lim_min, lim_max)
        axs[i].set_xlabel('sequence length')
        axs[i].set_xlim(0, len(seqs[i][:,0]))
    
        axs0 = axs[i].twiny()
        axs0.set_xlabel('time [seconds]')
        axs0.set_xlim(0, seconds)
        axs0.set_xticks(range(0, seconds+1, seconds//5))
        axs0.xaxis.set_ticks_position('bottom')
        axs0.xaxis.set_label_position('bottom')
        # Adjust position
        axs0.spines['bottom'].set_position(('outward', 40))

        plt.title(sensor_name_short[type_names[i]], pad=13,
                  fontsize=12, horizontalalignment='center')
        
    fig.legend(['X', 'Y', 'Z'], loc='center left', bbox_to_anchor=(1, 0.8), fontsize=14)

    fig.suptitle('Activity "'+ label.replace("_", " ")  +
              '" - User '+str(user)+extra_text,
              fontsize=16, horizontalalignment='center')
    plt.tight_layout()
    if save_file: 
        plt.savefig(script_dir+str("-".join(type_names))+"_activity_"+str(label)+\
                    "_user_"+str(user)+'_i_'+str(data_index)+'_rec_'+str(record_index)+'.png')

    plt.show()

#%% settings for loading data and labels 
# XXX Define sensors to load data from
seq_sensors = ["acc_p", "acc_w", "gyr_p", "gyr_w"]
#seq_sensors = ["acc_w", "gyr_w"]
#seq_sensors = ["acc_p", "gyr_p"]
#seq_sensors = ["acc_w"]

chosen_activities = 'all'
dictionary = {}

# XXX Division into windows
windows = False
# XXX Define window length defined in seconds
seconds = 10
# XXX '1' for sequences, '2' for spectrograms
load_type = '1'
# XXX dataset
ds = '1'

if ds == '1':
    recording_time = 150
    interpolate = False
    
    boundary_vals = {'acc_p': (-5, 14), 'acc_w': (-21, 19), 
                     'gyr_p': (-6, 6), 'gyr_w': (-6, 6)}
    
elif ds == '2':
    recording_time = 150
    # activities matching HA24 categories:
    #chosen_activities = ['stille Hände', 'Tippen am Smartphone', 'Tippen am Tablet',
    #                  'Tippen am Computer', 'Schreiben mit Stift', 'Zappeln']
    interpolate = True  
    
    boundary_vals = {'acc_p': (-9, 17), 'acc_w': (-16, 15), 
                     'gyr_p': (-3, 3), 'gyr_w': (-6, 6)}

elif ds == '3': 
    recording_time = 180
    ds3_dir = os.getcwd()+'/WISDM/'
    sys.path.insert(0, ds3_dir)
    from WISDM import wisdm_activities as wa
    # Activity lists from wa: wa.learn_hand_activities /  wa.hand_activities
    
    interpolate = True
    dictionary = wa.activities_dict # string names
    
    boundary_vals = {'acc_p': (-15, 14), 'acc_w': (-15, 15), 
                     'gyr_p': (-3, 3), 'gyr_w': (-8, 7)}

X_data, labels, add_lst, feats = ld.get_inputs(ds, load_type, 
                 sensor_names=seq_sensors,
                 chosen_activities=chosen_activities, 
                 dictionary = dictionary,
                 sliding_window=windows,
                 window_seconds=seconds, 
                 interpolate=interpolate)

for i in range(len(X_data)):
    print('Shape of array '+seq_sensors[i]+':', X_data[i].shape)
    
plot_seconds = seconds if windows else recording_time
#%% Prepare data

users = add_lst[0]
activity_ids = add_lst[1]
num_samples = len(labels)
all_labels_this = sorted(np.unique(labels))
num_classes = len(all_labels_this) 
label_mapping = {label: i for i, label in enumerate(all_labels_this)}
int_labels = np.array([label_mapping[str_label] for str_label in labels])

#%% Print max min values for determining limitation of axes
# Values can be set in boundary_vals dictionary
for s in range(len(X_data)):
    
    max_of_each_arr = np.max(np.max(X_data[s], axis=1), axis=1)
    min_of_each_arr = np.min(np.min(X_data[s], axis=1), axis=1)
    max_reccom = np.mean(max_of_each_arr)+np.std(max_of_each_arr)
    min_reccom = np.mean(min_of_each_arr)-np.std(min_of_each_arr)
    print(seq_sensors[s],'\tmean of min/max values -/+ std of min/max values:')
    print(min_reccom)
    print(max_reccom)
    print()

#%% 1) Choose seq and plot

sensor_index = 1
data_index = 15
seq = X_data[sensor_index][data_index]
label = labels[data_index]
user = users[data_index]
type_name = seq_sensors[sensor_index]
if windows:
    num_window_of_act = np.sum(activity_ids[:data_index+1] == activity_ids[data_index])        
    sum_windows_of_act = np.sum(activity_ids == activity_ids[data_index])
    window_index_str = " - Window Number of Recording "+ str(num_window_of_act)+"/"+str(sum_windows_of_act)
else: window_index_str = ""
# Plot
plot_seq(seq, plot_seconds, label, user, type_name, data_index, activity_ids[data_index], boundary_vals, extra_text=window_index_str,)

#%% 2) same activity, same sensor, different/random users
sensor_index = 1
activity = all_labels_this[0] #[14]
indices = np.where(labels == activity)[0]
np.random.seed(46)
indices = np.random.permutation(indices)
c_printed = 0
limit = 9

for i in indices:
    # if limit number of plots printed, stop for loop
    if c_printed == limit:
        break
    seq = X_data[sensor_index][i]
    label = labels[i]
    user = users[i]
    type_name = seq_sensors[sensor_index]
    # print(seq[i].shape)
    
    if windows:
        num_window_of_act = np.sum(activity_ids[:i+1] == activity_ids[i])        
        sum_windows_of_act = np.sum(activity_ids == activity_ids[i])
        window_index_str = " - Window Number of Recording "+ str(num_window_of_act)+ "/"+str(sum_windows_of_act)
    else:
        window_index_str = ""
        
    plot_seq(seq, plot_seconds, label, user, type_name, i, activity_ids[i], 
             boundary_vals, extra_text=window_index_str)

    c_printed += 1

#%% 3) same activity, all sensors, for different/random users
activity = all_labels_this[1]
indices = np.where(labels == activity)[0]
np.random.seed(46)
indices = np.random.permutation(indices)
num_users = 2

for i in indices[:num_users]:
    for s in range(len(seq_sensors)):
        seq = X_data[s][i]
        label = labels[i]
        user = users[i]
        type_name = seq_sensors[s]

        if windows:
            num_window_of_act = np.sum(activity_ids[:i+1] == activity_ids[i])        
            sum_windows_of_act = np.sum(activity_ids == activity_ids[i])
            window_index_str = " - Window Number of Recording "+ str(num_window_of_act)+"/"+str(sum_windows_of_act)
        else:
            window_index_str = ""
            
        plot_seq(seq, plot_seconds, label, user, type_name, i, activity_ids[i], 
                 boundary_vals, extra_text=window_index_str)

#%% 4) Plot figure with multiple plots of one activity
activity = all_labels_this[0]
indices = np.where(labels == activity)[0]
np.random.seed(46)
indices = np.random.permutation(indices)

save_file = False
sensor_index = [0, 1, 2, 3] # indices of seq_sensors
num_users = 4
for i in indices[:num_users]:

        seqs = [X_data[s][i] for s in sensor_index]
        
        label = labels[i]
        user = users[i]
        type_name = [seq_sensors[s] for s in sensor_index]
        
        if windows:
            num_window_of_act = np.sum(activity_ids[:i+1] == activity_ids[i])        
            sum_windows_of_act = np.sum(activity_ids == activity_ids[i])
            window_index_str = " - Window Number of Recording "+ str(num_window_of_act)+"/"+str(sum_windows_of_act)
        else:
            window_index_str = ""
            
        if len(sensor_index) < 3:
            plot_fig_row(seqs, plot_seconds, label, user, type_name, i, activity_ids[i],
                         boundary_vals, extra_text=window_index_str, save_file=save_file)
        else: plot_fig_grid(seqs, plot_seconds, label, user, type_name, i, 
                            activity_ids[i], boundary_vals, 
                            extra_text=window_index_str, save_file=save_file)

#%% 5) Choose a specific index of the data and print all 15 windows
# (if not windows, prints 15 full sequences)

start_index=0
for i in range(start_index, start_index+15, 1):
    activity = labels[i]
    save_file = False
    sensor_index = [0,1,2,3]
    seqs = [X_data[s][i] for s in sensor_index]
    
    label = labels[i]
    user = users[i]
    type_name = [seq_sensors[s] for s in sensor_index]
    # print(seq[i].shape)
    if windows:
        num_window_of_act = np.sum(activity_ids[:i+1] == activity_ids[i])        
        sum_windows_of_act = np.sum(activity_ids == activity_ids[i])
        window_index_str = " - Window Number of Recording "+ str(num_window_of_act)+"/"+str(sum_windows_of_act)
    else:
        window_index_str = ""
    
    plot_fig_grid(seqs, plot_seconds, label, user, type_name, i, activity_ids[i], 
                  boundary_vals, extra_text=window_index_str, save_file=save_file)

#%% Visualize Patches

from keras.layers import Resizing
from keras import ops
import tensorflow as tf

def to_patches(patch_shape, images):
        patch_size_h = patch_shape[0]
        patch_size_w = patch_shape[1]

        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // patch_size_h
        num_patches_w = width // patch_size_w
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size_h, patch_size_w, 1],
            strides=[1, patch_size_h, patch_size_w, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        num_patches_h = height // patch_size_h
        num_patches_w = width // patch_size_w
        patches = tf.reshape(
        patches,
            (
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size_h,
            patch_size_w,
            channels,
            ),
        )
        return patches

# Plotting the patches for the first image
def plot_patches(patches, patch_shape, id_x=12):
    num_patches_h = patches.shape[1]
    num_patches_w = patches.shape[2]
    
    for axis in range(3):
        # 8x8
        #fig, ax = plt.subplots(num_patches_h, num_patches_w, figsize=(8, 6))
        # 16x16
        fig, ax = plt.subplots(num_patches_h, num_patches_w, figsize=(8, 8))

        
        vmin, vmax = np.min(patches[id_x, :, :,: , :, axis]), np.max(patches[id_x, :, :,: , :, axis])
        print(vmin)
        print(vmax)
        
        # Loop through the patches and plot them
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = patches[id_x, i, j,: , :, axis]  # Selecting the first image in the batch
                ax[num_patches_h-1-i, j].imshow(patch, vmin=vmin, vmax=vmax, origin='lower')
                ax[num_patches_h-1-i, j].axis('off')
        
        plt.show()

# 8x8
#image_shape = (32,48)
#patch_shape = (8,8) 

# 16x16
image_shape = (64,64)
patch_shape = (16,16) 

num_patches = (image_shape[0] // patch_shape[0]) * (image_shape[1] // patch_shape[1])

# resize images and divide into patches, of sensor i in X_data[i]
resized_images = Resizing(image_shape[0], image_shape[1])(X_data[1])
patches = to_patches(patch_shape, resized_images)
plot_patches(patches, patch_shape)
