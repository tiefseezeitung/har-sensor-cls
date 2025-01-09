#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains data preprocessings: 
- Dividing sequences into (overlapping) windows
- Create spectrgrams
- Extract higher features

Also contains functions for interpolating or decimating sequences.
"""
import numpy as np
from skimage.util.shape import view_as_windows
import pandas as pd
import scipy
import matplotlib.pyplot as plt

sensor_name_short = {'acc_p': 'Acc. Phone', 'acc_w': 'Acc. Watch',
               'gyr_p': 'Gyr. Phone', 'gyr_w': 'Gyr. Watch',
               'mag_p': 'Mag. Phone', 'mag_w': 'Mag. Watch'}


def apply_sliding_window(sequences, labels, additional_list=[], window_size=100, step_size=20,
                         delete_till=3):
    """Takes in sequences, a list of sequential data, with sequences[i].shape
    being (3, number datapoints, sequence length). 
    Returns sequences with sequences[i] shaped (3, num datapoints * windows, window_size). 
    The labels are adjusted to/multiplied with the number of windows. 
    Regarding WISDM dataset:
    For an ideal sequence length of 3600 the window size of 100 corresponds to 
    a 5 second interval. The step size 20 corresponds to 1 second.
    By dismissing the first 3 x 100 windows with step size 20, the first 12
    seconds are dismissed.
    Number of windows will be ceil((array_length-window_size+1)/step_size).
    The additional_list is a list containing numpy arrays with shape like 'labels'
    storing additional meta data.
    
    Deleting first 15 seconds of an activity (likely transitional activity),
    is idea from Heydarian, Mohammadreza, and Thomas E. Doyle.
    'rWISDM: Repaired WISDM, a Public Dataset for Human Activity Recognition.'
    arXiv preprint arXiv:2305.10222 (2023).
    """
  
    window_shape = (1, window_size)
    # Iterate through all sensors
    for i in range(len(sequences)):
        num_datapoints = sequences[i].shape[1]
        result = []
        for d in range(3):
            sequence = sequences[i][d]
            # Convert to windows
            sequence_windows = view_as_windows(
                sequence, window_shape, step=(1, step_size))

            result.append(sequence_windows)

        result = np.array(result)
        if delete_till>0:  # delete first 3 windows
            result = result[:, :, delete_till:, :, :]
            
        # Number of windows per activity/datapoint
        num_windows = result.shape[2]
        
        sequences[i] = result.reshape(
            3, num_datapoints*num_windows, window_size)

    label_windows = np.repeat(labels, num_windows)
    for i in range(len(additional_list)):
        additional_list[i] = np.repeat(additional_list[i], num_windows)
    
    return sequences, label_windows, additional_list


def apply_sliding_window_by_seconds(sequences, labels, frequencies, additional_list=[], window_seconds=10, percentage_overlap=0.2,
                         delete_till_sec=2):
    """Takes in sequences, a list of sequential data, with sequences[i].shape
    being (3, number datapoints, sequence length). 
    Returns sequences with sequences[i] shaped (3, num datapoints * windows, window_size). 
    The labels are adjusted to/multiplied with the number of windows. 
    Regarding WISDM dataset:
    For an ideal sequence length of frequencies[i]*record_time the window size 
    is defined by frequencies[i]*window_seconds. 
    delete_till_sec defines how many seconds (from the start of one recording) 
    should be dismissed (might be a transitional period).
    Number of windows will be ceil((array_length-window_size+1)/step_size).
    The additional_list is a list containing numpy arrays with shape like 'labels'
    storing additional meta data.
    """
    
    # Iterate through all sensors
    for i in range(len(sequences)):
        # Calculate corresponding window size and step size given the frequency
        window_size = int(frequencies[i] * window_seconds)
        window_shape = (1, window_size)
        step_size = int((1-percentage_overlap) * window_size)
        
        num_datapoints = sequences[i].shape[1]
        result = []
        for d in range(3):
            # Get sequence for dimension d from the defined skipped seconds
            sequence = sequences[i][d][:,int(frequencies[i]*delete_till_sec):]

            # Convert to windows
            sequence_windows = view_as_windows(
                sequence, window_shape, step=(1, step_size))

            result.append(sequence_windows)

        result = np.array(result)
        
        # Number of windows per activity/sample
        num_windows = result.shape[2]
        
        sequences[i] = result.reshape(
            3, num_datapoints*num_windows, window_size)
        
    label_windows = np.repeat(labels, num_windows)
    for i in range(len(additional_list)):
        additional_list[i] = np.repeat(additional_list[i], num_windows)
    
    return sequences, label_windows, additional_list


def get_higher_features(sequences):
    """ Receives a list of 3 dimensional arrays with 
    shape (num samples, windows/sequence length, 3) """

    # Create higher level feature sets
    df_features = []
    for i in range(len(sequences)):
        # Compute mean, median, and standard deviation for every dimension
        mean_values = np.mean(sequences[i], axis=1)
        median_values = np.median(sequences[i], axis=1)
        std_values = np.std(sequences[i], axis=1)
        max_values = np.max(sequences[i], axis=1)
        min_values = np.min(sequences[i], axis=1)

        # Average absolute difference between each of the sequential values and 
        # the mean of the sequential values per axis
        absoldev_values = np.mean((np.abs(sequences[i] - mean_values[:, np.newaxis, :])), axis=1)
        
        # Compute Fast Fourier Transform for first 8 coefficients
        fft_values = np.abs(np.fft.fft(sequences[i], axis=1))[:, :, :4]  
    
        cos_xy, cos_xz, cos_yz = [], [], []
        cor_xy, cor_xz, cor_yz = [], [], []
        
        for j in range(len(sequences[i])):
            cos_xy.append(scipy.spatial.distance.cosine(sequences[i][j, :, 0], sequences[i][j, :, 1]))
            cos_xz.append(scipy.spatial.distance.cosine(sequences[i][j, :, 0], sequences[i][j, :, 2]))
            cos_yz.append(scipy.spatial.distance.cosine(sequences[i][j, :, 1], sequences[i][j, :, 2]))
            cor_xy.append(np.corrcoef(sequences[i][j, :, 0], sequences[i][j, :, 1])[0, 1])
            cor_xz.append(np.corrcoef(sequences[i][j, :, 0], sequences[i][j, :, 2])[0, 1])
            cor_yz.append(np.corrcoef(sequences[i][j, :, 1], sequences[i][j, :, 2])[0, 1])
            

        # Combine the computed features into a pandas DataFrame
        df_features.append(pd.DataFrame({
                         'Mean_X': mean_values[:, 0],
                         'Mean_Y': mean_values[:, 1],
                         'Mean_Z': mean_values[:, 2],
                         'Median_X': median_values[:, 0],
                         'Median_Y': median_values[:, 1],
                         'Median_Z': median_values[:, 2],
                         'Std_X': std_values[:, 0],
                         'Std_Y': std_values[:, 1],
                         'Std_Z': std_values[:, 2],
                         'Max_X': max_values[:, 0],
                         'Max_Y': max_values[:, 1],
                         'Max_Z': max_values[:, 2],
                         'Min_X': min_values[:, 0],
                         'Min_Y': min_values[:, 1],
                         'Min_Z': min_values[:, 2],
                         'Absoldev_X': absoldev_values[:, 0],
                         'Absoldev_Y': absoldev_values[:, 1],
                         'Absoldev_Z': absoldev_values[:, 2],
                         # Cosine distances between pairs of axes
                         'Cos_XY': np.array(cos_xy),
                         'Cos_XZ': np.array(cos_xz),
                         'Cos_YZ': np.array(cos_yz),
                         # Correlations between pairs of axes
                         'Corr_XY': np.array(cor_xy),
                         'Corr_XZ': np.array(cor_xz),
                         'Corr_YZ': np.array(cor_yz),
                          # 'FFT1', 'FFT2', ..., 'FFT4' for each dimension
                          **{f'FFT{i+1}_X': fft_values[:, j, 0] for j in range(4)},
                          **{f'FFT{i+1}_Y': fft_values[:, j, 1] for j in range(4)},
                          **{f'FFT{i+1}_Z': fft_values[:, j, 2] for j in range(4)},
                        }))
        
    # Combine all information from the dataframes in one dataframe
    #concatenated_df = pd.concat([df.add_suffix("_"+suffix) for df, suffix in zip(df_features, sensors)], axis=1)
    return df_features

def get_spect_data(sequences, labels, add_lst, sensor_names, frequencies, recording_time,
                   plot_example=False):
    """sequences[i] shaped (num_windows, sequence/window_length, axes)"""

    users = add_lst[0]
    activity_ids = add_lst[1]
    #######################################################################
    # 2: Compute short-time Fourier transforms for every axis in every window of each sensor

    from scipy.signal import stft # ShortTimeFT
    spectrograms = [[] for i in range(len(sensor_names))]
    
    nperseg = 32
    noverlap = 0 #default: None
    
    # Sensors
    for s in range(len(sensor_names)):
        #divide length by record time in seconds to get frequency / sampling rate
        sampling_rate = frequencies[s]
        
        # Windows
        for i in range(sequences[s].shape[0]):
            window_spectrograms = []
            # Axes
            for j in range(sequences[s].shape[2]):  # Iterate over axes
                f, t, Zxx = stft(sequences[s][i, :, j], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
                # note: f, t are overwritten with the same value each iteration
                magnitude = np.abs(Zxx)
                window_spectrograms.append(magnitude)

            spectrograms[s].append(window_spectrograms)
        
        spectrograms[s] = np.array(spectrograms[s])
        # Resulting Shape (num_windows, num_axes, num_frequency_bins, num_time_segments/frames)
        
        if plot_example:
            # Example Plot of spectrogram for a window and all axes
            window_idx = 12
            axis_name = {0: 'X', 1: 'Y', 2: 'Z'}
            
            fig, axs = plt.subplots(1, 3, figsize=(19, 10))
    
            label_str = str(labels[window_idx].replace("_", " "))
            user_str = str(users[window_idx])
            rec_id_str = str(activity_ids[window_idx])
            for i in range(len(axs)):
                axis_idx = i
                num_window_of_act = np.sum(activity_ids[:window_idx+1] == activity_ids[window_idx])        
                sum_windows_of_act = np.sum(activity_ids == activity_ids[window_idx])
                window_index_str = " - Window Number of Recording "+ str(num_window_of_act)+"/"+str(sum_windows_of_act)
                #print(type(axis_name[axis_idx]))
                im = axs[i].pcolormesh(t, f, spectrograms[s][window_idx, axis_idx])
                axs[i].set_title(axis_name[axis_idx]+'-axis', fontsize=20)
                axs[i].set_ylabel('frequency [Hz]', fontsize=20)
                axs[i].set_xlabel('time [s]', fontsize=20)
                axs[i].tick_params(labelsize=17)
                
                cbar = fig.colorbar(im, ax=axs[i], orientation='horizontal')
                cbar.set_label('Magnitude', fontsize=20) 
                cbar.ax.tick_params(labelsize=17)
    
                #ticklabs = cbar.ax.get_ylabel()
                #cbar.set_label_position('bottom')
                #cbar.ax.set_ylabel(ticklabs, fontsize=15)
                
            fig.suptitle('STFT Magnitude '+sensor_name_short[sensor_names[s]]+' - Activity "'+label_str+'" - User '+user_str+window_index_str,
                      fontsize=22, horizontalalignment='center')
            #plt.subplots_adjust(left=None, bottom=None, right=None, top=1, wspace=0.4, hspace=0.4)
            fig.tight_layout()
            #plt.savefig(sensor_names[s]+"_activity_"+str(labels[window_idx])+\
            #            "_user_"+user_str+'_i_'+str(window_idx)+'_rec_'+rec_id_str+'.png')
    
            plt.show()
        
            #plt.specgram(sequences[0][0,0,:], cmap=cmap, Fs=104)
        
        # Reshape to (num_frequency_bins, num_time_segments, num_axes=3)
        spectrograms[s] = spectrograms[s].transpose(0, 2, 3, 1)
         
    return spectrograms
   

def decimate_seq(seq, desired_len=3600):
    """Decimation: When having downsample_factor of n, delete every nth value.
    Downsampling is only applied if at least used_freq of 2x desired/expected frequence
    (multiple of the original frequency, achieved by having a downsample_factor 
    of 2 or more)"""
    
    from scipy.signal import resample
    
    # Factor by which sequence is too long, to be used for downsampling
    factor = len(seq) / desired_len

    # Factor by which to downsample
    downsample_factor = int(factor)

    # Resample the sequence
    downsampled_seq = resample(seq, len(seq) // downsample_factor) 

    return downsampled_seq

def interpolate_seq(seq, desired_len=3600, allow_upsampling=False):
    """Linear Interpolation: every single sequence is modified to achieve 
    (about) the desired length."""
    
    length = len(seq)
    #used_freq = length / desired_freq
    
    if allow_upsampling and length < desired_len:
        # Factor by which sequence is too short, to be used for upsampling
        factor = desired_len / length

        # Generate the indices for the upsampled sequence
        indices = np.arange(0, length, 1 / factor)
        
    else:
        # Factor by which sequence is too long, to be used for downsampling
        factor = length / desired_len
    
        # Generate the indices for the downsampled sequence
        indices = np.arange(0, length, factor)

    # Linear interpolation of every dimension
    x_seq = np.interp(indices, np.arange(length), seq[:,0])
    x_seq = x_seq.reshape(-1,1)
    y_seq = np.interp(indices, np.arange(length), seq[:,1])
    y_seq = y_seq.reshape(-1,1)
    z_seq = np.interp(indices, np.arange(length), seq[:,2])
    z_seq = z_seq.reshape(-1,1)
    # Concatenate to retain initiative shape
    interpolated_seq = np.concatenate((x_seq, y_seq, z_seq), axis=-1)

    return interpolated_seq
  
    
def interpolate_arr_lst(seq_lst, desired_len=3600, upsample=True):
    """Linear Interpolation: every single sequence is modified to achieve 
    (about) the desired length.
    Receives a list of one dimensional np arrays with shape (n,)"""
    
    for i in range(len(seq_lst)):
        
        # Array length
        length = len(seq_lst[i])
        # Upsample
        if upsample and length < desired_len:

            # Factor by which sequence is too short, to be used for upsampling
            factor = desired_len / length
    
            # Generate the indices for the upsampled sequence
            indices = np.arange(0, length, 1 / factor)
            
            # Linear interpolation
            seq_lst[i] = np.interp(indices, np.arange(length), seq_lst[i])
            
        # Downsample
        elif length > desired_len:
            # Factor by which sequence is too long, to be used for downsampling
            factor = length / desired_len
        
            # Generate the indices for the downsampled sequence
            indices = np.arange(0, length, factor)
        
            # Linear interpolation
            seq_lst[i] = np.interp(indices, np.arange(length), seq_lst[i])

        
    return seq_lst
