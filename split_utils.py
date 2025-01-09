#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains functions receiving data, splitting it accordingly into training, 
validation and test set and returning it.
"""

import numpy as np
import pandas as pd

# Auxiliary function which returns the data of indices in index_list.
def get_data_split_on_index(inputs, index_list):
    """ Receives list of np.arrays / pd.DataFrames or np.array / pd.DataFrame
    and splits into len(index_list) chunks based on indices. """
    
    outputs = [[] for i in range(len(inputs))]

    for i in range(len(inputs)):
        # s np array
        if isinstance(inputs[i], np.ndarray):
            for indices in index_list:
                outputs[i].append(inputs[i][indices])
        # is dataframe
        elif isinstance(inputs[i], pd.DataFrame):
            for indices in index_list:
                outputs[i].append((inputs[i]).iloc[indices])
        # is list of np arrays / dataframes
        else:
            if isinstance(inputs[i][0], np.ndarray):
                # Create list based on splits, not sensors
                outputs[i] = [[] for j in range(len(index_list))]
                
                for j in range(len(index_list)):
                    for k in range(len(inputs[i])):
                        outputs[i][j].append(inputs[i][k][index_list[j]])
            elif isinstance(inputs[i][0], pd.DataFrame):
                # Create list based on splits, not sensors
                outputs[i] = [[] for j in range(len(index_list))]
                
                for j in range(len(index_list)):
                    for k in range(len(inputs[i])):
                        outputs[i][j].append((inputs[i][k]).iloc[index_list[j]])
                
        
    return outputs
###############################################################################
# Functions for spltting the data:
    
def prepare_stratified(sequences, labels, int_labels, feats, random_seed=46, 
                       val_size=0.1, test_size=0.1):
    """ Shuffle data (evenly across classes) and split into training, 
    validation, test set. 
    
    Parameters:
    sequences -- list of np arrays. 
    labels -- np array.
    int_labels -- np array.
    feats -- list of pd dataframes.
    random_seed -- Seed for shuffling data in the dataset. Defaults to 46.
    val_size -- Proportion of the data for validation set (float <1). Defaults to 0.1.
    test_size -- Proportion of the data for test set (float <1). Defaults to 0.1.
    
    Returns:
    set_sequences, set_labels, set_int_labels, set_feats --
        The inputs each split into a list of 3.
    """
    
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

    # Define the number of splits
    n_splits = 1

    # Initialize StratifiedShuffleSplit and split data twice to gain 3 sets
    stratified_splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=(val_size+test_size), random_state=random_seed)
    for train_index, other_index in stratified_splitter.split(labels, labels):
         val_index, test_index = train_test_split(other_index, test_size=test_size/(
             val_size+test_size), random_state=random_seed, stratify=labels[other_index])
     
    # Call function 'get_data_on_index' to split the data
    indices = [train_index, val_index, test_index]
    set_sequences, set_labels, set_int_labels, set_feats = get_data_split_on_index([sequences, labels, int_labels, feats], indices)
    
    # reshape adds dimension (n) -> (n,1)
    for i in range(len(set_int_labels)):
        set_int_labels[i] = set_int_labels[i].reshape(-1, 1)
         
    return set_sequences, set_labels, set_int_labels, set_feats


def prepare_impersonal(sequences, labels, int_labels, users, feats, leave_out_users=[], leave_out_set_size = 2,
                       user_random_seed=42, shuffle_random_seed=46):
    """ Chooses users to be left out for test set, shuffles data and splits into 
    training, validation, test set. 
    
    Parameters:
    sequences -- list of np arrays. 
    labels -- np array.
    int_labels -- np array.
    users -- np array.
    feats -- list of pd dataframes.
    leave_out_users -- List containing values from users, will be left out of 
        test set (list). Defaults to empty.
    leave_out_set_size -- If list 'leave_out_users' is empty this parameter
        decides the number of users (randomnly chosen) to be left out (int). 
        Defaults to 2.
    user_random_seed -- Seed for choosing random users to be left out from 
        training (int). Defaults to 42.
    shuffle_random_seed -- Seed for shuffling data in the dataset. Defaults to 46.
    
    Returns:
    set_sequences, set_labels, set_int_labels, set_users, set_feats --
        The inputs each split into a list of 3.
    test_user_ids -- Np array of unique test users.""" 

    from sklearn.model_selection import train_test_split

    # Shuffle a set of users to be left out of the training set
    unique_users = np.unique(users)
    if len(leave_out_users) == 0:
        np.random.seed(user_random_seed)
        test_user_ids = np.random.choice(
            unique_users, size=leave_out_set_size, replace=False)
    else: 
        # Check whether user ids exist 
        if set(leave_out_users).issubset(unique_users):
            test_user_ids = np.array(leave_out_users)
    
   
    # Create binary masks for picking desired users
    train_val_ids = np.where(~np.isin(users, test_user_ids))[0]
    test_ids = np.where(np.isin(users, test_user_ids))[0]
    

    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.11, 
                                          random_state=shuffle_random_seed, 
                                          stratify=labels[train_val_ids])
       
    # Split based on mask values
    indices = [train_ids, val_ids, test_ids]
    set_sequences, set_labels, set_int_labels, set_users, set_feats = get_data_split_on_index([sequences, labels, int_labels, users, feats], indices)

    # Reshape adds dimension (n) -> (n,1)
    for i in range(len(set_int_labels)):
        set_int_labels[i] = set_int_labels[i].reshape(-1, 1)
    
    return set_sequences, set_labels, set_int_labels, set_users, set_feats, test_user_ids

def prepare_actseparated(sequences, labels, int_labels, act_ids, feats,
                     shuffle_random_seed=46):
    """ Splits the data evenly wrt labels and doesn't separate individual activities. 
    Shuffle data and split into training, validation, test set. 
    
    Parameters:
    sequences -- list of np arrays. 
    labels -- np array.
    int_labels -- np array.
    act_ids -- np array.
    feats -- list of pd dataframes.
    shuffle_random_seed -- Seed for shuffling data in the dataset. Defaults to 46.
    
    Returns:
    set_sequences, set_labels, set_int_labels, set_act_ids, set_feats --
        The inputs each split into a list of 3.  
    """
    
    from sklearn.model_selection import train_test_split

    train_indices = []
    val_indices = []
    test_indices = []
    
    # Iterate over unique labels
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # Get indices corresponding to current activity ID
        indices = np.where(labels == label)[0]
        unique_label_activity_ids = np.unique(act_ids[indices])
        
        # These cases for splitting would lead to errors as
        # not enough activities for each categories available
        if len(unique_label_activity_ids) < 2:
            train_ids = unique_label_activity_ids
            val_ids, test_ids = np.array([]), np.array([])
        elif len(unique_label_activity_ids) == 2:
            train_ids, val_ids = train_test_split(unique_label_activity_ids, test_size=0.2, random_state=shuffle_random_seed)
            test_ids = np.array([])
        elif len(unique_label_activity_ids) == 3: 
            train_ids, val_ids = train_test_split(unique_label_activity_ids, test_size=0.2, random_state=shuffle_random_seed)
            test_ids = np.array([])
        else:
            train_ids, rem_ids = train_test_split(unique_label_activity_ids, test_size=0.2, random_state=shuffle_random_seed)
            val_ids, test_ids = train_test_split(rem_ids, test_size=0.5, random_state=shuffle_random_seed)
            
            
        # Shuffle again for random results
        shuffle_random_seed += 2
        
        # Gather indices for each set based on activity IDs
        train_indices.extend([i for i, id in enumerate(act_ids) if id in train_ids])
        val_indices.extend([i for i, id in enumerate(act_ids) if id in val_ids])
        test_indices.extend([i for i, id in enumerate(act_ids) if id in test_ids])
    
    # Convert indices lists to numpy arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    # Shuffle indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    # Apply indices selection to arrays
    indices = [train_indices, val_indices, test_indices]
    set_sequences, set_labels, set_int_labels, set_act_ids, set_feats = get_data_split_on_index([sequences, labels, int_labels, act_ids, feats], indices)
    
    # Reshape adds dimension (n) -> (n,1)
    for i in range(len(set_int_labels)):
        set_int_labels[i] = set_int_labels[i].reshape(-1, 1)
        
    return set_sequences, set_labels, set_int_labels, set_act_ids, set_feats
    

def prepare_one_user(sequences, labels, int_labels, users, feats,
                     user_id='random', user_random_seed=42, shuffle_random_seed=46, 
                     val_size=0.1, test_size=0.1):
    """ Takes subset of data from only one person and splits it evenly across 
    classes. 
    
    Parameters:
    sequences -- list of np arrays. 
    labels -- np array.
    int_labels -- np array.
    users -- np array.
    feats -- list of pd dataframes.
    user_id -- A unique value of 'users'. Defaults to 'random'.
    user_random_seed -- Seed for choosing random user if user_id 'random'.
        Defaults to 42.
    shuffle_random_seed -- Seed for shuffling data in the dataset. Defaults to 46.
    val_size -- Proportion of the data for validation set (float <1). Defaults to 0.1.
    test_size -- Proportion of the data for test set (float <1). Defaults to 0.1.
    
    Returns:
    set_sequences, set_labels, set_int_labels, set_feats --
        The inputs each split into a list of 3.
    user_id: user id / name as in 'users' of the chosen user.
    """
       
    # Create binary mask for picking desired user
    user_mask = np.isin(users, user_id)
   
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Iterate over unique labels
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        label_mask = np.isin(labels, label)
        
        # Get indices where selected user and current label match
        label_indices = np.where(user_mask & label_mask)[0]
        num_samples = len(label_indices)

        num_test_samples = int(np.floor(num_samples * test_size))
        num_val_samples = int(np.floor(num_samples * val_size))
        num_train_samples = num_samples - num_val_samples - num_test_samples
        
        np.random.seed(shuffle_random_seed)  # Set seed for reproducibility
        np.random.shuffle(label_indices)
        
        train_index = label_indices[:num_train_samples]#.tolist()
        val_index = label_indices[num_train_samples:(num_train_samples + num_val_samples)]#.tolist()
        test_index = label_indices[(num_train_samples + num_val_samples):]#.tolist()
       
        train_indices.extend(train_index)
        val_indices.extend(val_index)
        test_indices.extend(test_index)

        shuffle_random_seed += 2

    indices = [train_indices, val_indices, test_indices]
    set_sequences, set_labels, set_int_labels, set_feats = get_data_split_on_index([sequences, labels, int_labels, feats], indices)
    
    # Reshape adds dimension (n) -> (n,1)
    for i in range(len(set_int_labels)):
        set_int_labels[i] = set_int_labels[i].reshape(-1, 1)

    return set_sequences, set_labels, set_int_labels, set_feats, user_id

