#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train all impersonal / personal models in a loop
for each load data, train model, evaluate.
"""

# Import libraries
import os
import sys
import numpy as np
import matplotlib.pylab as plt
import datetime 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.metrics import SparseTopKCategoricalAccuracy
from keras.models import load_model
# Import modules
import split_utils
import load_data as ld
from models import ViT

def main():
    script_dir = os.getcwd()+'/'
    
    plt.style.use('ggplot')
  
    ###########################################################################
    # Settings for loading data and labels 
    # XXX Define sensors to load data from
    seq_sensors = ["acc_p", "acc_w", "gyr_p", "gyr_w"]
    #seq_sensors = ["acc_w", "gyr_w"]
    #seq_sensors = ["acc_p", "gyr_p"]
    #seq_sensors = ["acc_w"]
    
    chosen_activities = 'all'
    dictionary = {}
    
    # XXX Division into windows
    windows = True
    # XXX Define window length defined in seconds
    seconds = 10
    # XXX '1' for sequences, '2' for spectrograms
    load_type = '2'
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
    
    
    X_data, labels, add_lst, feats = ld.get_inputs(ds, load_type, 
                     sensor_names=seq_sensors,
                     chosen_activities=chosen_activities, 
                     dictionary = dictionary,
                     sliding_window=windows,
                     window_seconds=seconds, 
                     interpolate=interpolate)
  
    ###########################################################################
    # Prepare data
    
    num_datapoints = len(labels)
    users = add_lst[0] #array of users with length same as num_datapoints
    activity_ids = add_lst[1]  #array of activity ids with length same as num_datapoints
    unique_users = np.unique(users)
    all_labels_this = sorted(np.unique(labels)) #sorted unique label list
    all_labels_this_wspace = [label.replace('_', ' ') for label in all_labels_this] #sorted unique label list without underscores
    
    num_classes = len(all_labels_this)  #number of possible categories
    # map each category to integer number 
    label_mapping = {label: i for i, label in enumerate(all_labels_this)} 
    int_labels = np.array([label_mapping[str_label] for str_label in labels])
    
    # onput dimensions for the different sensors for model construction
    # feat_shape is shape of higher feature array
    input_dimensions = [X_data[i][0].shape for i in range(len(X_data))]
    feat_shape = np.array(feats)[0][0].shape
    
    ###########################################################################
    # Train Impersonal or Personal Models
    # 10 iterations (for each of the 10 users), saves results in lists
    
    # XXX For training models set train True else models will only be evaluated 
    # given corresponding files exist
    train = False
    # XXX If personal is True Personal Models will be trained, else Impersonal 
    # Models are trained
    personal = False
    
    num_users = len(unique_users)
    # Assign weights to the users, which consider the number of samples of each user
    user_weights = [np.unique(users, return_counts=True)[1][u-1]/len(users) for u in unique_users] 
    # Empty lists with space for each users metric scores
    loss_train, accuracy_train, top3_train = [0]*num_users, [0]*num_users, [0]*num_users
    loss_val, accuracy_val, top3_val = [0]*num_users, [0]*num_users, [0]*num_users
    loss, accuracy, top3 = [0]*num_users, [0]*num_users, [0]*num_users
    
    # Directory for models to be saved
    models_directory = script_dir + 'trained_models/users/'
    if not os.path.exists(models_directory): os.makedirs(models_directory)

    for i in range(num_users):
        # Personal, only use data of user at i
        if personal:
            user_to_be_trained = unique_users[i]
            s, l, il, f, user_id = split_utils.prepare_one_user(X_data, labels, int_labels, users, \
                                                    feats, user_id=user_to_be_trained)
    
            # Unpack values into variable
            X_train, X_val, X_test = s
            y_train_str, y_val_str, y_test_str = l
            y_train_int, y_val_int, y_test_int = il #mapped label values
            
            # XXX File name for keras model
            name = models_directory + \
                'ds' + ds + '-ViT1-no-8b-user_'+str(user_to_be_trained)
    
        # Impersonal, exclude user from train/val
        else:
            user_to_be_tested = unique_users[i]
         
            s, l, il, u, f, test_user_ids = split_utils.prepare_impersonal(
                X_data, labels, int_labels, users, feats, leave_out_users=[user_to_be_tested], user_random_seed=13)
            
            # Unpack values into variables
            X_train, X_val, X_test = s
            y_train_str, y_val_str, y_test_str = l
            y_train_int, y_val_int, y_test_int = il #mapped label values
            train_users, val_users, test_users = u
            user_id = test_user_ids[0]
            
            # XXX File name for keras model
            name = models_directory + \
                'ds' + ds + '-ViT1-no-8b-withoutuser_'+str(user_to_be_tested)
    
    
        # Comment if features not needed
        # Unpack higher featurs, convert from DataFrame to Np Array
        '''f_arr = [[] for lst in range(len(f))]
        for j in range(len(f)):
            f_arr[j].append([np.array(sensor_feats) for sensor_feats in f[j]])
            f_arr[j] = f_arr[j][0]
        train_feats, val_feats, test_feats = f_arr
        '''
        
        train_input, val_input, test_input = X_train, X_val, X_test 
        
        print(f"Train Labels:\n{np.unique(y_train_str, return_counts=True)}")
        print(f"Val Labels:\n{np.unique(y_val_str, return_counts=True)}")
        print(f"Test Labels:\n{np.unique(y_test_str, return_counts=True)}")
    
        
        if train:
            # save starting time
            a = datetime.datetime.now()
        
            model = ViT.build_ViT1(seq_sensors, input_dimensions, num_classes)
    
            # Continue from last saved version / checkpoint
            # model = load_model(name+'.keras') 
        
            # Specify callbacks for logging results, early stopping, reducing 
            # learning rates and saving checkpoints
            csv_logger = CSVLogger(name+'_log.csv', append=True, separator=';')
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=9, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.2, patience=6, min_lr=0.00001)
            cp_filepath = name+'.keras'
            checkpoint_callback = ModelCheckpoint(
                filepath=cp_filepath,
                save_best_only=True,  # Save the best model
                monitor='val_accuracy',  # Choose the validation metric to monitor
                mode='max',  # Maximize the validation metric
                verbose=1)
        
            # Define parameters
            num_epochs = 80
            batch_size = 8
            #lr = 0.001
            lr = 0.0005
            optimizer = Adam(learning_rate=lr)
        
            # Compile and train
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', \
                          metrics=['accuracy', SparseTopKCategoricalAccuracy(k=3)])
        
            history = model.fit(train_input,
                                y_train_int,
                                validation_data=(val_input,
                                                 y_val_int),
                                epochs=num_epochs, batch_size=batch_size,
                                callbacks=[early_stopping, reduce_lr, checkpoint_callback, csv_logger])
        
            model.save(name+'.keras')
        
            # get elapsed time
            b = datetime.datetime.now()
            t2 = b-a
            print(f'{(t2.total_seconds()/3600):.2f} hours / {(t2.total_seconds()/60):.2f} minutes elapsed.')  # hours
        
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            #plt.savefig(name+'_accuracy.png')
            plt.show()
        
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            #plt.savefig(name+'_loss.png')
            plt.show()
        
        else:
            # evaluate
            model = load_model(name+'.keras')
            # model.summary()
            
        print()
        print("Results for user ", user_id)
        loss_train[i], accuracy_train[i], top3_train[i] = model.evaluate(train_input, y_train_int, verbose=0)
        print(f"Train Loss: {loss_train[i]:.4f}")
        print(f"Train Accuracy: {accuracy_train[i]:.4f}")
        print(f"Train Top 3 Accuracy: {top3_train[i]:.4f}\n")
    
        loss_val[i], accuracy_val[i], top3_val[i] = model.evaluate(val_input, y_val_int, verbose=0)
        print(f"Val Loss: {loss_val[i]:.4f}")
        print(f"Val Accuracy: {accuracy_val[i]:.4f}")
        print(f"Val Top 3 Accuracy: {top3_val[i]:.4f}\n")
    
        loss[i], accuracy[i], top3[i] = model.evaluate(test_input, y_test_int, verbose=0)
        print(f"Test Loss: {loss[i]:.4f}")
        print(f"Test Accuracy: {accuracy[i]:.4f}")
        print(f"Test Top 3 Accuracy: {top3[i]:.4f}\n")
    
    print()
    print(f"Mean Per-User Train Accuracy: {np.mean(accuracy_train)}")
    print(f"Mean Per-User Val Accuracy: {np.mean(accuracy_val)}")
    print(f"Mean Per-User Test Accuracy: {np.mean(accuracy):.4f}")
    print(f"Std Per-User Test Accuracy: {np.std(accuracy):.4f}")
    print(f"Mean Per-User Top 3 Test Accuracy: {np.mean(top3):.4f}")
    print(f"Std Per-User Top 3 Test Accuracy: {np.std(top3):.4f}")
    
    weighted_acc = np.dot(accuracy, user_weights)
    weighted_acc_std = np.sqrt(np.dot(user_weights, (accuracy - weighted_acc) ** 2))
    weighted_top3 = np.dot(top3, user_weights)
    weighted_top3_std = np.sqrt(np.dot(user_weights, (top3 - weighted_top3) ** 2))
    
    
    print()
    print(f"Weighted Mean Per-User Test Accuracy: {weighted_acc:.4f}")
    print(f"Weighted Std Per-User Test Accuracy: {weighted_acc_std:.4f}")
    print(f"Weighted Mean Per-User Test Top 3 Accuracy: {weighted_top3:.4f}")
    print(f"Weighted Std Per-User Test Top 3 Accuracy: {weighted_top3_std:.4f}")
    
    # Relevant only for dataset 1 / HA24    
    if ds == '1':
        print()
        # Result for only the samples having 2 recordings per category
        print(f"Acc. first 7 users (each having 2 recordings/category): {np.mean(accuracy[:7]):.4f}")
        # Result for only the samples having 1 recording per category
        print(f"Acc. last 3 users (each having 1 recording/category): {np.mean(accuracy[7:]):.4f}")
    
    print()
    print(accuracy)
if __name__ == "__main__":
    main()