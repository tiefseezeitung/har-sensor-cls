#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load data, split into sets, train/load model, evaluate
"""

# Import libraries
import os
import sys
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import datetime
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.metrics import SparseTopKCategoricalAccuracy
from keras.models import load_model
# Import modules
import split_utils
import eval_utils
import load_data as ld
from models import sequential, CNN, ViT, Swin, CSWin

def main():
    script_dir = os.getcwd()+'/'
    models_dir = os.getcwd()+'/models'
    sys.path.append(models_dir)
    
    plt.style.use('ggplot')
    
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
    
    for i in range(len(X_data)):
        print('Shape of array ' + seq_sensors[i] + ':', X_data[i].shape)
    print()
    ###########################################################################
    # Prepare data
    num_samples = len(labels)
    users = add_lst[0] #array of users with length same as num_samples
    activity_ids = add_lst[1]  #array of activity ids with length same as num_samples
    unique_users = np.unique(users)
    all_labels_this = sorted(np.unique(labels)) #sorted unique label list
    all_labels_this_wspace = [label.replace('_', ' ') for label in all_labels_this] #sorted unique label list without underscores
    
    num_classes = len(all_labels_this)  #number of possible categories
    # Map each category to integer number 
    label_mapping = {label: i for i, label in enumerate(all_labels_this)} 
    int_labels = np.array([label_mapping[str_label] for str_label in labels])
    
    # Input dimensions for the different sensors for model construction
    # Feat_shape is shape of higher feature array
    input_dimensions = [X_data[i][0].shape for i in range(len(X_data))]
    feat_shape = np.array(feats)[0][0].shape
    
    ###########################################################################
    # XXX Define a split option (options listed underneath)
    split_option = '4'
    
    # Shuffle data and split into training/validation/test set
    
    # 1st Option
    # Shuffle data (evenly across classes) and split into training, validation, test set
    if split_option == '1':
        s, l, il, f = split_utils.prepare_stratified(X_data, labels, int_labels, feats) 
    
    # 2nd Option: Leave users out / Impersonal
    elif split_option == '2':
        s, l, il, u, f, test_user_ids = split_utils.prepare_impersonal(
            X_data, labels, int_labels, users, feats)
        
        # Unpack values into variables
        train_users, val_users, test_users = u
    
        print(f"Test Users: {np.unique(test_users)}")
        
    # 3rd Option: One single user / Personal
    # Take subset of data from only one person and train on that
    elif split_option == '3':
        s, l, il, f, user_id = split_utils.prepare_one_user(
            X_data, labels, int_labels, users, feats, user_id='random', user_random_seed=42)
        
    # 4th Option: Like 1st but individual activities in separate sets
    elif split_option == '4':
        s, l, il, ai, f = split_utils.prepare_actseparated(X_data, labels, int_labels, activity_ids, feats)
        
        # Unpack values into variables
        train_actids, val_actids, test_actids = ai
        train_actids_uniq, val_actids_uniq, test_actids_uniq = [np.unique(idx) for idx in ai]

    
    # Unpack values into variables
    X_train, X_val, X_test = s
    y_train_str, y_val_str, y_test_str = l
    y_train_int, y_val_int, y_test_int = il #mapped label values
    
    # Unpack higher featurs, convert from DataFrame to Np Array
    f_arr = [[] for lst in range(len(f))]
    for i in range(len(f)):
        f_arr[i].append([np.array(sensor_feats) for sensor_feats in f[i]])
        f_arr[i] = f_arr[i][0]
    train_feats, val_feats, test_feats = f_arr
     
    ###########################################################################
    # Print label counts of Train, Val, Test Set
    
    setnames = ['Training' ,'Validation' ,'Test']
    for i, labelset in enumerate([y_train_str, y_val_str, y_test_str]):
        labels_pr = np.unique(labelset, return_counts=True)[0]
        counts_pr = np.unique(labelset, return_counts=True)[1]
        labels_int_pr = np.array([label_mapping[str_label] for str_label in labels_pr])
        sorted_lab_counts = sorted(zip(labels_pr, counts_pr, labels_int_pr), key=lambda x: x[1], reverse=True)
        print(setnames[i]+" Labels")
        print("label\tcount\tdescription")
        for (str_label, c, int_label) in sorted_lab_counts:
            print(f" {int_label}\t\t {c} \t\t {str_label}")
    
    ###############################################################
    # Train
    
    # XXX Train model and evaluate OR load available model and evaluate
    train = False
    evaluate = False # Further metrics, confusion matrices, class-wise scores
    
    # XXX Choose data inputs for model
    train_input = X_train #+ train_feats
    val_input = X_val #+ val_feats
    test_input = X_test #+ test_feats
    #train_input = train_feats
    #val_input = val_feats
    #test_input = test_feats
    
    # XXX Model path and name
    models_directory = script_dir + 'trained_models/'
    if not os.path.exists(models_directory): os.makedirs(models_directory)
    #name = models_directory + 'ds' + ds + '-ViT1-actmix-8b-no-rs15'
    name = models_directory + 'ds' + ds + '-ViT1-actsep-8b-no'


    if train:
        # Use GPU if one is available (for installing library: pip install tensorflow-gpu)
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        
        # XXX Choose function for constructing keras model
        # Models as in Evaluation of Thesis
        #model = sequential.build_2layer_LSTM(seq_sensors, input_dimensions, num_classes=num_classes)
        #model = CNN.build_2DCNN(seq_sensors, input_dimensions, num_classes)
        #model = sequential.build_FF_Feats(seq_sensors, feat_shape, num_classes)
        model = ViT.build_ViT1(seq_sensors, input_dimensions, num_classes) # best
        
        # Others
        #model = sequential.build_1DCNN(seq_sensors, input_dimensions, num_classes, units=256)
        #model = CSWin.build_CSWin(seq_sensors, input_dimensions, num_classes, projection_dim=128, num_transformer_blocks=1, mlp_ratio=4, num_heads=[4])

        # Specify callbacks
        csv_logger = CSVLogger(name+'_log.csv', append=True, separator=';')
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=9, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=6, min_lr=0.00001)
        cp_filepath = name+'.keras'
        checkpoint_callback = ModelCheckpoint(
            filepath=cp_filepath,
            save_best_only=True,  # Save the best model
            monitor='val_accuracy',  # Choose the validation metric to monitor
            mode='max',  # Maximize the validation metric
            verbose=1)
        
        # Define parameters and optimizer
        num_epochs = 80
        batch_size = 8
        #lr = 0.001
        lr = 0.0005
        optimizer = Adam(learning_rate=lr)

        # Compile model
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', \
                      metrics=['accuracy', SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy")])
        
        # Save starting time
        a = datetime.datetime.now()

        # Start training process
        history = model.fit(train_input, \
                  y_train_int, \
                  validation_data=(val_input, \
                  y_val_int), \
                  epochs=num_epochs, batch_size=batch_size,\
                  callbacks=[early_stopping, reduce_lr, checkpoint_callback, csv_logger])
        
        # Calculate elapsed time and print it
        b = datetime.datetime.now()
        t2 = b - a
        print(f'\n{(t2.total_seconds()/3600):.2f} hours / {(t2.total_seconds()/60):.2f} minutes elapsed.\n') #hours
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        #plt.savefig(name+'_accuracy.png')
        plt.show()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        #plt.savefig(name+'_loss.png')
        plt.show()
        
    else:
        # Load model
        model = load_model(name+'.keras')
        
        model.summary()
        print()
   
    ###########################################################################
    # Evaluate best model
    
    # Set seaborn setting for plotting confusion matrices
    sns.set(rc={'figure.figsize':(20,14)})

    # Evaluate Training, Validation, Test Set
    loss, accuracy, top3 = [0]*3, [0]*3, [0]*3
    for i, (set_input, int_set_labels) in enumerate([
                                        (train_input, y_train_int), 
                                        (val_input, y_val_int), 
                                        (test_input, y_test_int)]):
        
        loss[i], accuracy[i], top3[i] = model.evaluate(set_input, int_set_labels, verbose=0)
        print(f"{setnames[i]} Loss: {loss[i]:.4f}")
        print(f"{setnames[i]} Accuracy: {accuracy[i]:.4f}")
        print(f"{setnames[i]} Top-3 Accuracy: {top3[i]:.4f}")
    
    # Evaluate further metrics, in for loop for each set
    if evaluate:
        precisions, recalls, f1s = [], [], []
        # Iterate over Train, Val and Test Set
        for i, (set_input, int_set_labels, set_labels) in enumerate([
                                            (train_input, y_train_int, y_train_str), 
                                            (val_input, y_val_int, y_val_str), 
                                            (test_input, y_test_int, y_test_str)]):
            print()
            print(setnames[i] + " Evaluation:")
            # Predict class probabilities using the model
            y_pred_probs = model.predict(set_input)
            
            # Print evaluation (also category-wise) and save
            # macro precision, recall, f1 scores and confusion matrix cf
            precision, recall, f1, cf = eval_utils.print_extended_evaluation(
                y_pred_probs,
                int_set_labels, set_labels,
                all_labels_this, label_mapping
                )
            
            precisions.append(precision); recalls.append(recall); f1s.append(f1)
            
            ###################################################################
            # Plot confusion matrix
            fig, ax = plt.subplots()
            sns.heatmap(cf, ax=ax, annot=True, cmap='coolwarm',fmt='g', annot_kws={"fontsize":22},
                        xticklabels=all_labels_this_wspace, yticklabels=all_labels_this_wspace, vmax=sum(cf[0]))
            plt.xticks(rotation=45, ha="right",fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlabel('Predicted', fontsize=30)
            plt.ylabel('Actual', fontsize=30)
            plt.title('Confusion Matrix - ' + setnames[i] + ' Set', fontsize=35)
            cax = ax.figure.axes[-1]
            cax.tick_params(labelsize=22)
            plt.tight_layout()
            #plt.savefig(name+'_cf_'+setnames[i]+'.png')
            plt.show()
            

if __name__ == "__main__":
    main()
