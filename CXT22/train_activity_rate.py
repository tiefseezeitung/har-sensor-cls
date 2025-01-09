#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction of (estimated) learning activity rate (on-/off-task classification).
This Python file is organized into blocks using #%%, which allows it to be 
executed block-by-block in IDEs like Spyder or VS Code.

"""

import os
script_directory = os.getcwd()+'/'

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import sys
# Import modules
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
import mappings
import load_data

# Load selfreport and sensor tables
df_1, df_2 = load_data.load_df('2', script_directory)

#%% Analysis

df_1.shape
df_1.info(memory_usage='deep')
df_1.columns
df_1.dtypes
#df.head()
df_1[['user_id', 'hand_activity']].iloc[22]
df_1.iloc[1]
df_1.query('group_learning==1')

df_2.columns
df_2.query('seq_len_gyr_w>=8800')
df_1['hand_activity'].value_counts()

#df_1['interruptions'].plot(kind='hist', title='number of interruptions per interval', bins=20)
# Histogram number of interruptions
fig, ax = plt.subplots(figsize=(5,5))
max_val = int(np.max(df_1['interruptions']))
bins = np.arange(-0.5, max_val + 1.5, 1)
ax.hist(df_1['interruptions'],  bins=bins, edgecolor='black')
ax.set_xticks(range(0, max_val + 1))
ax.set_ylabel('Frequency')
ax.set_xlabel('Interruptions')
plt.title('Number of interruptions per interval')
plt.plot()
       
def plot_hexbin(df, x, y):
    fig, ax = plt.subplots()
    hb = ax.hexbin(df[x], df_1[y], gridsize=12, cmap='Blues')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.show()

# hexbin
plot_hexbin(df_1, 'productivity', 'pre_motivation')
plot_hexbin(df_1, 'digital_distraction', 'interruptions')

#scatter plot
#df_1.plot(kind='scatter', x='digital_distraction', y='interruptions')

#%% Analysis2
x = 'pre_motivation'
y = 'productivity'
df_reg = df_1.dropna(subset=[x,y])
x_arr = np.array(df_reg[x])
y_arr = np.array(df_reg[y])

# Fit linear regression
from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()
reg_model.fit(x_arr.reshape(-1, 1), y_arr)
m = reg_model.coef_[0]
b = reg_model.intercept_

plt.figure(figsize=(8, 6))
#cmap: veridis, binary, hot
plt.hexbin(x_arr, y_arr, gridsize=20, cmap='viridis', label='Data Points')
plt.plot(x_arr, m*x_arr + b, color='red', label='Linear Regression')


plt.legend(loc='upper left')
plt.colorbar(label='Density')
plt.xlabel(x)
plt.ylabel(y)
plt.title('Heated Hexagonal Binning Plot')
plt.show()

#%% Assign dtype category to specific columns

# Convert some metadata to categorical dtype (might be relevant/informative variable!)
user_metadata = ['user_id', 'session_id', 'interval_index']
for d in user_metadata:
    df_1[d] = df_1[d].astype('category')

# Convert string values to numerical categories with saved mappings string_cat_mappings
for cat in mappings.string_categories.keys():
    df_1[cat] = (df_1[cat].map(mappings.string_cat_mappings[cat])).astype('category')

#%% Inspect correlations
################ correlations 1 (selfreport data)
columns_metadata_1 = ['id', 
                      'timestamp_from', 'timestamp_to', 'minutes_elapsed']

##df_1['group_learning'] = df_1['group_learning'].astype(int)
# Calculate the correlation matrix
correlation_matrix = df_1.drop(columns=columns_metadata_1+user_metadata).corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(40, 24))
sns.set(font_scale=2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('')
plt.rcParams['figure.dpi'] = 100
plt.title('Correlation Heatmap: data from selfreports', fontsize=32, weight='bold', pad=18)
plt.tight_layout()
plt.show()

################ correlations 2 (sensor data)
columns_metadata_2 = ['id',
                      'seq_len_acc_w', 'seq_len_gyr_w', 'seq_len_mag_w',
                      'seq_len_acc_p', 'seq_len_gyr_p', 'seq_len_mag_p']

# Calculate the correlation matrix
correlation_matrix = df_2.drop(columns=columns_metadata_2+user_metadata).corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(32, 22))
sns.set(font_scale=2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.rcParams['figure.dpi'] = 100
plt.title('Correlation Heatmap: sensory data', fontsize=32, weight='bold', pad=18)
plt.tight_layout()
plt.show()

#%% Fill NaN values for post moods with pre moods (assume no change in mood)
df_1filled = df_1
moods = ['fatigue', 'boredom', 'motivation', 'concentration']
for m in moods:
    df_1filled = df_1.fillna({'post'+m: 'pre_'+m})

#%% fill other NaN values with the mean of this user
#fill next values with mean of this user
result0 = df_1.groupby('user_id')['difficulty'].agg(['mean', 'std', 'min', 'max'])
result1 = df_1.groupby('user_id')['digital_distraction'].agg(['mean', 'std', 'min', 'max'])
result2 = df_1.groupby('user_id')['nondigital_distraction'].agg(['mean', 'std', 'min', 'max'])

# Columns from In-Session questionnaire, not appearing in Pre-Session questionnaire
cols1 = ['productivity', 'interruptions']
# Columns from Post-Session questionnaire
cols_postq = ['difficulty', 'digital_distraction', 'nondigital_distraction', 'concentration_after_distraction',
              'learning_goals_reached', 'visual_disturbance', 'acoustic_disturbance', 'prefer_another_group', 
              'place_comfort', 'smell_comfort']   
for col in (cols1 + cols_postq): 
    if df_1filled[col].isnull().values.any():  # Check if column contains NaN values
        df_1filled[col].fillna(round((df_1filled[col].mean())), inplace=True)


#%% Delete columns with no valuable information

# Calculate number of unique values in each column in df_1
unique_counts = df_1filled.nunique()
# Columns with only one value / no variety 
columns_with_one_value = unique_counts[unique_counts == 1].index.tolist()
df_1filled = df_1filled.drop(columns = columns_with_one_value)

# Calculate number of unique values in each column in df_2
unique_counts = df_2.nunique()
# Columns with only one value / no variety 
columns_with_one_value = unique_counts[unique_counts == 1].index.tolist()
df_2 = df_2.drop(columns = columns_with_one_value)
#%% Create column to be used as label
def create_activity_col(row):
    """ With defined columns as assumed markers create linear function and 
    scale value mapped to [0,1] for one row of a dataframe."""
    likert_scale = [1,2,3,4,5]
    # Columns used as anticipated markers for on/off task classification
    markers = {#in session quest
               'productivity': likert_scale, 'interruptions': [0,1,2,3,4],
               #post session quest
               'digital_distraction': likert_scale, 'nondigital_distraction': likert_scale, 
               'concentration_after_distraction': likert_scale}
    #positive_m = ['productivity', 'concentration_after_distraction']
    # negative markers
    negative_m = ['interruptions', 'digital_distraction', 'nondigital_distraction']
    maxsum = 0
    on_task_score = 0
    for m in markers.keys():
        # see whether value is available (not a NaN)
        if not np.isnan(row[m]):
            # Negative marker
            if m in negative_m:
                # Reverse the score, for example add 1 as the score when the value is 4 on likert
                # -> [1,2,3,4,5] -> [5,4,3,2,1]                
                on_task_score += markers[m][-(markers[m].index(row[m]))-1]
            else: # Positive marker
                on_task_score += row[m]
            
            # Expand denominator with max value
            maxsum += max(markers[m])

    # Divide by sum to get a score
    on_task_score /= maxsum
    
    return on_task_score

# new column -> label 
df_1filled['on_task_score'] = df_1filled.apply(create_activity_col, axis=1)
#df_1[['user_id', 'session_id', 'on_task_score']]
# another column with scores mapped to a possible class value, defined by threshold
threshold = 0.65
df_1filled['on_task_class'] = np.where(df_1filled['on_task_score'] >= threshold, 1, 0)

# Print statistics of score column
print(df_1filled['on_task_score'].describe())
print(f'median: {format(np.median(df_1filled["on_task_score"]), ".6f")}')

plt.figure(figsize=(18,6))
df_1filled['on_task_score'].plot(kind='kde', title='activity/on-task scores')
df_1filled['on_task_score'].plot(kind='hist', title='activity/on-task scores', bins=26)#hist #kde bins=30
plt.axvline(x = threshold, color = 'r')
plt.title('activity/on-task scores across dataset')
plt.xlim([0, 1])
plt.xticks(np.arange(0, 1.01, 0.1))
plt.xlabel('on-task score')
plt.ylabel('occurrence')
plt.show()

#%% Analyse missing values
# where are values missing?

#no post quest available
print(df_1[df_1['concentration_after_distraction'].isna()])
#no in quest available
print(df_1[df_1['productivity'].isna()])

#values missing in row 0?
print(df_1.iloc[0].isna())
#list all columns with at least 1 missing value
print(df_1.isna().any())
#show all columns where any values are missing
print(df_1.loc[:, df_1.isna().any()])

#%% Choose what data to use and merge it into one dataframe

# XXX Define the data to be used
merge_type = '3'
 
# Merge with ids in sensor data table, only keep matching rows
if merge_type == '0':
    df_merged = pd.merge(df_1filled, df_2, on=['id', 'user_id', 'session_id', 'interval_index'])
    df_merged.columns

    # Calculate the correlation matrix and plot
    
    correlation_matrix = df_merged.drop(columns=columns_metadata_1+columns_metadata_2).corr()
    # Print the correlation matrix
    #print(correlation_matrix)
    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(96, 48))
    sns.set(font_scale=3)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.rcParams['figure.dpi'] = 100
    plt.title('Correlation Heatmap: selfreport and sensory data', fontsize=50, weight='bold', pad=18)
    plt.tight_layout()
    plt.show()
    
# Merge but keep all rows (also the ones without sensor data)
elif merge_type == '1':
    df_merged = pd.merge(df_1filled, df_2, on=['id', 'user_id', 'session_id', 'interval_index'], how='left')
    df_merged.columns

# Only work with sensor data
elif merge_type == '2':
    # Extract basic metadata and labels columns
    sub_df1 = df_1filled[['id', 'user_id', 'session_id', 'interval_index','on_task_score', 'on_task_class']]
    # Append / merge to sensor dataframe
    df_merged = pd.merge(df_2, sub_df1, on=['id', 'user_id', 'session_id', 'interval_index'])
    df_merged.columns
    
    delete_cols = columns_metadata_2
    for column in df_2.columns:
        print((df_merged[[column]].isna()).value_counts())
        
    # Show all columns containing missing values
    print(df_merged.loc[:, df_merged.isna().any()])

# Do not merge at all, only work with selfreport data
elif merge_type == '3':
    df_merged = df_1filled

#%% Remove rows with NaN values
# Get information about in which columns NaN vals exist
is_nan_col = df_merged.isna().any()
# Remove rows based only on missing selfreport data
for col in df_1filled.columns:
    if is_nan_col[col]:
        df_merged = df_merged.dropna(subset=col)
    
#print(df_1.iloc[0].isna())
#print(df_1.isna().any())
#print(df_1.loc[:, df_1.isna().any()])
#%% Select all features for input X

further_removals = ['productivity', 'interruptions', 
                    'digital_distraction', 'nondigital_distraction', 
                    'concentration_after_distraction']
labels = np.array(df_merged['on_task_score']) 
labels_class = np.array(df_merged['on_task_class']) 

X = df_merged.drop(columns=columns_metadata_1 + columns_metadata_2 + 
                   further_removals + ['on_task_score'] + ['on_task_class'],
                    errors='ignore')

#%% Shuffle
num_datapoints = len(labels)

seed = 46
np.random.seed(seed)
shuffle_indices = np.random.permutation(num_datapoints)


X_shuffled = X.iloc[shuffle_indices]
labels_shuffled = labels[shuffle_indices]
labels_class_shuffled = labels_class[shuffle_indices]

#%% Split shuffled data into training and validation set

train_ratio = 0.8
train_samples = int(train_ratio * num_datapoints)

# Input data
X_train_df = X_shuffled[:train_samples]
X_val_df = X_shuffled[train_samples:]
    
# Labels
y_train = labels_shuffled[:train_samples]
y_val = labels_shuffled[train_samples:]

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

# Corresponding binary (class) labels
y_train_class = labels_class_shuffled[:train_samples]
y_val_class = labels_class_shuffled[train_samples:]

y_train_class = y_train_class.reshape(-1, 1)
y_val_class = y_val_class.reshape(-1, 1)

num_features = X.shape[1]
#%% keras dependencies
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam

#pip install tensorflow-gpu
#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%% Models
# Regression
def build_simple_nn(num_features):
    inputs = Input(shape=(num_features,), name='input')
    dense0 = Dense(64, activation='relu')(inputs)
    
    dense1 = Dense(32, activation='relu', )(dense0)
    dense2 = Dense(16, activation='relu', )(dense1)
    dense3 = Dense(8, activation='relu', )(dense2)
   
    output = Dense(1, activation='sigmoid')(dense3)
    
    model = Model(inputs=inputs, outputs=output)
    model.summary()
    return model

def build_linear():
    
    from sklearn import linear_model

    model = linear_model.LinearRegression()
    return model

def build_rforest_reg(estimators=5):
    
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=estimators, random_state=22, verbose=0)
    return model

#################################################################
# Classification
def build_log():
    
    from sklearn import linear_model

    model = linear_model.LogisticRegression()
    return model

def build_svm():
    
    from sklearn.svm import SVC
    
    model = SVC()
    return model

#%% Define categorical and numerical columns

categorical_columns = X.select_dtypes(include=['category']).columns
non_categorical_columns = X.columns.difference(categorical_columns)

X_train_df_c = X_train_df.copy()
X_val_df_c = X_val_df.copy()

for col in categorical_columns:
    #X_train[col] = X_train[col].cat.codes
    #X_val[col] = X_val[col].cat.codes
    X_train_df_c.loc[:, col] = X_train_df[col].cat.codes
    X_val_df_c.loc[:, col] = X_val_df[col].cat.codes

X_train = X_train_df_c.to_numpy()
X_val = X_val_df_c.to_numpy()
#%% Train

model = build_simple_nn(num_features)
models_directory = parent_dir + 'trained_models/'
name = models_directory + 'nn-64-32-16-8'
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
cp_filepath = name + '.keras'
checkpoint_callback = ModelCheckpoint(
    filepath=cp_filepath,
    save_best_only=True,  # Save the best model
    monitor='val_loss',  # Choose the validation metric to monitor
    mode='min',  
    verbose=1)

num_epochs = 100
batch_size = 1
lr = 0.0006
optimizer = Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

# Save starting time
import datetime
a = datetime.datetime.now()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), \
          epochs=num_epochs, batch_size=batch_size,\
          callbacks=[early_stopping, reduce_lr, checkpoint_callback])


# Get elapsed time
b = datetime.datetime.now()
t2 = b - a
print(f'\n{(t2.total_seconds()/3600):.2f} hours / {(t2.total_seconds()/60):.2f} minutes elapsed.\n') #hours

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#%% Evaluate
# Import module
import eval_utils as et

# Import sklearn functions
from sklearn.metrics import confusion_matrix

model = load_model(name+'.keras')
#model.summary()
loss_train, mse_train, mae_train = model.evaluate(X_train, y_train)
print(f"Train Loss/MSE: {loss_train:.4f}")
print(f"Train MAE: {mae_train:.4f}")

loss_val, mse_val, mae_val = model.evaluate(X_val, y_val)
print(f"Test Loss/MSE: {loss_val:.4f}")
print(f"Test MAE: {mae_val:.4f}")
print()

# Predict class probabilities using the model
y_pred_probs = model.predict(X_train)
print("Train Label \t Train Prediction")
for i in range(len(y_train)):
    print(str(round(y_train[i][0],7)) + '\t' + str(y_pred_probs[i][0]))
    
#################### classification
# Convert probabilities to class predictions
y_pred_classes = np.where(y_pred_probs >= threshold, 1, 0)
#y_train_class = np.where(y_train>=threshold, 1, 0) #redifinition
print("\nTrain Label \t Train Prediction")
for i in range(len(y_train)):
    print(str(y_train_class[i][0] )+ '\t' + str(y_pred_classes[i][0]))

_, _, _, _ = et.calc_class_evaluation(y_train_class, y_pred_classes)

conf_matrix_train = confusion_matrix(y_train_class, y_pred_classes) 
sns.heatmap(conf_matrix_train)
#%% Predict with other / simple models

# Linear Regression
model = build_linear()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

_, _ = et.calc_regress_evaluation(y_val, y_pred, model_name='Linear')

# Random Forest
model = build_rforest_reg(estimators=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

_, _ = et.calc_regress_evaluation(y_val, y_pred, model_name='Random Forest')

# Inspect which features have most influence 
importances = model.feature_importances_
indices_sorted = np.argsort(importances)[::-1]

# Print feature importances
print("Feature Importances:")
for i in indices_sorted:
    print(f"{format(importances[i],'.4f')} - {X_shuffled.columns[i]} (Index {i})")

""" Results for 10 estimators of selfreport data 
Feature Importances:
0.2528 - pre_motivation (Index 7)
0.2153 - visual_disturbance (Index 28)
0.2005 - cause_interruption (Index 23)
0.0599 - cause_non_relevant_learning (Index 24)
0.0429 - learning_goals_reached (Index 27)
0.0403 - place_comfort (Index 31)
"""

#%% Predict with classification models

# Logistic Regression
model = build_log()
model.fit(X_train, y_train_class)
y_pred = model.predict(X_val)

_, _, _, _ = et.calc_class_evaluation(y_val_class, y_pred)

# Support Vector Machine
model = build_svm()
model.fit(X_train, y_train_class)
y_pred = model.predict(X_val)

_, _, _, _ = et.calc_class_evaluation(y_val_class, y_pred)


