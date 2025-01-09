#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for inspecting data, loaded from the parquet files into dataframes.
Only meta data tables, no sequential data.
Execute single lines.
"""
# Import libraries
import os
import sys
import pandas as pd
import matplotlib.pylab as plt

# Import modules
script_dir = os.getcwd()+'/'
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
import load_data

plt.style.use('ggplot')
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 60)

# Load sensor and selfreport tables
df_1, df_2 = load_data.load_df(ds='2', directory=script_dir)

#%%
df_1.shape
df_1.info(memory_usage='deep')
df_1.columns
df_1.dtypes

df_2.shape
df_2.info(memory_usage='deep')
df_2.columns
df_2.dtypes
df_2 = df_2.dropna(axis=1, how='all')
#df.head()
print(\
df_2['seq_len_acc_p'].describe(),
df_2['seq_len_acc_w'].describe(),
df_2['seq_len_gyr_p'].describe(),
df_2['seq_len_gyr_w'].describe(),
df_2['seq_len_mag_p'].describe(),
df_2['seq_len_mag_w'].describe()
)
    
#%%
 
df_1.groupby(['hand_activity'])['hand_activity'].count()
df_1['hand_activity'].value_counts()
df_1.query('hand_activity=="unsicher"')
df_1.groupby(['user_id'])['hand_activity'].value_counts()

# Activities for which all sequential data is available
df_merged = pd.merge(df_2, df_1, on=['id', 'user_id', 'session_id', 'interval_index'])
seq_len_cols = ['seq_len_acc_p', 'seq_len_acc_w', 'seq_len_gyr_p', 'seq_len_gyr_w', 'seq_len_mag_p', 'seq_len_mag_w']
sub_df = df_merged[['id', 'user_id', 'session_id', 'interval_index','hand_activity']+seq_len_cols]
sub_df = sub_df.dropna()
sub_df.columns
sub_df['hand_activity'].value_counts()
# Hand activities available (for every sensor)  grouped after user
sub_df.groupby(['user_id'])['hand_activity'].value_counts()


# kind=kde, kind=hist bins=30
df_2['seq_len_acc_p'].plot(kind='hist', title='acc p lengths', bins=50)
df_2['seq_len_acc_p'].plot(kind='kde', title='acc p lengths')
df_2['seq_len_gyr_p'].plot(kind='kde', title='gyr p lengths')
df_2['seq_len_acc_w'].plot(kind='kde', title='acc w lengths')
df_2['seq_len_gyr_w'].plot(kind='kde', title='gyr w lengths')
df_2['seq_len_mag_p'].plot(kind='kde', title='mag p lengths')
df_2['seq_len_mag_w'].plot(kind='kde', title='mag w lengths')

df_2.query('seq_len_acc_p<=16000')
df_2.query('seq_len_gyr_p<=16000')
df_2.query('user_id==22')
df_2.groupby(['user_id'])['seq_len_acc_p'].describe()
df_2.groupby(['user_id'])['seq_len_gyr_p'].describe()
df_2.groupby(['user_id'])['seq_len_acc_w'].describe()
df_2.groupby(['user_id'])['seq_len_gyr_w'].describe()
df_2.groupby(['user_id'])['seq_len_mag_p'].describe()
df_2.groupby(['user_id'])['seq_len_mag_w'].describe()
