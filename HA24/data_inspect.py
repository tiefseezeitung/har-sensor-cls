#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for inspecting data, loaded from the parquet file into a dataframe.
Only meta data tables, no sequential data.
Execute single lines or blocks.
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
pd.set_option('display.max_rows', 255)

# Load dataframe
df = load_data.load_df(ds='1', directory=script_dir)

#%% Inspect data

df.shape
df.info(memory_usage='deep')
df.columns
df.dtypes

print(\
df['seq_len_acc_p'].describe(),
df['seq_len_acc_w'].describe(),
df['seq_len_gyr_p'].describe(),
df['seq_len_gyr_w'].describe()
)
  
#%%
df.groupby(['hand_activity'])['hand_activity'].count()
df['hand_activity'].value_counts()
df.query('hand_activity=="Tippen am Computer"')
df.groupby(['user_id'])['hand_activity'].value_counts()
len(df.query('user_id==7'))
len(df.query('sec_elapsed!=150'))

drop_df = df.dropna()
drop_df.columns

# kind=kde, kind=hist bins=30
df['seq_len_acc_p'].plot(kind='hist', title='acc p lengths', bins=50)
df['seq_len_gyr_p'].plot(kind='hist', title='gyr p lengths', bins=50)
df['seq_len_acc_p'].plot(kind='kde', title='acc p lengths')
df['seq_len_gyr_p'].plot(kind='kde', title='gyr p lengths')
df['seq_len_acc_w'].plot(kind='kde', title='acc w lengths')
df['seq_len_gyr_w'].plot(kind='kde', title='gyr w lengths')

df.query('seq_len_acc_p<=14900')
df.query('seq_len_gyr_p<=14900')

df.groupby(['user_id'])['seq_len_acc_p'].describe()
df.groupby(['user_id'])['seq_len_gyr_p'].describe()
df.groupby(['user_id'])['seq_len_acc_w'].describe()
df.groupby(['user_id'])['seq_len_gyr_w'].describe()
   
