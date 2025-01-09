#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for inspecting data, loaded from the parquet file into a dataframe.
Only meta data tables, no sequential data.
Execute single lines.
"""

# =============================================================================
# this file uses the dataset from
# Weiss,Gary. (2019). WISDM Smartphone and Smartwatch Activity and Biometrics Dataset .
# UCI Machine Learning Repository. https://doi.org/10.24432/C5HK59.
#
# =============================================================================

# Import libraries
import os
import sys
import pandas as pd

# Import modules
script_dir = os.getcwd()+'/'
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
import load_data

pd.set_option('display.max_rows', 901)

# Load dataframe
df_m = load_data.load_df(ds='3', directory=script_dir)
#%% Inspect

df_m.shape
df_m.info(memory_usage='deep')
df_m.columns
df_m.dtypes
# df_dict['acc_w'].head()
df_m['seq_len_gyr_w'].describe()
df_m[['user', 'activity']].iloc[1]
df_m.iloc[1]

df_m.query('seq_len_acc_w>=8800')
df_m['activity'].value_counts()
# kind=kde, kind=hist bins=30
df_m['seq_len_acc_w'].plot(
    kind='hist', title='acc w lengths', bins=40)
#print(2802 in list(df_dict['acc_w'][3].query('seq_len>3960')['id']))


