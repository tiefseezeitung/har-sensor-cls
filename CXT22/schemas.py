#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema collection for parquet tables, used in extract_data.py.
"""
import pyarrow as pa

# Schema for 3 dimensional time-series data
schema_xyz = pa.schema([
    ('x', pa.float64()),
    ('y', pa.float64()),
    ('z', pa.float64())
    ])

# Schema containing identifying meta data, sensors for watch/phone,
# and the length of the time-series arrays
schema_sensor = pa.schema([
    ('id', pa.int16()),
    ('user_id', pa.int16()),
    ('session_id', pa.int16()),
    ('interval_index',pa.int16()),
    \
    ('wifi_count_w', pa.int16()),
    ('light_avg_lux_w', pa.float64()),
    ('noise_avg_max_amp_w', pa.float64()),
    ('temp_median_degree_w', pa.float64()),
    ('heartrate_avg_mbeats/min_w', pa.float64()),
    ('blood_oxygen_sat_avg_percentage_w', pa.float64()),
    ('SDNN_median_ms_w', pa.float64()),
    ('step_count_na1_w', pa.float64()),
    ('step_count_na2_w', pa.float64()),
    ('step_count_tilt_w', pa.float64()),
    ('step_count_step_w', pa.float64()),
    ('seq_len_acc_w', pa.int32()),
    ('seq_len_gyr_w', pa.int32()),
    ('seq_len_mag_w', pa.int32()),
    \
    ('wifi_count_p', pa.int16()),
    ('light_avg_lux_p', pa.float64()),
    ('noise_avg_max_amp_p', pa.float64()),
    ('temp_median_degree_p', pa.float64()),
    ('heartrate_avg_mbeats/min_p', pa.float64()),
    ('blood_oxygen_sat_avg_percentage_p', pa.float64()),
    ('?_median_(ms)_p', pa.float64()),
    ('step_count_na1_p', pa.float64()),
    ('step_count_na2_p', pa.float64()),
    ('step_count_tilt_p', pa.float64()),
    ('step_count_step_p', pa.float64()),
    ('seq_len_acc_p', pa.int32()),
    ('seq_len_gyr_p', pa.int32()),
    ('seq_len_mag_p', pa.int32()),
    ])

# Schema is subset of schema_sensor, only contains the real sensor names
# without suffixes for watch (w) / phone (p)
schema_sensor_names = pa.schema([
    ('wifi_count', pa.int16()),
    ('light_avg_lux', pa.float64()),
    ('noise_avg_max_amp', pa.float64()),
    ('temp_median_degree', pa.float64()),
    ('heartrate_avg_mbeats/min', pa.float64()),
    ('blood_oxygen_sat_avg_percentage', pa.float64()),
    ('SDNN_median_ms', pa.float64()),
    ('step_count_na1', pa.float64()),
    ('step_count_na2', pa.float64()),
    ('step_count_tilt', pa.float64()),
    ('step_count_step', pa.float64())
    ])

schema_usage1 = pa.schema([
    ('app', pa.string()),
    ('total_time_fg', pa.int32()) #foreground
    ])

schema_usage2 = pa.schema([
    ('app', pa.string()),
    ('total_time_fg', pa.int32()), #foreground
    ('launch_count', pa.int32())
    ])

# Schema notifications
schema_notifs = pa.schema([
    ('user_id', pa.int16()),
    ('session_id', pa.int16()),
    ('num_notifications', pa.int32()),
    ('num_muted_notifs', pa.int32())
    ])

# Schema containing metadata, and data pre and post attitudes
schema_base = pa.schema([
    ('id', pa.int16()),
    ('user_id', pa.int16()),
    ('session_id', pa.int16()),
    ('interval_index',pa.int16()),
    ('timestamp_from', pa.string()),
    ('timestamp_to', pa.string()),
    ('minutes_elapsed', pa.float64()),
    # fatigue, boredom, motivation and concentration (likert scale)
    # is interrogated in Pre and In Session Questionnaires
    ('pre_fatigue', pa.int16()),
    ('post_fatigue', pa.int16()), 
    ('pre_boredom', pa.int16()),
    ('post_boredom', pa.int16()),
    ('pre_motivation', pa.int16()),
    ('post_motivation', pa.int16()),
    ('pre_concentration', pa.int16()),
    ('post_concentration', pa.int16())
    ])

# Pre Session Questions:
schema_pre = pa.schema([
    ('goals_set', pa.int16()),
    ('learn_category', pa.string()),
    ('interest', pa.int16()),
    ('digital_device', pa.string()),
    ('place', pa.string()),
    ('tidied', pa.int16()),
    ('light', pa.int16()),
    ('temp', pa.int16()),
    ('air', pa.int16()),
    ('presence_of_others', pa.int16()),
    ('group_learning', pa.bool_())
    ])

# In Session Questions
schema_in = pa.schema([
    ('hand_activity', pa.string()),
    ('body_position', pa.string()),
    ('productivity', pa.int16()),
    ('interruptions', pa.int16()), #tricky one 
    ('cause_interruption', pa.string()),
    ('cause_non_relevant_learning', pa.string())
    ])

# Post Session Questions:
schema_post = pa.schema([
    ('reason_quitting', pa.string()), 
    ('difficulty', pa.int16()),
    ('digital_distraction', pa.int16()),
    ('nondigital_distraction', pa.int16()),
    ('concentration_after_distraction', pa.int16()),
    ('learning_goals_reached', pa.int16()),
    ('visual_disturbance', pa.int16()),
    ('acoustic_disturbance', pa.int16()),
    ('prefer_another_group', pa.int16()),
    ('place_comfort', pa.int16()),
    ('smell_comfort', pa.int16())
    ])

# Final combination
schema_selfrep = pa.unify_schemas([schema_base, 
                                   schema_pre, 
                                   schema_in, 
                                   schema_post])