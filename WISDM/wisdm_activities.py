#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains dictionaries and lists of WISDM categories """

activities_dict = {'A': 'walking', 'B':'jogging', 'C': 'stairs',
              'D': 'sitting', 'E': 'standing', 'F': 'typing',
              'G': 'teeth', 'H': 'soup', 'I': 'chips', 
              'J': 'pasta', 'K': 'drinking', 'L': 'sandwich',
              'M': 'kicking', 'O': 'catch', 'P': 'dribbling',
              'Q': 'writing', 'R': 'clapping', 'S':'folding'}

activities = [key for key in activities_dict.keys()]

hand_activities = ['F', 'G', 'H', 'I', 'J', 
                   'K', 'L', 'Q', 'R', 'S']

learn_hand_activities = ['F', 'Q']

# map the hand activities related to learning to the label names of other datasets
mapping_to_cxt22 = {'F': 'Tippen_am_Computer', 'Q': 'Schreiben_mit_Stift'}

mapping_to_ha24 = {'F': 'typing_on_the_keyboard', 'Q': 'writing_with_a_pen',
                   'K': 'drinking', 'I': 'eating', 'J': 'eating', 'L': 'eating'}
