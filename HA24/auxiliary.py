#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary module with functions, dictionaries and variables for HA24 dataset, 
used in extract_data.py.
"""

seq_length = {'p': 15050, 'w': 15620} 
def freq(length, time_sec=150): return (length / time_sec)
    

english_activity_names = {"Stille Hände": "idle hands",
        "Tippen am Smartphone": "typing on a smartphone",
        "Tippen am Tablet": "typing on a tablet",
        "Tippen am Computer": "typing on the keyboard",
        "Scrollen am Smartphone": "scrolling on a smartphone",
        "Scrollen am Tablet": "scrolling on a tablet",
        "Bedienung der Computermaus": "using the computer mouse",
        "Bedienung des Touchpads": "using the touchpad",
        "Schreiben mit Stift": "writing with a pen",
        "Schreiben mit digitalem Stift": "writing with a digital pen",
        "Lesen in einem Buch": "reading a book",
        "Telefonieren": "making a phone call",
        "Essen": "eating",
        "Trinken": "drinking",
        "Kratzen": "scratching",
        "Zappeln": "fidgeting"
    }
