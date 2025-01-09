#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file stores dictionaries/mappings to convert whole questions from study to keywords, 
to decode sensor service type,
to decode data source,
to store all string categories with possible answers,
and other miscellaneous tools.
"""
import numpy as np

# Dictionary for converting the raw questions into keywords
questions_abbrev = {'Ich fühle mich momentan erschöpft.': 'fatigue',
        'Ich fühle mich momentan gelangweilt.': 'boredom',
        'Ich fühle mich mometan gelangweilt.': 'boredom',
        'Ich fühle mich momentan motiviert.': 'motivation',
        'Ich fühle mich momentan konzentriert.': 'concentration',
        'Ich habe mir für diese Session Lernziele gesetzt.': 'goals_set',
        'Welcher Kategorie würdest Du Deine Lernkurse zuordnen?': 'learn_category',
        'Ich interessiere mich für meine Lernkurse.': 'interest',
        'Mit welchem digitalem Gerät lernst Du gerade?': 'digital_device',
        'An welchem Ort befindest Du Dich gerade?': 'place',
        'Mein Lernort ist momentan ordentlich.': 'tidied',
        'Die Beleuchtung in meinem Lernort empfinde ich momentan als angenehm.': 'light',
        'Die Temperatur in meinem Lernort empfinde ich momentan als angenehm.': 'temp',
        'Die Luftqualität in meinem Lernort empfinde ich momentan als angenehm.': 'air', 
        'In meiner direkten Umgebung nehme ich weitere Personen war.': 'presence_of_others',
        'Bearbeitest Du Deine momentane Lernaufgabe in einer Gruppe?': 'group_learning',\
        'Ich konnte in der letzten Stunde produktiv lernen.':'productivity',
        'In welcher Körperposition lernst Du momentan?': 'body_position',
        'Warum hast Du in den letzten 30 Min. Dein Lernen unterbrochen?': 'cause_interruption',
        'Welche Handbewegung hast Du in den letzten 3 Min. durchgeführt?': 'hand_activity',
        'Weswegen hast Du in den letzten 30 Min. vorwiegend nicht-lernrelevante Aktivitäten ausgeführt?': 'cause_non_relevant_learning',
        'Wie viele Male hast Du in den letzten 30 Min. Dein Lernen unterbrochen?': 'interruptions',
        \
        'Warum hast Du mit dem Lernen aufgehört?': 'reason_quitting',
        'Ich fand meine Lernaufgaben waren schwer zu bearbeiten.': 'difficulty',
        'Ich wurde durch multimediale Aktivitäten auf digitalen Geräten abgelenkt.': 'digital_distraction',
        'Ich wurde durch nicht-multimediale Aktivitäten abgelenkt.': 'nondigital_distraction',
        'Ich konnte mich nach Ablenkungen schnell wieder auf meine Lernaufgabe konzentrieren.': 'concentration_after_distraction',
        'Ich habe meine Lernziele für diese Session erreicht.': 'learning_goals_reached',
        'Ich wurde durch visuelle Einflüsse an meinem Lernort gestört.': 'visual_disturbance',
        'Ich wurde durch akustische Einflüsse an meinem Lernort gestört.': 'acoustic_disturbance',
        'Ich hätte besser in einer (anderen) Lerngruppe gelernt.': 'prefer_another_group',
        'Ich empfand meinen Lernort als bequem.': 'place_comfort',
        'Ich empfand den Geruch in meiner Lernumgebung als angenehm.': 'smell_comfort'
}
    
service_type_decoded = {0: 'Non Physical Env Data', 1: 'Physical Env Data', \
                    2: 'Physiological Data', 3: 'Behavioral Data', \
                    4: 'Virtual Data', 5: 'Device Processes Tracking'}    

datasource_decoded = {0: 'watch', 1: 'phone'}

# Dictionary of all non custom categories for every question, where custom answers were allowed
string_categories = {'learn_category': ['kreativ', 'logisch', 'komplex', 'andere Kategorie'],
                     'digital_device': ['Smartphone', 'Tablet', 'Laptop/Computer', 'keine Angabe'],
                     'place': ['zu Hause am Schreibtisch', 'zu Hause im Bett', 'zu Hause auf der Couch',
                               'in der Bibliothek', 'im Café', 'draußen', 'anderer Ort'],
                     'hand_activity': ['unsicher', 'stille Hände', 'Tippen am Smartphone', 'Tippen am Tablet',
                                       'Tippen am Computer', 'Surfen am Smartphone', 'Surfen am Tablet',
                                       'Schreiben mit Stift', 'Schreiben mit digitalem Stift', 'Zappeln'],
                     'body_position': ['sitzend', 'stehend', 'liegend', 	'andere Position'],
                     'cause_interruption': ['trifft nicht zu', 'Erschöpfung', 'Langweile', 'Motivationsverlust',
                                            'Konzentrationsverlust', 'digitale Störung', 'Störung durch anw. Personen',
                                            'Störung durch Lernort', 'unterbewusst', 'anderer Grund'],
                     'cause_non_relevant_learning': ['trifft nicht zu', 'Entspannung', 'Entertainment',
                                                     'digitale Kommunikation', 'Kommunikation mit anw. Personen',
                                                     'Gewohnheit', 'unterbewusst', 'anderer Grund'],
                     'reason_quitting': ['Lernen abgeschlossen', 'Erschöpfung', 'Langeweile', 'Motivationsverlust',
                                         'Konzentrationsverlust', 'digitale Störung', 'Störung durch anw. Personen',
                                         'Störung durch Lernort', 'unterbewusst', 'anderer Grund']}

# Map string categories to numbers so programs can handle it as input
string_cat_mappings = {}
for cat in string_categories.keys():
    string_cat_mappings[cat] = {cat: i for i, cat in enumerate([elem.rstrip().replace(" ", "_") for elem in string_categories[cat]]
)}

# Map likert scale to numbers
likert = {"trifft nicht zu": 1, "trifft eher nicht zu": 2, "weder noch": 3, \
      "trifft eher zu": 4, 	"trifft zu": 5}

boolean = {"Ja": True, "Nein": False}

def get_int_interruption(answer): 
    # for simplicity the answer "mehr als 4" is mapped to 4 
    if answer == "trifft nicht zu": answer = 0
    elif answer == "mehr als 4": answer =  4
    #else answers can be 1, 2 or 3
    return np.int16(answer)
