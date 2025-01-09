#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to use for printing and getting evaluation scores for classification or 
regression.
"""
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def calc_class_evaluation(y, y_pred, average="macro", all_classes=None):
    """ Calculate metrics and confusion matrix using sklearn. 
    Choices for average: 'macro', 'micro', 'average' """
    
    precision = precision_score(y, y_pred, average=average, zero_division=0)
    recall = recall_score(y, y_pred, average=average, zero_division=0)
    f1 = f1_score(y, y_pred, average=average, zero_division=0)
    conf_matrix = confusion_matrix(y, y_pred, labels=all_classes) 
    
    print()
    print("Recall:   ", format(recall, ".4f"))
    print("Precision:", format(precision,".4f"))
    print("F1-Score: ", format(f1, ".4f"))
    
    return precision, recall, f1, conf_matrix

def get_tp(conf_matrix):
    """"True Positives"""
    return conf_matrix.diagonal()

def get_tn(conf_matrix):
    """"True Negatives"""
    return conf_matrix.sum()-(conf_matrix.sum(axis=1)+conf_matrix.sum(axis=0)-conf_matrix.diagonal())

def get_fp(conf_matrix):
    """"False Positives"""
    return conf_matrix.sum(axis=0) - conf_matrix.diagonal()

def get_fn(conf_matrix):
    """False Negatives"""
    return conf_matrix.sum(axis=1) - conf_matrix.diagonal()

def get_recall(tp, fn):
    recall_per_label = np.divide(tp, (tp + fn), where=(tp + fn) != 0)
    return np.nan_to_num(recall_per_label)

def get_precision(tp, fp):
    precision_per_label = np.divide(tp, (tp + fp), where=(tp + fp) != 0)
    return np.nan_to_num(precision_per_label)

def get_f1(recall_per_label, precision_per_label):
    f1_per_label = [
        ((2 * r * p) / (r + p)) if (r + p) != 0 else 0 
        for (r, p) in zip(recall_per_label, precision_per_label)
        ]
    return np.array(f1_per_label)

def get_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fn + fp)

def get_acc_rec_prec_f1_per_label(cf):
    """ Given a confusion matrix cf, returns accuracy, recall, precision, f1
    scores for each label (np array with length of num classes)"""
    # Based on the confusion matrix, compute true positives, true negatives, 
    # false positives, false negatives
    tp = get_tp(cf)
    tn = get_tn(cf)
    fp = get_fp(cf)
    fn = get_fn(cf)
    
    # Compute category-wise metrics
    accuracy_per_label = get_accuracy(tp, tn, fp, fn)
    recall_per_label = get_recall(tp, fn)
    precision_per_label = get_precision(tp, fp)
    f1_per_label = get_f1(recall_per_label, precision_per_label)
    
    return accuracy_per_label, recall_per_label, precision_per_label, f1_per_label

def print_extended_evaluation(y_pred_probs, int_set_labels, set_labels, 
                              all_labels, label_mapping, single_predictions = False):    
    """
    Calculates and prints an evaluation.
    
    Parameters:
    y_pred_probs -- Array containing probability scores.
    int_set_labels -- Array with mapped labels (integers).
    set_labels -- Array containing the string labels.
    all_labels -- List containing unique labels of the loaded dataset / samples.
    label_mapping -- Dictionary containing integer values for each string label.
    single_predictions -- If True prints actual and predicted label of each data sample, 
        not recomm. for large dataset. Defaults to False.
    """
    # Convert probabilities to class predictions
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_pred_classes_str = np.array([next(key for key, value in label_mapping.items() if value == pred_label) for pred_label in y_pred_classes])
    if single_predictions:
        print("Labels:\n"+str(set_labels))
        print("Predictions:\n"+str(y_pred_classes_str))

    # All classes, including ones that are not predicted/available as sample in dataset, for confusion matrix completeness
    all_classes = list(label_mapping.values())

    # Macro precision, recall, f1 scores and confusion matrix cf
    precision, recall, f1, cf = calc_class_evaluation(int_set_labels, y_pred_classes, average='macro', all_classes=all_classes)
    
    # Get category-wise metrics
    accuracy_per_label, recall_per_label, precision_per_label, f1_per_label = \
        get_acc_rec_prec_f1_per_label(cf)

    all_labels_wspace = [label.replace('_', ' ') for label in all_labels] #sorted unique label list without underscores

    print('\n{:4s} {:25s} {:10s} {:13s} {:10s}'.format('Num', 'Label', 'Recall', 'Precision', 'F1 Score'))

    for j in range(len(all_labels)):
        print('{:4s} {:25s} {:.4f} {:3s} {:.4f} {:6s} {:.4f}'.format(str(j), 
                                                         all_labels_wspace[j],
                                                         recall_per_label[j],
                                                         '',
                                                         precision_per_label[j], 
                                                         '',
                                                         f1_per_label[j]))
              
    return precision, recall, f1, cf

def calc_regress_evaluation(y, y_pred, model_name=""):
    """ Computes Mean Squared Error and Mean Absolute Error and returns them."""
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"MSE{' '+model_name}: ", format(mse,".4f"))
    print(f"MAE{' '+model_name}: ", format(mae,".4f"))
    print()
    return mse, mae
