#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 11:53:45 2023

@author: james
"""

import pickle
import numpy as np
from pypots.classification import GRUD
from pypots.utils.metrics import cal_binary_classification_metrics
import optuna

file_path = '/home/james/Projects/datasets/sequences/MI/pulled_MI_14-clients/processed_data/'


with open(file_path + 'MI_advctz_XTrain_PyPots.pickle', 'rb') as handle:
    X_train = pickle.load(handle)

with open(file_path + 'MI_advctz_XTest_PyPots.pickle', 'rb') as handle:
    X_test = pickle.load(handle)

with open(file_path + 'MI_advctz_yTrain_PyPots.pickle', 'rb') as handle:
    y_train = pickle.load(handle)

with open(file_path + 'MI_advctz_yTest_PyPots.pickle', 'rb') as handle:
    y_test = pickle.load(handle)
    
print('X_train shape is: ', X_train.shape)
print('X_train type is: ', type(X_train))

print('y_train shape is: ', y_train.shape)
print('y_train type is: ', type(y_train))

print('X_test shape is: ', X_test.shape)
print('X_test type is: ', type(X_test))

print('y_test shape is: ', y_test.shape)
print('y_test type is: ', type(y_test))

# Assemble the datasets for training, validating, and testing.

dataset_for_training = {
    "X": X_train, #physionet2012_dataset['train_X'],
    "y": y_train, #physionet2012_dataset['train_y'],
}

dataset_for_validating = {
    "X": X_test, #physionet2012_dataset['val_X'],
    "y": y_test, #physionet2012_dataset['val_y'],
}

dataset_for_testing = {
    "X": X_test, #physionet2012_dataset['test_X'],
    "y": y_test, #physionet2012_dataset['test_y'],
}

def objective(trial):

    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'rnn_hidden_size': trial.suggest_int("rnn_hidden_size", 128, 512)
              }
    
    # initialize the model
    grud = GRUD(
        n_steps=60,   
        n_features=154, 
        n_classes=2, 
        rnn_hidden_size=params['rnn_hidden_size'], 
        epochs=1, # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        patience=None, # here we set patience=5 to early stop the training if the evaluting loss doesn't decrease for 5 epoches. You can leave it to defualt as None to disable early stopping.
        learning_rate=params['learning_rate'],
        # device='cpu', # just leave it to default, PyPOTS will automatically assign the best device for you. 
                    # Set it to 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices.
        saving_path="tutorial_results/classification/grud", # set the path for saving tensorboard files
    )
    
    # train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
    grud.fit(train_set=dataset_for_training, val_set=dataset_for_validating)
    
    # the testing stage, impute the originally-missing values and artificially-missing values in the test set
    grud_prediction = grud.classify(dataset_for_testing)
    
    # calculate mean absolute error on the ground truth (artificially-missing values)

    metrics = cal_binary_classification_metrics(grud_prediction, dataset_for_testing["y"])
    print("Testing classification metrics: \n"
        f'ROC_AUC: {metrics["roc_auc"]}, \n'
        f'PR_AUC: {metrics["pr_auc"]},\n'
        f'F1: {metrics["f1"]},\n'
        f'Precision: {metrics["precision"]},\n'
        f'Recall: {metrics["recall"]},\n'
    )
    
     

    return metrics["f1"]


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)

    
best_trial = study.best_trial
f = open('Optuna_GRU-D_results.txt', 'w')

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
    f.write(str(key))
    f.write(str(value))

f.close()
