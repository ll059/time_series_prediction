#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:18:28 2023

@author: james
"""

import pickle
import numpy as np
from pypots.optim import Adam
from pypots.classification import BRITS
from pypots.utils.metrics import cal_binary_classification_metrics
import optuna

from sklearn.metrics import confusion_matrix
import pandas as pd
from numpy import argmax
from sklearn.metrics import precision_recall_curve

def score_overall(dataset_for_testing, yhat):
    client_scores = {"client": [], "threshold": [], "tp": [], "fp": [], "fn": [], "tn": [], "ppv": [], "recall": [],
                     "npv": []}

    print("Scoring overall")

    predictions_for_score = yhat #df['p1']
    y_test_for_score = dataset_for_testing["y"] #df['result']

    if y_test_for_score.shape[0] > 1:

        t_holds = np.arange(.001, .999, 0.001)

        for t in t_holds:
            threshold = predictions_for_score > t

            tp = confusion_matrix(y_test_for_score, threshold)[1][1]
            fp = confusion_matrix(y_test_for_score, threshold)[0][1]
            fn = confusion_matrix(y_test_for_score, threshold)[1][0]
            tn = confusion_matrix(y_test_for_score, threshold)[0][0]

            client_scores['client'].append("All")
            client_scores["threshold"].append(t)
            client_scores["tp"].append(tp)
            client_scores["fp"].append(fp)
            client_scores["fn"].append(fn)
            client_scores["tn"].append(tn)

            ppv = tp / (tp + fp)
            recall = tp / (tp + fn)
            npv = tn / (tn + fn)

            client_scores["ppv"].append(ppv)
            client_scores["recall"].append(recall)
            client_scores["npv"].append(npv)

    overall_scores_df = pd.DataFrame(client_scores)
    ppv = overall_scores_df['ppv']
    recall = overall_scores_df['recall']
    overall_scores_df['f1'] = 2 * ((ppv * recall) / (ppv + recall))
    overall_scores_df = overall_scores_df.set_index('threshold')
    overall_scores_df = overall_scores_df[['client', 'tp', 'fp', 'fn', 'tn', 'npv', 'ppv', 'recall', 'f1']]
    return overall_scores_df


def objective(trial):

    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-3),
              'rnn_hidden_size': trial.suggest_int("rnn_hidden_size", X_train.shape[2], 2*X_train.shape[2])
              }
    
    # initialize the model
    brits = BRITS(
        n_steps=X_train.shape[1], 
        n_features=X_train.shape[2], 
        n_classes=2,
        rnn_hidden_size=params['rnn_hidden_size'],
        batch_size=128,
        # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        epochs=5,
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=None,
        # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
        # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
        optimizer=Adam(lr=params['learning_rate']),
        # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
        # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
        # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
        num_workers=0,
        # just leave it to default, PyPOTS will automatically assign the best device for you.
        # Set it to 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices.
        device=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], #'cpu',  
        # set the path for saving tensorboard and trained model files 
        saving_path=saving_path,
        # only save the best model after training finished.
        # You can also set it as "better" to save models performing better ever during training.
        model_saving_strategy="best",
    )
    
    # train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
    brits.fit(train_set=dataset_for_training, val_set=dataset_for_validating) 
    
    brits.save_model(saving_path, str(trial.number)+'.pypots', overwrite=True)
    
    return brits.best_loss

file_path = '/home/james/Projects/datasets/sequences/MI/pulled_MI_14-clients/processed_data/'
saving_path = "./Optuna_experiments/0.2Training_4GPUs"


with open(file_path + 'MI_advctz_XTrain_PyPots.pickle', 'rb') as handle:
    X_train = pickle.load(handle)

with open(file_path + 'MI_advctz_XTest_PyPots.pickle', 'rb') as handle:
    X_test = pickle.load(handle)

with open(file_path + 'MI_advctz_yTrain_PyPots.pickle', 'rb') as handle:
    y_train = pickle.load(handle)

with open(file_path + 'MI_advctz_yTest_PyPots.pickle', 'rb') as handle:
    y_test = pickle.load(handle)
    
print('Original X_train shape is: ', X_train.shape)
print('Original y_train shape is: ', y_train.shape)
print('Original X_test shape is: ', X_test.shape)
print('Original y_test shape is: ', y_test.shape)
 

#Select a certain percent (op_portion) of training data for Optuna hyperparameter optimization 
op_portion = 0.2
index = np.random.choice(X_train.shape[0], size=int(op_portion*X_train.shape[0]), replace=False)
X_train = X_train[index, :, :]
y_train = y_train[index]

#index = np.random.choice(X_test.shape[0], size=int(op_portion*X_test.shape[0]), replace=False)
#X_test = X_test[index, :, :]
#y_test = y_test[index]

print('Sampled X_train shape is: ', X_train.shape)
print('Sampled y_train shape is: ', y_train.shape)


# Assemble the datasets for training, validating, and testing.

dataset_for_training = {
    "X": X_train,  
    "y": y_train, 
}

dataset_for_validating = {
    "X": X_test, 
    "y": y_test, 
}

dataset_for_testing = {
    "X": X_test,  
    "y": y_test,  
}

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=1234))
study.optimize(objective, n_trials=100)


with open(file_path + 'MI_advctz_XTrain_PyPots.pickle', 'rb') as handle:
    X_train = pickle.load(handle)

with open(file_path + 'MI_advctz_yTrain_PyPots.pickle', 'rb') as handle:
    y_train = pickle.load(handle)

print('Original X_train shape is: ', X_train.shape)
print('Original y_train shape is: ', y_train.shape)
 
learning_rate = study.best_trial.params['learning_rate']
rnn_hidden_size = study.best_trial.params['rnn_hidden_size']

print('learning rate is: ', learning_rate)
print('rnn_hidden_size is: ', rnn_hidden_size)

# Assemble the datasets for training, validating, and testing.

dataset_for_training = {
    "X": X_train,  
    "y": y_train, 
}


# initialize the model
brits = BRITS(
    n_steps=X_train.shape[1], 
    n_features=X_train.shape[2], 
    n_classes=2,
    rnn_hidden_size=rnn_hidden_size,
    batch_size=128,
    # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
    epochs=100,
    # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
    # You can leave it to defualt as None to disable early stopping.
    patience=None,
    # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
    # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
    optimizer=Adam(lr=learning_rate),
    # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
    # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
    # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
    num_workers=0,
    # just leave it to default, PyPOTS will automatically assign the best device for you.
    # Set it to 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices.
    device= ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], #'cpu',  
    # set the path for saving tensorboard and trained model files 
    saving_path=saving_path,
    # only save the best model after training finished.
    # You can also set it as "better" to save models performing better ever during training.
    model_saving_strategy="best",
)

# Load the best model from Optuna optimization
best_optuna_model = saving_path + '/' + "{}.pypots".format(study.best_trial.number)
print('best optuna model is: ', best_optuna_model)
brits.load_model(best_optuna_model)
brits.fit(train_set=dataset_for_training, val_set=dataset_for_validating)

# the testing stage, impute the originally-missing values and artificially-missing values in the test set
brits_prediction = brits.classify(dataset_for_testing)

# calculate mean absolute error on the ground truth (artificially-missing values)

metrics = cal_binary_classification_metrics(brits_prediction, dataset_for_testing["y"])
print("Testing classification metrics: \n"
    f'ROC_AUC: {metrics["roc_auc"]}, \n'
    f'PR_AUC: {metrics["pr_auc"]},\n'
    f'F1: {metrics["f1"]},\n'
    f'Precision: {metrics["precision"]},\n'
    f'Recall: {metrics["recall"]},\n'
)


yhat = brits_prediction[:, 1]
# calculate roc curves
precision, recall, thresholds = precision_recall_curve(dataset_for_testing["y"], yhat)
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = argmax(fscore)
print('In precision_recall_curve, the best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))


#Scores over 1000 thresholds
overall_scores_df = score_overall(dataset_for_testing, yhat)
max_f1 = overall_scores_df['f1'].max()
print('After scoring over 1000 thresholds, max_f1 is:', max_f1)


f = open(saving_path + '/' + 'brits_Optuna_results.txt', 'w')
f.write('learning rate is: ' + str(learning_rate) + '\n')
f.write('rnn hidden size:' + str(rnn_hidden_size)  + '\n')
f.write('best optuna model is: ' + str(best_optuna_model)  + '\n') 
f.write('In PyPOTS, f1 is: ' + str(metrics['f1']) + '\n')
f.write('In precision_recall_curve, the best Threshold is: ' + str(thresholds[ix]) + '\n')
f.write('In precision_recall_curve, the f1-score is: ' + str(fscore[ix]) + '\n')
f.write('After scoring over 1000 thresholds, max_f1 is: ' + str(max_f1) + '\n')
f.close()