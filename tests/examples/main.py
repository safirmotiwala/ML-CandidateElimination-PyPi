#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 21:13:58 2020

@author: safir
"""

# Building the DataSet

fields = ['State', 'Climate', 'Age', 'Body_Shape', 'Immune_Strength', 'Mood', 'Migrant', 'Chocolate_Lover', 'Hangout', 'Stressed', 'Depressed', 'Literate', 'Food_preference', 'Friend_Circle', 'Married', 'Sports', 'Party', 'Covid_19']
state = ['Maharashtra', 'Karnataka']
Age = ['Young', 'Old']
Immune_Strength = ['Weak', 'High']
Mood = ["Happy", 'Sad']
Migrant = ['Yes', 'No']
Chocolate_Lover = ['Yes', 'No']
Hangout = ['Sometimes', 'Often']
Stressed = ['Sometimes', 'Often']
Depressed = ['Never', 'Sometimes', 'Often']
Literate = ['Yes', 'No']
Food_preference = ['Veg', 'NonVeg']
Friend_Circle = ['Small', 'Huge']
Married = ['Yes', 'No']
Sports = ['Cricket', 'Football']
Party = ['Sometimes', 'Often']
Covid_19 = ['Yes', 'No']

import csv
import random

filename = 'Covid-19_Data.csv'
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    row = [random.choice(state), 'Sunny', random.choice(Age), 'Under-weight', random.choice(Immune_Strength), random.choice(Mood), random.choice(Migrant), random.choice(Chocolate_Lover), random.choice(Hangout), random.choice(Stressed), random.choice(Depressed), random.choice(Literate), random.choice(Food_preference), random.choice(Friend_Circle), random.choice(Married), random.choice(Sports), random.choice(Party), 'Yes']
    csvwriter.writerow(row)
    for i in range(2):
        row = [random.choice(state), 'Sunny', random.choice(Age), 'Under-weight', random.choice(Immune_Strength), random.choice(Mood), random.choice(Migrant), random.choice(Chocolate_Lover), random.choice(Hangout), random.choice(Stressed), random.choice(Depressed), random.choice(Literate), random.choice(Food_preference), random.choice(Friend_Circle), random.choice(Married), random.choice(Sports), random.choice(Party), 'No']
    csvwriter.writerow(row)
    for i in range(0, 390):
        row = [random.choice(state), 'Sunny', random.choice(Age), 'Under-weight', random.choice(Immune_Strength), random.choice(Mood), random.choice(Migrant), random.choice(Chocolate_Lover), random.choice(Hangout), random.choice(Stressed), random.choice(Depressed), random.choice(Literate), random.choice(Food_preference), random.choice(Friend_Circle), random.choice(Married), random.choice(Sports), random.choice(Party), random.choice(Covid_19)]
        csvwriter.writerow(row)
    for i in range(0,10):
        row = [random.choice(state), 'Winter', random.choice(Age), 'Over-weight', random.choice(Immune_Strength), random.choice(Mood), random.choice(Migrant), random.choice(Chocolate_Lover), random.choice(Hangout), random.choice(Stressed), random.choice(Depressed), random.choice(Literate), random.choice(Food_preference), random.choice(Friend_Circle), random.choice(Married), random.choice(Sports), random.choice(Party), random.choice(Covid_19)]
        csvwriter.writerow(row)
        
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Covid-19_Data.csv')

result = {'Yes':1, 'No':0}
dataset['Covid_19'] = dataset['Covid_19'].map(result)

X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
#print(kf.get_n_splits(X))

for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

'''
from FindSAlgorithm import FindS
fs = FindS()            
S_hypothesis = fs.fit(X_train, y_train)
print("Specific Hypothesis : ", S_hypothesis)

y_pred = fs.predict(X_test)
y_pred1 = fs.predict(X_train)
print(y_pred)
'''

from CandidateEliminationAlgorithm import Candidate_Elimination
ce = Candidate_Elimination()
ce.fit(X_train, y_train)

y_pred = ce.predict(X_test)
y_pred1 = ce.predict(X_train)
print(y_pred)

from sklearn.metrics import accuracy_score, average_precision_score, f1_score

# Average Precision Score

# For Training Data
Avg1 = average_precision_score(y_train, y_pred1, average='micro', pos_label=1)
# For Test Data
Avg2 = average_precision_score(y_test, y_pred, average='micro', pos_label=1)

print("Average Precision Score (Training Data) : ", Avg1)
print("Average Precision Score (Test Data) : ", Avg2)

# F1 Score

# For training Data
FS1 = f1_score(y_train, y_pred1, average='binary', pos_label=1)
# For Test Data
FS2 = f1_score(y_test, y_pred, average='binary', pos_label=1)

print("F-1 Score (Training Data) : ", FS1)
print("F-1 Score (Test Data) : ", FS2)

# AUC
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
auc = metrics.auc(fpr, tpr)
print("Area under the Curve : ", auc)

# Plotting AUC (Area Under the Curve)
from sklearn.metrics import roc_auc_score, roc_curve
ns_probs = [0 for _ in range(len(y_test))]
lr_probs = y_pred

ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()