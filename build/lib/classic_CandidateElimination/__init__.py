#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:22:52 2020

@author: safir
"""

import numpy as np

class Candidate_Elimination:
    def __init__(self):
        self.Xtrain = ""
        self.ytrain = ""
        self.Xtest = ""
        self.ytest = ""
        self.specific_hypothesis = []
        self.general_hypothesis = []
        self.version_space = []
        
    def fit(self, X, y):
        count, count1 = 0, 0
        self.Xtrain, self.ytrain = X, y
        for i, val in enumerate(y):
            if val==1:
                S_hypothesis = list(X[0].copy())
                G_hypothesis = [['?' for _ in range(len(S_hypothesis))]]
                print("Initial Specific Hypothesis : ", S_hypothesis)
                print("Initial General Hypothesis : ", G_hypothesis)
                break
        for i, val in enumerate(X):
            if y[i]==1:
                count+=1
                for x in range(len(S_hypothesis)):
                    #print(G_hypothesis)
                    if val[x] == S_hypothesis[x]:
                        pass
                    else:
                        S_hypothesis[x] = '?'
                        temp = ['?' for _ in range(len(S_hypothesis))]
                        temp[x] = val[x]
                        #print(temp)
                        if temp not in G_hypothesis:
                            pass
                        else:
                            G_hypothesis.remove(temp)
                            #print("Removed")
            elif y[i]==0:
                count1+=1
                temp = []
                for x in range(len(S_hypothesis)):
                    if val[x] != S_hypothesis[x] and S_hypothesis[x] != '?':
                        temp = ['?' for _ in range(len(S_hypothesis))]
                        temp[x] = S_hypothesis[x]
                        if temp not in G_hypothesis:
                            G_hypothesis.append(temp)
                        temp = []
                        
        if len(G_hypothesis)==1:
            v1 = [S_hypothesis]
        else:
            v1 = []
            for i in G_hypothesis:
                for j in range(len(i)):
                    if i[j] != '?':
                        h = i[j]
                        for z in range(len(S_hypothesis)):
                            if S_hypothesis[z] != '?':
                                temp = ['?' for _ in range(len(S_hypothesis))]
                                temp[z] = S_hypothesis[z]
                                temp[j] = i[j]
                                v1.append(temp)
                                
        print('Final Specific Hypothesis : ', S_hypothesis)
        print('Final General Hypothesis : ', G_hypothesis)
        print('Final Version Space : ', v1)
        print(count, " ", count1)
        self.specific_hypothesis, self.general_hypothesis = S_hypothesis, G_hypothesis
        self.version_space = v1
        return 1
    
    def predict(self, X):
        y = np.array([0 for i in range(len(X))])
        self.Xtest = X
        for i, val in enumerate(X):
            val = list(val)
            check = 0
            for x in range(len(val)):
                for z in self.version_space:
                    if val[x] == z[x]:
                        check+=1
                    else:
                        pass
            if check>0:
                y[i] = 1
            else:
                y[i] = 0
        self.ytest = y
        return y