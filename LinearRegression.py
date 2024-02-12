#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:43:43 2024

@author: rahulodedra
"""

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv('//Users//rahulodedra//Downloads//housing.csv')

X=df.values[:,:-1]
y=df.values[:,-1]

class Linearregression:
    
    def __init__(self,X,y,max_iteration,epsilon,learning_rate,gd) -> None:
        self.X=X
        self.y=y
        self.max_iteration=max_iteration
        self.epsilon=epsilon
        self.learning_rate=learning_rate
        self.gd=gd
        
    def data_split(self):
        self.trainX,self.testX,self.trainy,self.testy = train_test_split(self.X, self.y,test_size=0.3)
        
    def add_X0(self,X):
        return np.column_stack([np.ones(X.shape[0]),X])
    
    def train_scaling(self,X):
        mean=np.mean(X,axis=0)
        std=np.std(X,axis=0)
        X=(X-mean)/std
        X=self.add_X0(X)
        return X,mean,std
    
    def test_scaling(self,X,mean,std):
        X=(X-mean)/std
        X=self.add_X0(X)
        return X
    
    def rank(self,X):
        v,s,u=np.linalg.svd(X)
        rank=np.sum(s>0.00001)
        # rank=np.linalg.matrix_rank(X)
        return rank
    
    def full_rank(self,X):
        rank = self.rank(X)
        if rank == min(X.shape):
            self.full_rank=True
            print('Full rank')
        else:
            self.full_rank=False
            print(('Not full rank'))
    
    def low_rank(self,X):
        if X.shape[0]<X.shape[1]:
            self.low_rank = True
        else:
            self.low_rank=False
            print('Its not Low rank')
            
    def closed_form_solution(self,X,y):
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    def predict(self,X):
        return X.dot(self.w)
    
    def sse(self,X,y): #sum of squared error
        y_hat = self.predict(X)
        return ((y_hat-y)**2).sum()
        
    def cost_function(self,X,y):
        loss = self.sse(X,y)
        return loss/2
    
    def cost_derivative(self,X,Y):
        y_hat = self.predict(X)
        return X.T.dot(y_hat-y)
    
    def gradient_descent(self,X,y):
        errors = []
        prev_error = float('inf')
        for t in tqdm(range(self.max_iteration),colour='blue'):
            self.w -= self.learning_rate * self.cost_derivative(X, y)        
            loss = self.sse(X,y)
            errors.append(loss)
            if abs(loss - prev_error) < self.epsilon:
                print('Model stopped learning')
                break
            prev_error = loss
    
    def fit(self):
        self.data_split()
        self.trainX, mean, std = self.train_scaling(self.trainX)
        self.testX = self.test_scaling(self.testX,mean,std)
        
        self.full_rank(self.trainX)
        self.low_rank(self.trainX)
        
        if self.full_rank and not self.low_rank and self.trainX.shape[1]<1000 and not self.gd:
            self.closed_form_solution(self.trainX, self.trainy)
        else:
            self.w=np.ones(self.trainX.shape[1])
            self.gradient_descent(self.trainX,self.trainy)
        
        print(self.w)
        
    def plot_rmse(self, error_sequence):
        """
        @X: error_sequence, vector of rmse
        @does: Plots the error function
        @return: plot
        """
        # Data for plotting
        s = np.array(error_sequence)
        t = np.arange(s.size)

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel='iterations', ylabel=self.error,
               title='{} trend'.format(self.error))
        ax.grid()

        plt.legend(bbox_to_anchor=(1.05,1), loc=2, shadow=True)
        plt.show()
    
    
lr = Linearregression(X, y, max_iteration=5000, epsilon=0.003, 
                      learning_rate=0.00001, gd=True) 
lr.fit()    
    
    
    
    
    
    
    
    
        