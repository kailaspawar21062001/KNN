# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 19:30:33 2023

@author: kailas
"""
################################################################################
1]PROBLEM   --"glass.csv"

    
BUSINESS OBJECTIVE:-Prepare a model for glass classification using KNN
                    Target Variable is --'Type'



#Import Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



# Datset
df=pd.read_csv("D:/data science assignment/Assignments/13.KNN/glass.csv")

#EDA
df.isnull().sum()
df.head()
df.tail()
df.shape
df.describe()


#Predictors
inpu=df.iloc[:,0:9]
#Target
target=df.iloc[:,[9]]

#Split the Dataset 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inpu,target,test_size=0.3)

#Importing KNN model
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=13)
model.fit(x_train,y_train)#Fit the model

#Evaluate on Test Data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix=confusion_matrix(y_test,testpred)

#Evaluate on Train Data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
confusion_matrix=confusion_matrix(y_train,trainpred)
test_report=classification_report(y_train,trainpred)
pip install numpy.ndarray

##################################################################################
2]PROBLEM   --"Zoo.csv"

BUSINESS OBJECTIVE:--Implement a KNN model to classify the animals in to categorie



#Dataset
df=pd.read_csv("D:/data science assignment/Assignments/13.KNN/Zoo.csv")

#EDA
df.isnull().sum()
df.head()
df.tail()
df.shape
df.describe()


#Predictors
inpu=df.iloc[:,1:17]
#Target
inpu.head()
target=df.type


#Split the Dataset 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inpu,target,test_size=0.3)

#Importing KNN model
from sklearn.neighbors import KNeighborsClassifier
#I did Hypertunning again and again,and found n_neighbors=5
# is optimum no of neighbors 
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)


#Evaluate on Test Data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix=confusion_matrix(y_test,testpred)

#Evaluate on Train Data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
confusion_matrix=confusion_matrix(y_train,trainpred)
test_report=classification_report(y_train,trainpred)
