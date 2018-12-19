# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 23:21:54 2018

@author: rajat
"""
#importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
"""
import keras
from keras.models import Sequential
from keras.layers import Dense"""
#importing dataset
dataset = pd.read_csv('creditcard.csv')
dataset.head()

#counting and plotting the class 
count_classes = pd.value_counts(dataset['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

#Applying Feature Scaling to the Amount column of the given dataset

sc = StandardScaler()
dataset['NormAmount'] = sc.fit_transform(dataset['Amount'].reshape(-1, 1))
dataset = dataset.drop(['Time','Amount'],axis=1)
dataset.head()

#Splitting dataset into Dependent And Independent variables
X = dataset.loc[:, dataset.columns != 'Class']
y = dataset.loc[:, dataset.columns == 'Class']

#Calculating the number of fraud records and storing the indices of the false records
number_records_fraud = len(dataset[dataset.Class == 1])
fraud_indices = np.array(dataset[dataset.Class == 1].index)

#Calculating the number of true records and storing the indices of the true records
number_records_true = len(dataset[dataset.Class == 0])
normal_indices = np.array(dataset[dataset.Class == 0].index)

#Resampling the dataset
#1. we have choosen random indices from the correct dataset of length= no. of fraud records   
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

#We have concatenated the true and false undersampled records
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

#We have made a new dataset
under_sample_data = dataset.iloc[under_sample_indices,:]

#Splitting the dataset into dependent and independent variables
X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

#printing the percentages of the undersampled data
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

#Splitting the actual dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

#Splitting the undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,y_undersample,test_size = 0.3,random_state = 0)

#Trainng model using Logistic Regression

classifier_logistic_regression = LogisticRegression()
classifier_logistic_regression.fit(X_train_undersample,y_train_undersample)

y_pred_undersample_logistic = classifier_logistic_regression.predict(X_test_undersample)
cm_LR = confusion_matrix(y_test_undersample ,y_pred_undersample_logistic)

# Fitting SVM to the Training set
classifier_SVM = SVC(kernel = 'poly', random_state = 0)
classifier_SVM.fit(X_train_undersample,y_train_undersample)

# Predicting the Test set results
y_pred_undersample_SVM = classifier_SVM.predict(X_test_undersample)

# Making the Confusion Matrix
cm_SVM = confusion_matrix(y_test_undersample, y_pred_undersample_SVM)

#Fitting Naive Bayes to the training set
classifier_NB = GaussianNB()
classifier_NB.fit(X_train_undersample, y_train_undersample)

# Predicting the Test set results
y_pred_undersample_NB = classifier_NB.predict(X_test_undersample)

# Making the Confusion Matrix
cm_NB = confusion_matrix(y_test_undersample, y_pred_undersample_NB)

# Fitting Random Forest to our dataset
classifier_Random_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_Random_forest.fit(X_train_undersample, y_train_undersample)

# Predicting the Test set results
y_pred_undersample_Random_forest = classifier_Random_forest.predict(X_test_undersample)

# Making the Confusion Matrix
cm_Random_forest = confusion_matrix(y_test_undersample, y_pred_undersample_Random_forest)


#Predicting Results on Actual Dataset
ytest1=classifier_NB.predict(X_test)
cm_test_NB=confusion_matrix(y_test,ytest1)

ytest2=classifier_Random_forest.predict(X_test)
cm_test_RF=confusion_matrix(y_test,ytest2)

ytest3=classifier_SVM.predict(X_test)
cm_test_SVM=confusion_matrix(y_test,ytest3)

ytest4=classifier_logistic_regression.predict(X_test)
cm_test_LR=confusion_matrix(y_test,ytest4)
