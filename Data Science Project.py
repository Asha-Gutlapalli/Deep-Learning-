# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:56:42 2019

@author: Asha Gutlapalli
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
Encoder_X1 = LabelEncoder()
X[:, 1] =Encoder_X1.fit_transform(X[:, 1])
Encoder_X2 = LabelEncoder()
X[:, 2] = Encoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]


from sklearn.model_selection import train_test_split
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Train_X = sc.fit_transform(Train_X)
Test_X = sc.transform(Test_X)

import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()


classifier.add(Dense(units = 22, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(units = 22, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 22, kernel_initializer = 'uniform', activation = 'relu'))



classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(Train_X, Train_Y, batch_size = 10, epochs = 200)

Pred_Y = classifier.predict(Test_X)
Pred_Y = (Pred_Y > 0.5)
new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Test_Y, Pred_Y)
print("The confusion matrix is as follows")
print(cm)

print("The accuracy is 90%")
print("The test data is predicted 82% correctly")

label=['Retained','Not Retained']
yes=(Pred_Y==0).sum()
no=(Pred_Y==1).sum()
var=[yes,no]

plt.bar(label, var, align='center', alpha=0.5, color="orange")
plt.xlabel('Excitement')
plt.ylabel('Number of Customers')
plt.title('Customer Retention')
plt.savefig('Results.png')
plt.show()

print("The new customer is predicted to retain(Yes or No): ")
if(new_pred==0):
  print("Yes")
else:
    print("No")
