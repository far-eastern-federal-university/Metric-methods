import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os

from IPython.display import Image
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets
from cross_validation_plotter import plot_cross_validation
from sklearn.model_selection import cross_val_score


from sklearn.neighbors import KNeighborsClassifier



#numders

# load the data
digits = datasets.load_digits()

X = digits.data
Y = digits.target
# Split into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42, stratify=Y)
# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)
# Fit the classifier to the training data
knn.fit(X_train, Y_train)
# Print the accuracy
print(knn.score(X_test, Y_test))
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 15)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, Y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, Y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, Y_test)

print("test_accuracy characteristics")
    
print(np.argmax(test_accuracy))
print(np.max(test_accuracy))

print(test_accuracy)
print(neighbors)




print("Блок вывода информации")
print("---------------------------------------")
print("Размерность набора данных")
print(digits.images.shape)
print("---------------------------------------")



print("---------------------------------------")
help(plot_cross_validation) 
print("---------------------------------------")
print("---------------------------------------")
param=plot_cross_validation(X=X, y=Y, clf=knn, title="KNeighborsClassifier")
print("---------------------------------------")
print(param)