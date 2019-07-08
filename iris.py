<<<<<<< Updated upstream

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from cross_validation_plotter import plot_cross_validation
#Iris
#Метод к ближацших соседей
data = load_iris()
X, Y = load_iris(return_X_y=True)
#X=data.data
#Y=data.target
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    #Установить классификатор knn с k соседями
    clf = KNeighborsClassifier(n_neighbors=k)
    #Fit the model
    clf.fit(X,Y)
    #Точность вычислений на тренировочном наборе
    train_accuracy[i] = clf.score(X, Y)
    #Compute accuracy on the test set
    test_accuracy[i] = clf.score(X, Y)
param_grid = {'n_neighbors':np.arange(1,50)}
clf = KNeighborsClassifier()
clf_cv= GridSearchCV(clf,param_grid,cv=5)
clf_cv.fit(X,Y)
n=clf_cv.best_params_
best=max(n.values())
print(n)

clf =KNeighborsClassifier(n_neighbors=best)
clf.fit(X, Y)
print(clf.predict(X[:7]))
print((X[:7]))
print("Доля ошибок классификации, при разных разбиениях X для cross-validation")
print(cross_val_score(clf, X, Y, cv=10))
print(cross_val_score(clf, X, Y, cv=10))
param= plot_cross_validation(X=X, y=Y, clf=clf, title="KNeighborsClassifier")
=======

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from cross_validation_plotter import plot_cross_validation
#Iris
#Метод к ближацших соседей
data = load_iris()
X, Y = load_iris(return_X_y=True)
#X=data.data
#Y=data.target
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    #Установить классификатор knn с k соседями
    clf = KNeighborsClassifier(n_neighbors=k)
    #Fit the model
    clf.fit(X,Y)
    #Точность вычислений на тренировочном наборе
    train_accuracy[i] = clf.score(X, Y)
    #Compute accuracy on the test set
    test_accuracy[i] = clf.score(X, Y)
param_grid = {'n_neighbors':np.arange(1,50)}
clf = KNeighborsClassifier()
clf_cv= GridSearchCV(clf,param_grid,cv=5)
clf_cv.fit(X,Y)
n=clf_cv.best_params_
best=max(n.values())
print(n)

clf =KNeighborsClassifier(n_neighbors=best)
clf.fit(X, Y)
print(clf.predict(X[:7]))
print((X[:7]))
print("Доля ошибок классификации, при разных разбиениях X для cross-validation")
print(cross_val_score(clf, X, Y, cv=10))
print(cross_val_score(clf, X, Y, cv=10))
param= plot_cross_validation(X=X, y=Y, clf=clf, title="KNeighborsClassifier")
>>>>>>> Stashed changes
print(param)