import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from cross_validation_plotter import plot_cross_validation
from sklearn.model_selection import cross_val_score


#nb
data = pd.read_csv("wdbc.data", sep = ",")
data = data.iloc[:,0:12]
#print(data.describe())
X = data.iloc[:,2:12]
Y = data.iloc[:,1]
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


clf = KNeighborsClassifier(n_neighbors=best)
clf.fit(X, Y)

# Забираем рабочий код у ирисов


print("Where is malignant cancer?")
n = 1
print(clf.predict(X[14:23]))
print((X[14:23]))
print(cross_val_score(clf, X, Y, cv=10))

# Забираем функцию для построения графиков (не забываем подгрузить cross_validation_plotter)
print("---------------------------------------")
help(plot_cross_validation) # 
print("---------------------------------------")
print("Вызов функции")
param= plot_cross_validation(X=X, y=Y, clf=clf, title="KNeighborsClassifier")
print(param)


