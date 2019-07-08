import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from cross_validation_plotter import plot_cross_validation
plt.style.use('ggplot')
#Load the dataset
df = pd.read_csv("diabetes.csv")

#Print the first 5 rows of the dataframe.
print("-----------------------------------")
print("Data view")
print("-----------------------------------")
print(df.head())
print("-----------------------------------")
print("Dimensions")
print("-----------------------------------")
print(df.shape)

#создадим пустые массивы для объектов и цели
X = df.drop('Outcome',axis=1).values
Y = df['Outcome'].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.4,random_state=42, stratify=Y)

#Setup для хранения точности обучения и тестирования
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Установить классификатор knn с k соседями
    knn = KNeighborsClassifier(n_neighbors=k)
    #Fit the model
    knn.fit(X_train, Y_train)
    #Точность вычислений на тренировочном наборе
    train_accuracy[i] = knn.score(X_train, Y_train)
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, Y_test)
    #Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
#Получить точность. Примечание: в случае алгоритмов классификации метод оценки представляет точность.
knn.score(X_test,Y_test)
#давайте получим прогнозы, используя классификатор, который мы использовали выше
Y_pred = knn.predict(X_test)
confusion_matrix(Y_test,Y_pred)
pd.crosstab(Y_test, Y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print(classification_report(Y_test,Y_pred))

Y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=7) ROC curve')
plt.show()
#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test,Y_pred_proba)
#В случае классификатора, такого как knn, параметр для настройки равен n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,Y)
knn_cv.best_score_
n=knn_cv.best_params_
best=max(n.values())
print(n)
knn_cv.best_params_

#.................

knn= KNeighborsClassifier(n_neighbors=best)
knn.fit(X, Y)

print("have diabetes or not?")
n = 1
print(knn.predict(X[14:23]))
print((X[14:23]))

print(cross_val_score(knn, X, Y, cv=10))

param=plot_cross_validation(X=X, y=Y, clf=knn, title="k-neighbors")
print("---------------------------------------")
print(param)
