# Метрические методы классификации 
## Отчет по выполненным заданиям.
Выполнили студентки группы Б8320: Влучко А.А Романова М.С Шальтис В.А.

По результатам выполненных работ мы имеем 5 рабочих программ:

- digits.py (Метод К ближайших соседей для набора чисел)
- iris.py (Метод К ближайших соседей для ирисов)
- log_reg_diabets.py (Метод К ближайших соседей для набора данных по диабету)
- cancer.py (Метод К ближайших соседей для набора данных по раковым больным)
- titanic.py (Метод К ближайших соседей для набора данных по пассажирам Титаника)

# Описание кода для метода К ближайших соседей

Для объяснения принципа работы метода К ближайших соседей мы привели пример на программе iris.py. В других программах использующих данный метод меняется только блок выгрузки данных.

### Описание кода iris.py

```sh
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X=data.data
Y=data.target
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

print(cross_val_score(clf, X, Y, cv=10))

plt.figure()
plt.title("KNeighborsClassifier")
plt.xlabel("Training examples")
plt.ylabel("Score")
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=241)
train_sizes=np.linspace(.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
        clf, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.legend(loc="best")

plt.show()

```

Изначально, нужно подключить необходимые библиотеки и функции, которые будут задействованны в написании программы. load_iris() нам потребуется для импорта данных из библиотеки. Из бибилиотеки sklearn модуль KNeighborsClassifier, который является ключевым для рассчета данных. Далее, мы подключаем написанную функцию cross_validation_plotter, cross_val_score используется для оценки модели прогнозирования. 

# Время работы каждого из кодов
|     Программа    | Процент ошибок для тестовых данных | Процент ошибок для обучающей выборки |
|:----------------:|:----------------------------------:|:------------------------------------:|
|           iris   | 0.956 | 0.9741 |
|         digits   | 0.978 | 0.9827 |
|         diabetes | 0.735 | 0.78 |
|         rak      | 0.878 | 0.89493 |

