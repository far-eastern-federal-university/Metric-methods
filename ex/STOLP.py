# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:41:16 2019

@author: Dmitry
"""


import numpy as np
from scipy.spatial import distance_matrix
 
X = [[-1,   -1    ],
     [0,    0     ],
     [-1,   0     ],
     [0,    -1    ],
     [2,    2     ],
     [2,    3     ],
     [3,    3     ]]
X = np.array(X) # точки

y = [-1, -1, -1, -1, 1, 1, 1]
y = np.array(y) # классы, соответствующие точкам

unique = np.unique(y) # виды классов

dist = distance_matrix(X, X) # матрица всех расстояний

indices = [] # для каждого объекта будет хранить индексы ближайших соседей по порядку

for row in dist:
    sorted_dist = sorted(range(len(row)), key=lambda k: row[k])
    indices.append(sorted_dist)
    
indices = np.array(indices)

def Gamma_for_stolp(indices, y, i, k): # функция Г, но только для объектов, которые есть в выборке X
    s = 0
    for j in range(1, k+1):
        s += y[indices[i, j]] == y[i]
    return s
        
print(Gamma_for_stolp(indices, y, 4, 3))
        