# Iris-flower-clarification-ML-project

import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

dataset=load_iris()

print(dataset.DESCR)
X = dataset.data
y = dataset.target 
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r.',label='setosa')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'g.',label='versicolour')
plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], 'b.',label='virginica')
plt.legend()
plt.show()
plt.plot(X[:, 0][y == 0]* X[:, 1][y == 0], X[:, 1][y == 0]* X[:, 2][y == 0],'r.',label='setosa')
plt.plot(X[:, 0][y == 1]* X[:, 1][y == 1], X[:, 1][y == 1]* X[:, 2][y == 1],'g.',label='versicolour')
plt.plot(X[:, 0][y == 2]* X[:, 1][y == 2], X[:, 1][y == 2]* X[:, 2][y == 2],'b.',label='virginica')
plt.legend()
plt.show()
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train , y_test=train_test_split(X, y)
from sklearn.linear_model import LogisticRegression 
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
                   
                   log_reg.score(X_test,y_test)
                   
                   log_reg.score(X, y)
