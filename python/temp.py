# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:36:54 2019

@author: Leopold
"""

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[0.69, 0.49], [-1.31, -1.21], [0.39, 0.99], [0.09, 0.29], [1.29, 1.09], [0.49, 0.79], [0.19, -0.31], [-0.81, -0.81], [-0.31, -0.31], [-0.71, -1.01]])
pca = PCA(n_components=1)
X1= pca.fit_transform(X)
print (X1)
print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)


plt.plot(*mean.T)
plt.plot(*shapes[0].T,label='0')
plt.plot(*shapes[1].T,label='1')
plt.plot(*shapes[2].T,label='2')
plt.plot(*shapes[3].T,label='3')
plt.legend()
plt.show()