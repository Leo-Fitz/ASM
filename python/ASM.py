# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:40:36 2019

@author: Leopold
"""
import numpy as np
import matplotlib.pyplot as plt

import json
import random

decoder = json.decoder

def rotationMatrix(fi):
    return np.array([[np.cos(fi), -np.sin(fi)],[np.sin(fi), np.cos(fi)]])

def procrustesDistance(X1, X2):
    return np.sqrt(np.sum(np.square(X1 - X2)))

def centerOfGravity(X): # 各点x,y坐标的平均值
    M = X.shape[0]
    return np.sum(X, axis=0, keepdims = True)/M

def sizeFrobenious(X, centroidFun=centerOfGravity): # 所有点到中心点的距离之和
    center = centroidFun(X)
    return np.sqrt(np.sum(np.square(X-center)))

def centroidSize(X, centroidFun=centerOfGravity):
    center = centroidFun(X)
    return np.sum(np.sqrt(np.sum(np.square(X - center), axis = 1)), axis = 0)

def tangentSpaceProjection(shapes, mean):
    meanSize = np.sum(mean*mean)
    projections = []

    for shape in shapes:
        alpha = meanSize/np.sum(mean*shape)
        projection = alpha * shape
        projections.append(projection)
    return projections


def toPCAform(shape):
    # reshape to (x1, x2, ..., y1, y2, ...) to avoid tensor SVD when doing PCA
    return shape.T.reshape(-1,1)

def toNormalform(shape):
    # reshape to (m x 2), for m landmarks
    return shape.reshape(2,-1).T




if __name__ == '__main__':
    # Choose size metric and centroid function
    sizeFun = sizeFrobenious
    centroidFun = centerOfGravity

    shapes = []
    tmpshapes = []

    with open('data/hand-landmarks.json', 'r') as saveFile:
        for line in saveFile.readlines():
            a = json.loads(line)
            shapes.append(np.array(a["coords"]))
            tmpshapes.append(None)

    # Shuffle data
    random.shuffle(shapes)

    # Take the first shape as mean, center it at origin
    mean = shapes[0]
    mean -= centroidFun(mean)
    mean /= sizeFun(mean)

    # Make mean hand shape look up
    mean = rotationMatrix(-np.pi/2).dot(mean.T).T

    # Center shapes at origin, resize them to unity
    for shape in shapes:
        # Center shapes around 0
        shape -= centroidFun(shape)

        # Resize to unity
        shape /= sizeFun(shape)


    # Calculate mean iteratively
    for iteration in range(10):
        for i, shape in enumerate(shapes):
            # Rotate shape to mean
            # SVD
            corelationMatrix = mean.T.dot(shape)
            U, S, VT = np.linalg.svd(corelationMatrix)

            # Rotate
            rotation = U.dot(VT)
            shapes[i] = rotation.dot(shape.T).T

        # Calculte mean, and normalize
        mean = sum(shapes)/len(shapes)
        mean /= sizeFun(mean)

    # Display mean shape
    plt.plot(*mean.T)
    plt.scatter(*centroidFun(mean).T)
    plt.show()

    # Ravel mean and shapes to vectors for PCA
    mean2 = toPCAform(mean)

    shapes2 = []
    for shape in shapes:
        shapes2.append(toPCAform(shape))


    # Shift shapes to tangent space:
    shapes2 = tangentSpaceProjection(shapes2, mean2)

    # prepare for covariance calculation
    N = mean2.shape[0]
    covarianceX = np.zeros(shape = (N, N))

    # Calculate shape covariance
    for shape2 in shapes2:
        diff = shape2 - mean2
        covarianceX += diff.dot(diff.T)
    covarianceX /= N    #有时候用N-1做分母

    # Calculate eigenbasis and eigenvalue
    U, covariance, Vt = np.linalg.svd(covarianceX) #U的列向量为特征向量, covariance为特征值

    # Standard deviations for each mode
    sigma = np.sqrt(covariance).reshape(-1,1)

    # Display a few shapes of +/- sigma for first mode
    print('first mode----------------------------------------------------------')
    b = sigma*0
    for i in range(9):
        b[0] = sigma[0]*0.25*i - sigma[0]
        x = mean2 + U.dot(b)
        plt.plot(*toNormalform(mean2).T)
        plt.plot(*toNormalform(x).T)
        plt.show()

    # Display a few shapes of +/- sigma for second mode
    print('second mode---------------------------------------------------------')
    b = sigma*0
    for i in range(9):
        b[1] = sigma[1]*0.25*i - sigma[1]
        x = mean2 + U.dot(b)
        plt.plot(*toNormalform(mean2).T)
        plt.plot(*toNormalform(x).T)
        plt.show()
    
