'''
HW2 problem
'''

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt

########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return cross entropy loss

    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:].squeeze()
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    s = x @ W.T + b

    # one_hot_encoding
    ans_train = np.array([[0 for _ in range(num_class)] for _ in range(n)])
    for i in range(n):
        ans_train[i][y_train[i]] = 1
    L = np.array(list(map(lambda x: np.log(np.exp(x) / sum(np.exp(x))), s)))
    L = np.array(list(map(lambda x, y: -1 * np.dot(x, y), ans_train, L)))
    avg_loss = L.sum() / n
    return avg_loss
    pass

########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return SVM loss

    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:].squeeze()
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    s = x @ W.T + b

    s_y = np.array(list(map(lambda s, y: s[y], s, y)))
    L = np.array(list(map(lambda s, s_y: s - s_y + 1, s, s_y)))
    L = np.array(list(map(lambda x: np.delete(x, np.where(x == 1)), L)))
    L = np.array(list(map(lambda x: np.maximum(x, 0), L)))
    L = np.array(list(map(lambda x: sum(x), L)))
    avg_loss = L.sum() / n

    return avg_loss
    pass

########################################
# Part 3. kNN classification
########################################
def knn_test(X_train, y_train, X_test, y_test, k):
    # implement your function here

    num_test = len(X_test)
    num_train = len(X_train)

    u_square = np.array([list(map(lambda x: x @ x.T, X_train))] * num_test)
    v_square = np.array([list(map(lambda x: x @ x.T, X_test))] * num_train).T
    uv = (2 * (X_test @ X_train.T))
    data = u_square + v_square - uv

    idx = np.array(list(map(lambda x: x.argsort()[0:k], data)))
    votes = np.array(list(map(lambda x: y_train[x], idx)))
    ans = np.array(list(map(lambda votes: stats.mode(votes)[0][0], votes)))
    accuracy = (ans == y_test).astype('uint8').sum() / n_test

    return accuracy

    pass


# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x, y, num_class):
    n_test = x.shape[0]
    feat_dim = x.shape[1]
    
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:].squeeze()
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    accuracy = 0

    # W has shape (num_class, feat_dim), b has shape (num_class,)

    # score
    s = x@W.T + b
    # score has shape (n_test, num_class)
    
    # get argmax over class dim
    res = np.argmax(s, axis = 1)

    # get accuracy
    accuracy = (res == y).astype('uint8').sum()/n_test
    
    return accuracy


# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'svm'

if classifiers == 'svm':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class)*100,'%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, k)*100
          , '% for k value of ', k)
