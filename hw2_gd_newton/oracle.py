#!/usr/bin/env python

from scipy.special import expit as sigmoid
import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import load_svmlight_file

class Oracle:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.vol = len(y)
    
    # скаляр
    def value(self, w): 
        delta = 10 ** (-8)
        p = sigmoid(self.X @ w)
        return -(self.y.T @ np.log(p + delta) + (1 - self.y).T @ np.log(1 - p + delta)) / self.vol

    # вектор
    def grad(self, w):
        p = sigmoid(self.X @ w)
        return - self.X.T @ (self.y - p) / self.vol

    # матрица
    def hessian(self, w): 
        p = sigmoid(self.X @ w)
        return self.X.T @ np.diagflat(p * (1 - p)) @ self.X / self.vol

    def hessian_vec_product(self, w, d):
        u1 = self.X @ d
        p = sigmoid(self.X @ w)
        u2 = np.diagflat(p * (1 - p)) @ u1
        u3 = self.X.T @ u2
        return u3 / self.vol

    def fuse_value_grad(self, w):
        return self.value(w), self.grad(w)

    def fuse_value_grad_hessian(self, w):
        return self.value(w), self.grad(w), self.hessian(w)

    def fuse_value_grad_hessian_vec_product(self, w, d):
        return self.value(w), self.grad(w), self.hessian_vec_product(w, d)


def make_oracle(data_path, format="libsvm"):
    if format == "libsvm":
        X, y = load_svmlight_file(data_path)
        X = scipy.sparse.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
        y[y == -1] = 0
        y[y == 4] = 0
        y[y == 2] = 1
        y = y.reshape(-1, 1)
    elif format == "tsv":
        data = pd.read_csv("train.tsv", sep='\t').values
        y = data[:, 0].reshape(-1, 1) 
        X = data[:, 1:]
        X = np.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
    return Oracle(X, y)

def diff_grad(oracle, w):
    f = oracle.value
    eps = 10**(-8)
    n = oracle.X.shape[1]
    res = np.zeros(n)
    for i in range(n):
        e = np.zeros(n).reshape(-1, 1)
        e[i] = [1.]
        res[i] = (f(w + eps * e) - f(w)) / eps
    return res.reshape(-1, 1)

def diff_hessian(oracle, w):
    f = oracle.grad
    eps = 10**(-8)
    n = oracle.X.shape[1]
    res = [] 
    for i in range(n):
        e = np.zeros(n).reshape(-1, 1)
        e[i] = [1.]
        res.append(((f(w + eps * e) - f(w)) / eps).squeeze())
    return np.array(res)

def hess_grad_test(oracle, n=5):
    max_err_grad = 0
    max_err_hess = 0
    for i in range(n):
        w = np.random.rand(oracle.X.shape[1]).reshape(-1, 1)
        true_grad = oracle.grad(w)
        true_hessian = oracle.hessian(w)
        grad = diff_grad(oracle, w)
        hess = diff_hessian(oracle, w)
        max_err_grad = max(max_err_grad, np.max(grad - true_grad))
        max_err_hess = max(max_err_hess, np.max(hess - true_hessian))
    print("Максимальное значение ошибки приближения градиента: ", max_err_grad)
    print("Максимальное значение ошибки приближения гессиана: ", max_err_hess)