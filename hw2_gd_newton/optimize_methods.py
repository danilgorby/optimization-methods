#!/usr/bin/env python

import numpy as np
import scipy
from time import time

class optimize_gd:
    def __init__(self):
        self.values = []
        self.iterations = []
        self.oracle_calls = []
        self.times = []
        self.grads = []
    
    def __call__(self, oracle, start_point, line_search_method, tol=1e-8, max_iter=10000):
        x = start_point

        oracle_call = 0

        v, g = oracle.fuse_value_grad(x)
        oracle_call += 1
        d = -g
        d_start = d
        iter_num = 0
        time0 = time()

        self.values.append(v)
        self.iterations.append(iter_num)
        self.oracle_calls.append(oracle_call)
        self.times.append(0)
        self.grads.append(1)

        norm_d_start_sq = d_start.T @ d_start
        # print("norm(d_0)^2: ", norm_d_start_sq[0][0])

        while d.T @ d > tol * norm_d_start_sq:
            iter_num += 1
            if iter_num >= max_iter:
                print("break")
                break

            alpha, oraclecalls = line_search_method(oracle, x, d, stata=True)
            oracle_call += oraclecalls

            x = x + alpha * d
            v, g = oracle.fuse_value_grad(x)
            oracle_call += 1
            d = -g


            if iter_num % 1000 == 0:
                print('iteration: {}'.format(iter_num))
                print('value of CE: {}'.format(v[0][0]))
                print('norm(d)^2 / norm(d_0)^2:', (d.T @ d / norm_d_start_sq)[0][0])
                print('alpha', alpha.squeeze())
                print()

            self.values.append(v)
            self.iterations.append(iter_num)
            self.oracle_calls.append(oracle_call)
            self.times.append(time() - time0)
            self.grads.append((d.T @ d / norm_d_start_sq)[0][0])

        return x
    
    
    
def hessian_pro(H):
    flag = True
    tau = 10**(-8)
    while flag:
        try:
            L = np.linalg.cholesky(H)
            flag = False
        except:
            tau *= 2
            H += tau * np.eye(H.shape[0])
    return L

class optimize_newton:
    def __init__(self):
        self.values = []
        self.iterations = []
        self.oracle_calls = []
        self.times = []
        self.grads = []

    def __call__(self, oracle, start_point, line_search_method, tol=1e-8, max_iter=100):
        x = start_point

        oracle_call = 0

        v, g, H = oracle.fuse_value_grad_hessian(x)
        oracle_call += 1
        L = hessian_pro(H)
        d = scipy.linalg.cho_solve((L.T, L), -g)
        norm_d = d.T @ d
        if norm_d ** 0.5 >= 1000:
            d = d / (norm_d ** 0.5)
            
        g_start = g
        iter_num = 0
        time0 = time()

        self.values.append(v)
        self.iterations.append(iter_num)
        self.oracle_calls.append(oracle_call)
        self.times.append(0)
        self.grads.append(((g.T @ g) / (g_start.T @ g_start))[0][0])

        while (g.T @ g) / (g_start.T @ g_start) > tol:                              
            iter_num += 1
            if iter_num >= max_iter:
                print("break")
                break

            alpha, oraclecalls = line_search_method(oracle, x, d, stata=True)
            oracle_call += oraclecalls

            x = x + alpha * d
            v, g, H = oracle.fuse_value_grad_hessian(x)
            oracle_call += 1
            L = hessian_pro(H)
            d = scipy.linalg.cho_solve((L.T, L), -g)
             
            norm_d = d.T @ d
            if norm_d ** 0.5 >= 1000:
                d = d / (norm_d ** 0.5)
            
            print('iteration: {}'.format(iter_num))
            print('value of CE: {}'.format(v[0][0]))
            print('norm(grad)^2 / norm(grad_0)^2:', (g.T @ g)[0][0] / (g_start.T @ g_start)[0][0])
            print('alpha', alpha.squeeze())
            print() 

            self.values.append(v)
            self.iterations.append(iter_num)
            self.oracle_calls.append(oracle_call)
            self.times.append(time() - time0)
            self.grads.append(((g.T @ g) / (g_start.T @ g_start))[0][0])

        return x
    

def eta1(norm_r_sq):
    return min(0.5, norm_r_sq**0.25)

def eta2(norm_r_sq):
    return min(0.5, norm_r_sq**0.5)

def eta3(norm_r_sq):
    return 0.1

def eta4(norm_r_sq):
    return 0.5

def eta5(norm_r_sq):
    return 0.9

def CG(oracle, g, x, eta_fun, max_iter=1000):
    f_calls = 0
    # инициализируем неточное решение системы
    z = np.zeros(oracle.X.shape[1]).reshape(-1, 1)
    r = g
    d = -g
    norm_r_sq = r.T @ r

    # точность решения системы
    # eta = min(0.5, norm_r_sq**0.25) 
    eta = eta_fun(norm_r_sq)
    eps = eta * norm_r_sq**0.5
    
    for j in range(max_iter):
        Bd = oracle.hessian_vec_product(x, d)
        f_calls += 1
        
        dBd = d.T @ Bd
        if dBd <= 0:
            if j == 0:
                return d, f_calls
            else:
                return z, f_calls
           
        alpha = norm_r_sq / dBd
        z += alpha * d
        r_new = r + alpha * Bd
        norm_r_new_sq = r_new.T @ r_new
        
        if norm_r_new_sq**0.5 < eps:
            return z, f_calls
        
        beta = norm_r_new_sq / norm_r_sq
        d = -r_new + beta * d
        r = r_new
        norm_r_sq = norm_r_new_sq
        

class hfn_optimize:
    def __init__(self):
        self.values = []
        self.iterations = []
        self.oracle_calls = []
        self.times = []
        self.grads = []
        
    def __call__(self, oracle, start_point, line_search_method, eta_fun, tol=1e-8, max_iter=1000): 
        x = start_point
        
        oracle_call = 0
        time0 = time()


        for k in range(max_iter):
            v, g = oracle.fuse_value_grad(x)
            oracle_call += 1
            if k == 0:
                g_start = g

            if g.T @ g <= tol:
                break

            p, f_calls = CG(oracle, g, x, eta_fun, max_iter=1000)
            oracle_call += f_calls
            
            norm_p = p.T @ p
            if norm_p ** 0.5 >= 1000:
                p = p / (norm_p ** 0.5)
                
            alpha, funcalls = line_search_method(oracle, x, p, stata=True)
            oracle_call += funcalls           
                
            x = x + alpha * p

            print('iteration: {}'.format(k))
            print('value of CE: {}'.format(v[0][0]))
            print('norm_grad_sq:', (g.T @ g)[0][0])
            print('alpha', alpha.squeeze())
            print() 

            self.values.append(v)
            self.iterations.append(k)
            self.oracle_calls.append(oracle_call)
            self.times.append(time() - time0)
            self.grads.append(((g.T @ g) / (g_start.T @ g_start))[0][0])

        return x

