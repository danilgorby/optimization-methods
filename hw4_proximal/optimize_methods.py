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