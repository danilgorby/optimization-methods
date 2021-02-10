#!/usr/bin/env python3
"""Parabola method for 1d optimization"""
import numpy as np


def optimize(a, b, f, eps=1e-8, stat=False):
    #I = b - a    
    f_a, _ = f(a) 
    f_b, _ = f(b)
    x1, x2, x3 = a, (a + b) / 2, b
    f_x1 = f_a
    f_x2, _ = f(x2)
    f_x3 = f_b
    
    cnt = 0              # количество итераций
    x_prev = float('inf')
    x_new = x2
    I = abs(x_prev - x_new)
    
    accuracy = [(b - a)]
    iter_nums = [0]
    
    while I >= eps:
        cnt += 1
        x_prev = x_new
        A = np.array([[x1 ** 2, x1, 1], 
                      [x2 ** 2, x2, 1], 
                      [x3 ** 2, x3, 1]])
        B = np.array([f_x1, f_x2, f_x3])
        
        a2, a1, a0 = np.linalg.solve(A, B)
        x_new = - a1 / (2 * a2) 
        f_new, _ = f(x_new)

        if f_x2 <= f_new:
            x3 = x_new
            f_x3 = f_new
            # x2, x1 остались теми же
        else:
            x1 = x2
            f_x1 = f_x2
            x2 = x_new
            f_x2 = f_new
            # x3 - остался тем же
        
        I = abs(x_prev - x_new)
        accuracy.append(I)
        iter_nums.append(cnt)

    x_min = x_new
    f_min = f_new
    if stat:
        return x_min, f_min, iter_nums, accuracy
    return np.array(x_min)