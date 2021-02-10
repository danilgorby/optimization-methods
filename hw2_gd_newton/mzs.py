#!/usr/bin/env python3
"""Golden-section search method for 1d optimization"""
import numpy as np


def optimize(a, b, f, eps=1e-5, stat=False):    
    K = (5**0.5 - 1) / 2
    I = K * (b - a)      # I_0 = (b - a), I_1 = K * I_0
    
    oraclecalls = 0
    
    
    accuracy = []
    accuracy.append(I)
    iter_nums = [0]
     
    x = b - I       
    y = a + I       
    f_x, _ = f(x)
    f_y, _ = f(y)
    oraclecalls += 2
    cnt = 0
    
    while I >= eps:
        cnt += 1
        if cnt > 15:  # 
            break     #
        I *= K
        accuracy.append(I)
        iter_nums.append(cnt)
        
        if f_x >= f_y:     # берём правый отрезок
            a = x
            # b - остался прежним
            x = y
            f_x = f_y
            y = a + I
            f_y, _ = f(y)
            oraclecalls += 1
        else:              # берём левый отрезок
            # a - остался прежним
            b = y
            y = x
            f_y = f_x
            x = b - I
            f_x, _ = f(x)
            oraclecalls += 1
    
    if f_x <= f_y:
        f_min = f_x
        x_min = x
    else:
        f_min = f_y
        x_min = y
        
    if stat:
        return x_min, f_min, iter_nums, accuracy
    else:
        return np.array(x_min), oraclecalls