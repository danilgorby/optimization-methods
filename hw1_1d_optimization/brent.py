#!/usr/bin/env python3
"""Brent's method for 1d optimization"""
import numpy as np


def optimize(a, b, f, eps=1e-8, stat=False):
    K = (3 - 5**0.5) / 2
    
    # инициализация лучших точек и значений в них
    x1, x2, x3 = [a + K * (b - a)] * 3
    f_x1, _ = f(x1)
    f_x2, _ = f(x2)
    f_x3, _ = f(x3)
    
    # длины текущего и предыдущего интервалов
    I_curr = b - a
    I_prev = b - a 
    
    cnt = 0
    accuracy = [I_curr]
    iter_nums = [0]
    
    # x_new - следующая точка итерационного процесса
    x_new = float('inf')
    
    while I_curr > eps:
        parabola = False
        cnt += 1
        
        # I_preprev - длина на предпредыдущем шаге
        I_preprev = I_prev 
        I_prev = I_curr
        
        tol = eps * abs(x1) + eps / 10

        # критерий остановки
        if abs(x1 - (a + b) / 2) + (b - a) / 2 <= 2 * tol:
            break
        
        # пробуем применить метод парабол
        if x1 != x2 and x1 != x3 and x2 != x3: 
            if f_x1 < f_x2 and f_x1 < f_x3: 
                A = np.array([[x1 ** 2, x1, 1], 
                              [x2 ** 2, x2, 1], 
                              [x3 ** 2, x3, 1]])
                B = np.array([f_x1, f_x2, f_x3])
                
                # коэффициенты аппроксимирующего многочлена 
                a2, a1, a0 = np.linalg.solve(A, B)
                # точка минимума многочлена
                x_new = - a1 / (2 * a2) 
                
                # принимать ли найденную точку x_new
                if  (a <= x_new <= b) and abs(x_new - x1) < I_preprev / 2:
                    # x_new - принята
                    parabola = True
                    if (x_new - a) < 2 * tol or (b - x_new) < 2 * tol:
                        x_new = x1 - np.sign(x1 - (a + b) / 2) * tol
                else:
                    parabola = False
            
        if not parabola:                       # применяем МЗС
            if x1 < (a + b) / 2:
                x_new = x1 + K * (b - x1)
                I_prev = b - x1 
            else:
                x_new = x1 - K * (x1 - a)
                I_prev = x1 - a 
                
        I_curr = abs(x_new - x1)
        
        f_x_new, _ = f(x_new)
        if f_x_new <= f_x1:
            if x_new >= x1:
                a = x1
            else:
                b = x1
            # сдвиг 3-х лучших
            x3 = x2
            f_x3 = f_x2
            x2 = x1
            f_x2 = f_x1
            x1 = x_new
            f_x1 = f_x_new
        else:
            if x_new >= x1:
                b = x_new
            else:
                a = x_new
            # правильный сдвиг 3-х лучших
            if f_x_new <= f_x2 or x2 == x1:
                x3 = x2
                x2 = x_new
                f_x3 = f_x2
                f_x2 = f_x_new
            elif f_x_new <= f_x3 or x3 == x1 or x3 == x2:
                x3 = x_new
                f_x3 = f_x_new 
                
        accuracy.append(I_curr)
        iter_nums.append(cnt)
                
    x_min = x_new
    f_min = f_x_new
    if stat:
        return x_min, f_min, iter_nums, accuracy
    return np.array(x_min)