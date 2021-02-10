#!/usr/bin/env python

import numpy as np
from scipy.optimize import line_search as wolf

class line_search_wolf:
    def __init__(self, c1=0.0001, c2=0.9):
        self.c1 = c1
        self.c2 = c2
        
    def __call__(self, f, w, direction, stata=False):
        F = f.value
        gr_F = lambda x: f.grad(x).reshape(-1)
        alpha, fc, gc, _, _, _ = wolf(F, gr_F, w, direction, c1=self.c1, c2=self.c2)
        funcalls = max(fc, gc)
        
        if alpha is None:
            armijo = line_search_armijo()
            alpha, funcalls = armijo(f, w, direction, stata=True)
            
        if stata:
            return np.array(alpha), funcalls
        return np.array(alpha)
    
class line_search_armijo:
    def __init__(self, c1=0.4, eta=2.):
        self.c1 = c1
        self.eta = eta
        
    def __call__(self, f, w, direction, stata=False):
        print("ima in armijo w", type(w))
        print("ima in armijo d", type(direction))
        phi = lambda a: f.value(w + a * direction)
        dphi = lambda a: direction.T @ f.grad(w + a * direction)
        
        funcalls = 0      
        phi0 = phi(0)
        dphi0 = dphi(0)        
        funcalls += 1
        
        iter_num = 0
        alpha = 1.
        first_cond = phi(alpha) <= phi0 + self.c1 * alpha * dphi0
        print("ima in armijo w", type(self.eta * alpha))
        second_cond = phi(self.eta * alpha) >= phi0 + self.c1 * self.eta * alpha * dphi0
        funcalls += 2

        if first_cond:
            #while first_cond and not second_cond:
            while not second_cond:
                iter_num += 1
                if iter_num >= 100:
                    break
                alpha = self.eta * alpha
                second_cond = phi(self.eta * alpha) >= phi0 + self.c1 * self.eta * alpha * dphi0
                funcalls += 1
            return np.array(alpha), funcalls

        if second_cond:
            # while second_cond and not first_cond:
            while not first_cond:
                iter_num += 1
                if iter_num >= 100:
                    break
                alpha = alpha / self.eta
                first_cond = phi(alpha) <= phi0 + self.c1 * alpha * dphi0
                funcalls += 1
                
        if stata:
            return np.array(alpha), funcalls
        return np.array(alpha)

