import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class UnconstrainedOptimizer:
    def __init__(self, obj_func, grad_func, hess_func=None, tol=1e-8, max_iter=20000):
        self.f = obj_func
        self.grad = grad_func
        self.hess = hess_func
        self.tol = tol
        self.max_iter = max_iter

    def armijo_line_search(self, x, d, g, c1=1e-4, rho=0.5):
        alpha = 1.0
        f_x = self.f(x)
        directional_derivative = np.dot(g, d)
        
        while True:
            x_new = x + alpha * d
            f_new = self.f(x_new)
            # Thêm kiểm tra np.isnan để chống nổ (overflow)
            if np.isnan(f_new) or f_new > f_x + c1 * alpha * directional_derivative:
                alpha *= rho
                if alpha < 1e-15:  # Bước quá nhỏ, dừng tìm kiếm
                    break
            else:
                break
        return alpha

    def gradient_descent(self, x0):
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        for i in range(self.max_iter):
            g = self.grad(x)
            if np.any(np.isnan(g)) or np.linalg.norm(g) < self.tol:
                break
            
            d = -g
            alpha = self.armijo_line_search(x, d, g)
            x = x + alpha * d
            
            if not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
                history.append(x.copy())
            else:
                break
        # Trả về đúng 3 giá trị: history, số vòng lặp, giá trị f(x) cuối
        return np.array(history), i, self.f(history[-1])

    def newtons_method(self, x0, delta=1e-8):
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        I = np.eye(len(x))
        for i in range(self.max_iter):
            g = self.grad(x)
            if np.any(np.isnan(g)) or np.linalg.norm(g) < self.tol:
                break
                
            H = self.hess(x)
            H_mod = H + delta * I 
            
            try:
                d = np.linalg.solve(H_mod, -g)
            except np.linalg.LinAlgError:
                d = -g
                
            if np.dot(d, g) >= 0:
                d = -g
                
            alpha = self.armijo_line_search(x, d, g)
            x = x + alpha * d
            
            if not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
                history.append(x.copy())
            else:
                break
        return np.array(history), i, self.f(history[-1])

    def bfgs(self, x0):
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        N = len(x)
        H_inv = np.eye(N)
        g = self.grad(x)
        
        for i in range(self.max_iter):
            if np.any(np.isnan(g)) or np.linalg.norm(g) < self.tol:
                break
                
            d = -np.dot(H_inv, g)
            if np.dot(d, g) >= 0:
                d = -g
                
            alpha = self.armijo_line_search(x, d, g)
            x_next = x + alpha * d
            g_next = self.grad(x_next)
            
            s = x_next - x
            y = g_next - g
            
            rho_den = np.dot(y, s)
            if rho_den > 1e-10:
                rho = 1.0 / rho_den
                I = np.eye(N)
                A = I - rho * np.outer(s, y)
                B = I - rho * np.outer(y, s)
                H_inv = np.dot(A, np.dot(H_inv, B)) + rho * np.outer(s, s)
            
            x = x_next
            g = g_next
            
            if not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
                history.append(x.copy())
            else:
                break
        return np.array(history), i, self.f(history[-1])