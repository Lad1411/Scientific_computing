import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ==========================================
# ĐỊNH NGHĨA BÀI TOÁN (Hàm Rosenbrock)
# ==========================================
class Rosenbrock:
    """Định nghĩa hàm Rosenbrock N-chiều, Gradient và ma trận Hessian."""
    
    @staticmethod
    def evaluate(x):
        """Tính giá trị hàm mục tiêu f(x)."""
        x = np.asarray(x)
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

    @staticmethod
    def gradient(x):
        """Tính vector gradient bậc 1."""
        x = np.asarray(x)
        N = len(x)
        g = np.zeros(N)
        if N == 2:
            g[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
            g[1] = 200 * (x[1] - x[0]**2)
        else:
            g[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
            g[1:-1] = 200 * (x[1:-1] - x[:-2]**2) - 400 * x[1:-1] * (x[2:] - x[1:-1]**2) - 2 * (1 - x[1:-1])
            g[-1] = 200 * (x[-1] - x[-2]**2)
        return g

    @staticmethod
    def hessian(x):
        """Tính ma trận Hessian bậc 2."""
        x = np.asarray(x)
        N = len(x)
        H = np.zeros((N, N))
        if N == 2:
            H[0, 0] = 1200 * x[0]**2 - 400 * x[1] + 2
            H[0, 1] = H[1, 0] = -400 * x[0]
            H[1, 1] = 200
        else:
            H[0, 0] = 1200 * x[0]**2 - 400 * x[1] + 2
            H[0, 1] = -400 * x[0]
            for i in range(1, N - 1):
                H[i, i-1] = -400 * x[i-1]
                H[i, i] = 200 + 1200 * x[i]**2 - 400 * x[i+1] + 2
                H[i, i+1] = -400 * x[i]
            H[-1, -2] = -400 * x[-2]
            H[-1, -1] = 200
        return H