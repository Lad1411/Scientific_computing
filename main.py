import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from Optimizer import UnconstrainedOptimizer
from Problem import Rosenbrock

def plot_trajectory(case_name, x0, hist_gd, hist_nt, hist_bfgs, N):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Thiết lập giới hạn khung hình dựa trên điểm bắt đầu để đồ thị không bị lệch
    x_min, x_max = min(-2.0, x0[0] - 1), max(2.0, x0[0] + 1)
    y_min, y_max = min(-1.0, x0[1] - 1), max(2.5, x0[1] + 1)
    
    x_grid = np.linspace(x_min, x_max, 400)
    y_grid = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    # Tính toán Contour (Nếu N>2, chiếu lên mặt phẳng x1-x2 bằng cách gán các chiều còn lại = 1.0)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if N == 2:
                Z[i, j] = Rosenbrock.evaluate([X[i, j], Y[i, j]])
            else:
                point = [X[i, j], Y[i, j]] + [1.0] * (N - 2)
                Z[i, j] = Rosenbrock.evaluate(point)

    contour = ax.contourf(X, Y, Z, levels=np.logspace(-1, 4, 35), norm=LogNorm(), cmap='inferno', alpha=0.8)
    plt.colorbar(contour, label='log(f(x))', ax=ax)
    
    # Lọc quỹ đạo chỉ lấy phần nằm trong khung hình để đồ thị không bị kéo giãn do Divergence
    def filter_hist(hist):
        valid_idx = (hist[:, 0] >= x_min-5) & (hist[:, 0] <= x_max+5) & \
                    (hist[:, 1] >= y_min-5) & (hist[:, 1] <= y_max+5)
        return hist[valid_idx]

    h_gd = filter_hist(hist_gd)
    h_nt = filter_hist(hist_nt)
    h_bfgs = filter_hist(hist_bfgs)

    # Vẽ quỹ đạo chiếu lên 2 trục đầu tiên
    ax.plot(h_gd[:, 0], h_gd[:, 1], 'o-', color='orange', label=f'Gradient Descent', markersize=3, linewidth=1.5, alpha=0.8)
    ax.plot(h_nt[:, 0], h_nt[:, 1], 's-', color='cyan', label=f"Newton's Method", markersize=3, linewidth=1.5, alpha=0.8)
    ax.plot(h_bfgs[:, 0], h_bfgs[:, 1], '^-', color='purple', label=f'BFGS', markersize=4, linewidth=1.5, alpha=0.8)

    ax.plot(x0[0], x0[1], 'wX', markersize=12, label='Start', markeredgecolor='black')
    ax.plot(1.0, 1.0, 'w*', markersize=18, label='Global Optimum (1, 1)', markeredgecolor='black')

    ax.set_title(f'{case_name} (N={N})', fontsize=14, fontweight='bold')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hiển thị biểu đồ mà không chặn quá trình chạy để render tuần tự các case
    plt.show(block=False)


def main():
    opt = UnconstrainedOptimizer(Rosenbrock.evaluate, Rosenbrock.gradient, Rosenbrock.hessian, max_iter=10000)
    
    # Định nghĩa 1 Case bình thường và 4 Edge Cases
    cases = {
        "0. Normal Case": [-1.5, 1.0],
        "1. Standard Hard-Start": [-1.2, 1.0],
        "2. Far Initialization": [4.0, 4.0],
        "3. Flat Origin": [0.0, 0.0],
        "4. High-Dimensional Trap (N=10)": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    }
    
    print("=" * 105)
    print(f"{'CASE NAME':<33} | {'ALGORITHM':<15} | {'ITERS':<7} | {'FINAL LOSS f(x)':<15} | {'STATUS'}")
    print("=" * 105)
    
    for case_name, x0 in cases.items():
        N = len(x0)
        
        # 1. Chạy Gradient Descent
        hist_gd, iters_gd, loss_gd = opt.gradient_descent(x0)
        status_gd = "Diverged/Max Iters" if iters_gd == opt.max_iter or np.isnan(loss_gd) else "Converged"
        
        # 2. Chạy Newton's Method
        hist_nt, iters_nt, loss_nt = opt.newtons_method(x0)
        # Bắt bẫy Local Minimum (Thường xảy ra ở N>=4, f(x) bị kẹt ở mức ~3.99)
        if 3.0 < loss_nt < 5.0 and iters_nt < opt.max_iter:
            status_nt = "Local Minimum Trap"
        else:
            status_nt = "Diverged/Max Iters" if iters_nt == opt.max_iter or np.isnan(loss_nt) else "Converged"

        # 3. Chạy BFGS
        hist_bfgs, iters_bfgs, loss_bfgs = opt.bfgs(x0)
        status_bfgs = "Diverged/Max Iters" if iters_bfgs == opt.max_iter or np.isnan(loss_bfgs) else "Converged"
        
        # In log kết quả
        print(f"\033[1m{case_name} (N={N})\033[0m")
        print(f"{'':<33} | {'Gradient Desc':<15} | {iters_gd:<7} | {loss_gd:<15.4e} | {status_gd}")
        print(f"{'':<33} | {'Newton Method':<15} | {iters_nt:<7} | {loss_nt:<15.4e} | {status_nt}")
        print(f"{'':<33} | {'BFGS':<15} | {iters_bfgs:<7} | {loss_bfgs:<15.4e} | {status_bfgs}")
        print("-" * 105)
        
        # Gọi hàm visualize (Vẽ lần lượt)
        plot_trajectory(case_name, x0, hist_gd, hist_nt, hist_bfgs, N)

    # Giữ các cửa sổ biểu đồ không bị tắt sau khi vòng lặp kết thúc
    print("\nQuá trình tối ưu đã hoàn tất. Vui lòng kiểm tra các cửa sổ đồ thị (5 Figures).")
    plt.show()

if __name__ == "__main__":
    main()