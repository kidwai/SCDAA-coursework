import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from lqr_problem import LQRProblem

# Setup strict LQR using example matrices (as in Figure 1)
H = torch.tensor([[0.5, 0.5],[0.0, 0.5]])
M = torch.tensor([[1.0, 1.0],[0.0, 1.0]])
C = torch.tensor([[1.0, 0.1],[0.1, 1.0]])
D = torch.tensor([[1.0, 0.1],[0.1, 1.0]]) * 0.1
R = torch.tensor([[1.0, 0.3],[0.3, 1.0]]) * 10.0
T = 0.5
time_grid = np.linspace(0, T, 201)
lqr = LQRProblem(H, M, C, D, R, T, time_grid)
lqr.set_noise(torch.eye(2)*0.5)
lqr.solve_riccati()

def estimate_value(lqr, x0, N, M_samples):
    dt = lqr.T / N
    x0 = np.array(x0, dtype=float)
    total_cost = 0.0
    for _ in range(M_samples):
        x = x0.copy()
        cost = 0.0
        t = 0.0
        for n in range(N):
            t_tensor = torch.tensor([t], dtype=torch.float32)
            x_tensor = torch.tensor([x], dtype=torch.float32)
            a = lqr.optimal_action(t_tensor, x_tensor).numpy()[0]
            cost += (x @ lqr.C @ x + a @ lqr.D @ a) * dt
            dw = np.sqrt(dt) * np.random.randn(*x.shape)
            x = x + (lqr.H @ x + lqr.M @ a) * dt + (0.5 * np.eye(lqr.dim)) @ dw
            t += dt
        cost += x @ lqr.R @ x
        total_cost += cost
    return total_cost / M_samples

# --- Convergence test (1): Vary time steps ---
x0 = [1.0, 1.0]
M_samples = 10000  # fix a large number of samples
errors_dt = []
dt_list = []
for k in range(1, 12):
    N = 2**k
    est = estimate_value(lqr, x0, N, M_samples)
    true_val = lqr.value(torch.tensor([0.0]), torch.tensor([x0]))[0].item()
    error = abs(est - true_val)
    errors_dt.append(error)
    dt_list.append(T/N)
    print(f"Time steps: N={N}, dt={T/N:.5f}, estimate={est:.4f}, true={true_val:.4f}, error={error:.3e}")

# Plot error vs dt (time step) on a log–log scale
plt.figure()
plt.loglog(dt_list, errors_dt, marker='o')
plt.xlabel("Time step (dt)")
plt.ylabel("Error")
plt.title("Time-step Convergence of Monte Carlo Simulation")
plt.grid(True, which="both")
plt.savefig("time_step_convergence.png")
plt.show()

# --- Convergence test (2): Vary number of Monte Carlo samples ---
errors_M = []
M_list = []
N_fixed = 10000  # use a fine time discretization
for m_exp in range(0, 6):
    M_s = 2*4**m_exp
    est = estimate_value(lqr, x0, N_fixed, M_s)
    true_val = lqr.value(torch.tensor([0.0]), torch.tensor([x0]))[0].item()
    error = abs(est - true_val)
    errors_M.append(error)
    M_list.append(M_s)
    print(f"MC Samples: M={M_s}, estimate={est:.4f}, true={true_val:.4f}, error={error:.3e}")

# Plot error vs number of Monte Carlo samples on a log–log scale
plt.figure()
plt.loglog(M_list, errors_M, marker='o')
plt.xlabel("Number of Monte Carlo samples")
plt.ylabel("Error")
plt.title("Monte Carlo Sample Size Convergence")
plt.grid(True, which="both")
plt.savefig("mc_sample_convergence.png")
plt.show()
