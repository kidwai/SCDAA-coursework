import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from lqr_problem import LQRProblem
from soft_lqr_problem import SoftLQRProblem

# Define the problem parameters (using Figure 1 values)
H = torch.tensor([[0.5, 0.5],[0.0, 0.5]])
M = torch.tensor([[1.0, 1.0],[0.0, 1.0]])
C = torch.tensor([[1.0, 0.1],[0.1, 1.0]])
D = torch.tensor([[1.0, 0.1],[0.1, 1.0]]) * 0.1
R = torch.tensor([[1.0, 0.3],[0.3, 1.0]]) * 10.0
T = 0.5
time_grid = np.linspace(0, T, 101)

# Strict LQR
strict_lqr = LQRProblem(H, M, C, D, R, T, time_grid)
strict_lqr.set_noise(torch.eye(2)*0.5)
strict_lqr.solve_riccati()

# Soft LQR (tau=0.1, gamma=10)
tau = 0.1
gamma = 10.0
soft_lqr = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
soft_lqr.set_noise(torch.eye(2)*0.5)
soft_lqr.solve_riccati()

# Four different initial states:
init_states = [np.array([2.0, 2.0]), np.array([2.0, -2.0]), 
               np.array([-2.0, -2.0]), np.array([-2.0, 2.0])]

N_steps = 100
dt = T / N_steps

# Pre-generate Brownian increments for each trajectory (same for both controllers)
brownian_increments = [np.random.randn(N_steps, 2) * np.sqrt(dt) for _ in init_states]

# Simulate trajectories
def simulate_trajectories(controller, init_states, brownian_increments):
    trajectories = []
    for i, x0 in enumerate(init_states):
        x = x0.copy()
        traj = [x.copy()]
        t = 0.0
        for n in range(N_steps):
            # Get same noise for both controllers
            dW = (0.5 * np.eye(2)) @ brownian_increments[i][n]
            t_tensor = torch.tensor([t], dtype=torch.float32)
            x_tensor = torch.tensor([x], dtype=torch.float32)
            a = controller.optimal_action(t_tensor, x_tensor).numpy()[0]
            x = x + (controller.H @ x + controller.M @ a) * dt + dW
            traj.append(x.copy())
            t += dt
        trajectories.append(np.array(traj))
    return trajectories

strict_trajectories = simulate_trajectories(strict_lqr, init_states, brownian_increments)
soft_trajectories = simulate_trajectories(soft_lqr, init_states, brownian_increments)

# Plot trajectories for strict LQR
plt.figure(figsize=(8, 6))
for traj, state in zip(strict_trajectories, init_states):
    plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f"Init: {state}")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Strict LQR Trajectories")
plt.legend()
plt.grid(True)
plt.savefig("strict_lqr_trajectories.png")
plt.show()

# Plot trajectories for soft LQR
plt.figure(figsize=(8, 6))
for traj, state in zip(soft_trajectories, init_states):
    plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f"Init: {state}")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Soft (Relaxed) LQR Trajectories")
plt.legend()
plt.grid(True)
plt.savefig("soft_lqr_trajectories.png")
plt.show()
