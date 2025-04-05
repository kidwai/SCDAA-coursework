import numpy as np
import torch
from lqr_problem import LQRProblem
from soft_lqr_problem import SoftLQRProblem

# Fix random seeds for reproducibility
np.random.seed(123)
torch.manual_seed(123)

# Define the problem parameters (using Figure 1 values)
H = torch.tensor([[0.5, 0.5],
                  [0.0, 0.5]])
M = torch.tensor([[1.0, 1.0],
                  [0.0, 1.0]])
C = torch.tensor([[1.0, 0.1],
                  [0.1, 1.0]])
D = torch.tensor([[1.0, 0.1],
                  [0.1, 1.0]]) * 0.1
R = torch.tensor([[1.0, 0.3],
                  [0.3, 1.0]]) * 10.0

T = 0.5
time_grid = np.linspace(0, T, 101)

# Create strict LQR environment
strict_lqr = LQRProblem(H, M, C, D, R, T, time_grid)
strict_lqr.set_noise(torch.eye(2)*0.5)
strict_lqr.solve_riccati()

# Create soft LQR environment (tau=0.1, gamma=10)
tau = 0.1
gamma = 10.0
soft_lqr = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
soft_lqr.set_noise(torch.eye(2)*0.5)
soft_lqr.solve_riccati()

# Define initial states
init_states = [
    np.array([ 2.0,  2.0]),
    np.array([ 2.0, -2.0]),
    np.array([-2.0, -2.0]),
    np.array([-2.0,  2.0])
]

N_steps = 100
dt = T / N_steps

# Pre-generate Brownian increments (same for both controllers)
brownian_increments = [
    np.random.randn(N_steps, 2) * np.sqrt(dt) for _ in init_states
]

def simulate_trajectories(controller, init_states, brownian_increments):
    trajectories = []
    for i, x0 in enumerate(init_states):
        x = x0.copy()
        traj = [x.copy()]
        t = 0.0
        for n in range(N_steps):
            dW = 0.5 * brownian_increments[i][n]  # diffusion = 0.5 * I
            # Convert to torch
            t_tensor = torch.tensor([t], dtype=torch.float32)
            x_tensor = torch.tensor([x], dtype=torch.float32)
            a = controller.optimal_action(t_tensor, x_tensor).numpy()[0]

            # Euler-Maruyama step
            x = x + (controller.H @ x + controller.M @ a)*dt + dW
            traj.append(x.copy())
            t += dt
        trajectories.append(np.array(traj))
    return trajectories

# Simulate for strict and soft LQR
strict_trajectories = simulate_trajectories(strict_lqr, init_states, brownian_increments)
soft_trajectories   = simulate_trajectories(soft_lqr, init_states, brownian_increments)

# Print trajectory data
print("\n=== Strict LQR Trajectories Data ===")
for i, (traj, init_state) in enumerate(zip(strict_trajectories, init_states)):
    print(f"\nTrajectory {i+1} (Initial state: {init_state})")
    print(f"{'Time':<10} {'x1':<15} {'x2':<15}")
    for n, point in enumerate(traj):
        t = n * dt
        print(f"{t:.3f}      {point[0]:<15.4f} {point[1]:<15.4f}")

print("\n=== Soft LQR Trajectories Data ===")
for i, (traj, init_state) in enumerate(zip(soft_trajectories, init_states)):
    print(f"\nTrajectory {i+1} (Initial state: {init_state})")
    print(f"{'Time':<10} {'x1':<15} {'x2':<15}")
    for n, point in enumerate(traj):
        t = n * dt
        print(f"{t:.3f}      {point[0]:<15.4f} {point[1]:<15.4f}")
