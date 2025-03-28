import numpy as np
import torch
import matplotlib.pyplot as plt

# Local imports
from lqr_problem import LQRProblem
from soft_lqr_problem import SoftLQRProblem

##############################################################################
# 1) Define problem parameters (from Figure 1 in the coursework)
##############################################################################
H = torch.tensor([[0.5, 0.5],
                  [0.0, 0.5]], dtype=torch.float32)
M = torch.tensor([[1.0, 1.0],
                  [0.0, 1.0]], dtype=torch.float32)
C = torch.tensor([[1.0, 0.1],
                  [0.1, 1.0]], dtype=torch.float32)
D = (torch.tensor([[1.0, 0.1],
                   [0.1, 1.0]], dtype=torch.float32) 
     * 0.1)
R = (torch.tensor([[1.0, 0.3],
                   [0.3, 1.0]], dtype=torch.float32) 
     * 10.0)

T = 0.5
time_grid = np.linspace(0, T, 101)

##############################################################################
# 2) Instantiate Strict and Soft LQR Environments
##############################################################################
# Strict (classical) LQR
strict_lqr = LQRProblem(H, M, C, D, R, T, time_grid)
strict_lqr.set_noise(torch.eye(2) * 0.5)   # diffusion = sigma = 0.5 * I
strict_lqr.solve_riccati()

# Soft (relaxed) LQR (tau=0.1, gamma=10)
tau = 0.1
gamma = 10.0
soft_lqr = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
soft_lqr.set_noise(torch.eye(2) * 0.5)
soft_lqr.solve_riccati()

##############################################################################
# 3) Define trajectories simulation
##############################################################################
def simulate_trajectories(controller, init_states, brownian_increments, N_steps, dt):
    """
    Given a controller (strict_lqr or soft_lqr), simulate forward from
    each initial state using the same set of Brownian increments.

    Args:
      controller : LQRProblem or SoftLQRProblem instance
      init_states: list of np.array, each shape (2,)
      brownian_increments: list of np.arrays of shape (N_steps, 2) for each init state
      N_steps    : number of time steps
      dt         : step size

    Returns:
      trajectories: list of np.array, each of shape (N_steps+1, 2),
                    containing x1, x2 over time
    """
    trajectories = []
    for i, x0 in enumerate(init_states):
        x = x0.copy()
        traj = [x.copy()]
        t = 0.0
        for n in range(N_steps):
            # Brownian increment scaled by the diffusion 0.5
            dW = 0.5 * brownian_increments[i][n]
            # Convert to torch for calling optimal_action
            t_tensor = torch.tensor([t], dtype=torch.float32)
            x_tensor = torch.tensor([x], dtype=torch.float32)
            # Sample an action from the (strict or soft) LQR policy
            a = controller.optimal_action(t_tensor, x_tensor).numpy()[0]
            # Euler-Maruyama step
            x = x + (controller.H @ x + controller.M @ a) * dt + dW
            traj.append(x.copy())
            t += dt
        trajectories.append(np.array(traj))
    return trajectories

##############################################################################
# 4) Compare Strict vs Soft LQR from Different Initial States
##############################################################################
init_states = [
    np.array([ 2.0,  2.0]), 
    np.array([ 2.0, -2.0]),
    np.array([-2.0, -2.0]),
    np.array([-2.0,  2.0])
]

N_steps = 100
dt = T / N_steps

# Pre-generate Brownian increments for each initial state 
# so we can apply the same random path to both controllers
brownian_increments = [
    np.random.randn(N_steps, 2) * np.sqrt(dt) for _ in init_states
]

# Simulate with both controllers
strict_trajectories = simulate_trajectories(strict_lqr, init_states, brownian_increments, N_steps, dt)
soft_trajectories   = simulate_trajectories(soft_lqr, init_states, brownian_increments, N_steps, dt)

##############################################################################
# 5) Plot the results
##############################################################################
# Plot Strict LQR
plt.figure(figsize=(8, 6))
for traj, state in zip(strict_trajectories, init_states):
    plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f"Init: {state}")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Strict (Classical) LQR Trajectories")
plt.legend()
plt.grid(True)
plt.savefig("strict_lqr_trajectories.png", dpi=150)
plt.show()

# Plot Soft LQR
plt.figure(figsize=(8, 6))
for traj, state in zip(soft_trajectories, init_states):
    plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f"Init: {state}")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Soft (Relaxed) LQR Trajectories")
plt.legend()
plt.grid(True)
plt.savefig("soft_lqr_trajectories.png", dpi=150)
plt.show()
