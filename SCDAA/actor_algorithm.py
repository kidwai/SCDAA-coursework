import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from soft_lqr_problem import SoftLQRProblem

# Problem setup for actor algorithm
H = torch.tensor([[0.5, 0.5],[0.0, 0.5]])
M = torch.tensor([[1.0, 1.0],[0.0, 1.0]])
C = torch.tensor([[1.0, 0.1],[0.1, 1.0]])
D = torch.tensor([[1.0, 0.1],[0.1, 1.0]]) * 0.1
R = torch.tensor([[1.0, 0.3],[0.3, 1.0]]) * 10.0
T = 0.5
tau = 0.1
gamma = 10.0
time_grid = np.linspace(0, T, 101)
env = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
env.set_noise(torch.eye(2)*0.5)
env.solve_riccati()

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 32)
        self.phi_out = nn.Linear(32, 4)  # to form 2x2 feedback matrix
    def forward(self, t):
        h = torch.relu(self.hidden(t))
        phi_flat = self.phi_out(h)
        phi = phi_flat.view(-1, 2, 2)
        return phi

policy_net = PolicyNet()
optim_actor = optim.Adam(policy_net.parameters(), lr=1e-3)

def V_opt(t, x):
    # Use analytic value from env
    return env.value(torch.tensor([t], dtype=torch.float32),
                     torch.tensor([x], dtype=torch.float32))[0].item()

N_sim = 100
dt = T / N_sim
for epoch in range(50):
    x = (torch.rand(2) * 4 - 2.0).numpy()
    t = 0.0
    traj_log_probs = []
    traj_advantages = []
    for n in range(N_sim):
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        phi = policy_net(t_tensor)[0].detach().numpy()
        mean_action = phi @ x
        a = np.random.multivariate_normal(mean_action, env.tau * env.D_eff_inv)
        # Compute log probability manually
        diff = a - mean_action
        cov = env.tau * env.D_eff_inv
        log_prob = -0.5 * diff.T @ np.linalg.inv(cov) @ diff - 0.5 * np.log(np.linalg.det(2*np.pi*cov))
        traj_log_probs.append(torch.tensor(log_prob, dtype=torch.float32))
        x_next = x + (env.H @ x + env.M @ a) * dt + (0.5*np.eye(2)) @ (np.sqrt(dt)*np.random.randn(2))
        cost = x.dot(env.C @ x) + a.dot(env.D @ a)
        A_hat = cost*dt + V_opt(t+dt, x_next) - V_opt(t, x)
        traj_advantages.append(A_hat)
        x = x_next
        t += dt
    loss_actor = 0.0
    for logp, A in zip(traj_log_probs, traj_advantages):
        loss_actor += logp * A
    loss_actor = loss_actor / len(traj_advantages)
    optim_actor.zero_grad()
    loss_actor.backward()
    optim_actor.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: avg advantage = {np.mean(traj_advantages):.3f}")
