import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from soft_lqr_problem import SoftLQRProblem

# Problem setup for actor-critic
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

# Actor network
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 32)
        self.phi_out = nn.Linear(32, 4)
    def forward(self, t):
        h = torch.relu(self.hidden(t))
        phi_flat = self.phi_out(h)
        phi = phi_flat.view(-1, 2, 2)
        return phi

# Critic network
class OnlyLinearValueNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 64)
        self.out_matrix = nn.Linear(64, 4)
        self.out_offset = nn.Linear(64, 1)
    def forward(self, t):
        h = torch.relu(self.hidden(t))
        mat_elems = self.out_matrix(h)
        M_raw = mat_elems.view(-1, 2, 2)
        K = M_raw @ M_raw.transpose(1, 2)
        offset = self.out_offset(h)
        return K, offset

policy_net = PolicyNet()
critic_net = OnlyLinearValueNN()
optim_actor = optim.Adam(policy_net.parameters(), lr=1e-3)
optim_critic = optim.Adam(critic_net.parameters(), lr=1e-3)

N_sim = 100
dt = T / N_sim

for iteration in range(100):
    trajectory = []
    x = (torch.rand(2) * 4 - 2.0).numpy()
    t = 0.0
    for n in range(N_sim):
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        phi = policy_net(t_tensor)[0].detach().numpy()
        mean_action = phi @ x
        a = np.random.multivariate_normal(mean_action, env.tau * env.D_eff_inv)
        x_next = x + (env.H @ x + env.M @ a) * dt + (0.5*np.eye(2)) @ (np.sqrt(dt)*np.random.randn(2))
        r = x.dot(env.C @ x) + a.dot(env.D @ a)
        trajectory.append((t, x.copy(), a.copy(), r, x_next.copy(), t+dt))
        x = x_next
        t += dt
    # Critic update
    for (t, x, a, r, x_next, t_next) in trajectory:
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, -1)
        t_next_tensor = torch.tensor([[t_next]], dtype=torch.float32)
        x_next_tensor = torch.tensor(x_next, dtype=torch.float32).view(1, -1)
        K, off = critic_net(t_tensor)
        V_pred = (x_tensor.unsqueeze(1) @ K @ x_tensor.unsqueeze(2)).squeeze() + off.squeeze()
        K_next, off_next = critic_net(t_next_tensor)
        V_next_pred = (x_next_tensor.unsqueeze(1) @ K_next @ x_next_tensor.unsqueeze(2)).squeeze() + off_next.squeeze()
        target = torch.tensor(r*dt, dtype=torch.float32) + V_next_pred.detach()
        td_error = V_pred - target
        loss_critic = td_error**2
        optim_critic.zero_grad()
        loss_critic.backward()
        optim_critic.step()
    # Actor update
    loss_actor = 0.0
    for (t, x, a, r, x_next, t_next) in trajectory:
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, -1)
        t_next_tensor = torch.tensor([[t_next]], dtype=torch.float32)
        x_next_tensor = torch.tensor(x_next, dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            K, off = critic_net(t_tensor)
            V_est = (x_tensor.unsqueeze(1) @ K @ x_tensor.unsqueeze(2)).squeeze() + off.squeeze()
            K_next, off_next = critic_net(t_next_tensor)
            V_next_est = (x_next_tensor.unsqueeze(1) @ K_next @ x_next_tensor.unsqueeze(2)).squeeze() + off_next.squeeze()
        A_hat = torch.tensor(r*dt, dtype=torch.float32) + V_next_est - V_est
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        phi = policy_net(t_tensor)[0]
        mean_action = (phi @ torch.tensor(x, dtype=torch.float32))
        cov = torch.tensor(env.tau * env.D_eff_inv, dtype=torch.float32)
        dist = torch.distributions.MultivariateNormal(mean_action, covariance_matrix=cov)
        logp = dist.log_prob(torch.tensor(a, dtype=torch.float32))
        loss_actor += logp * A_hat
    loss_actor = loss_actor / len(trajectory)
    optim_actor.zero_grad()
    (-loss_actor).backward()  # gradient ascent
    optim_actor.step()
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Actor loss = {loss_actor.item():.3e}")
