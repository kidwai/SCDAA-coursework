import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from soft_lqr_problem import SoftLQRProblem

# Problem setup for critic algorithm demonstration
H = torch.tensor([[0.1, 0.0],[0.0, 0.1]])
M = torch.tensor([[1.0, 0.0],[0.0, 1.0]])
C = torch.eye(2)
D = torch.eye(2)*0.1
R = torch.eye(2)
T = 0.5
tau = 0.5
gamma = 1.0
time_grid = np.linspace(0, T, 51)
env = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau=tau, gamma=gamma)
env.set_noise(torch.eye(2)*0.5)
env.solve_riccati()

class OnlyLinearValueNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 64)
        self.out_matrix = nn.Linear(64, 4)  # for 2x2 matrix
        self.out_offset = nn.Linear(64, 1)
    def forward(self, t):
        h = torch.relu(self.hidden(t))
        mat_elems = self.out_matrix(h)
        M_raw = mat_elems.view(-1, 2, 2)
        K = M_raw @ M_raw.transpose(1, 2)  # make PSD
        offset = self.out_offset(h)
        return K, offset

critic_net = OnlyLinearValueNN()
optimizer = optim.Adam(critic_net.parameters(), lr=1e-3)

# Generate training data (simulate trajectories)
N_sim = 100   # time steps per trajectory
N_ep = 100    # number of trajectories
dt = T / N_sim
train_data = []
for _ in range(N_ep):
    x = (torch.rand(2) * 4 - 2.0).numpy()
    t = 0.0
    for n in range(N_sim):
        t_tensor = torch.tensor([t], dtype=torch.float32)
        x_tensor = torch.tensor([x], dtype=torch.float32)
        a = env.optimal_action(t_tensor, x_tensor).numpy()[0]
        x_next = x + (env.H @ x + env.M @ a) * dt + (0.5 * np.eye(2)) @ (np.sqrt(dt)*np.random.randn(2))
        r = x.dot(env.C @ x) + a.dot(env.D @ a)
        train_data.append((t, x.copy(), r*dt, x_next.copy(), t+dt))
        x = x_next
        t += dt

# Critic training loop
for epoch in range(10):
    total_loss = 0.0
    for (t, x, cost_dt, x_next, t_next) in train_data:
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        t_next_tensor = torch.tensor([[t_next]], dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, -1)
        x_next_tensor = torch.tensor(x_next, dtype=torch.float32).view(1, -1)
        K, off = critic_net(t_tensor)
        V_pred = (x_tensor.unsqueeze(1) @ K @ x_tensor.unsqueeze(2)).squeeze() + off.squeeze()
        K_next, off_next = critic_net(t_next_tensor)
        V_next_pred = (x_next_tensor.unsqueeze(1) @ K_next @ x_next_tensor.unsqueeze(2)).squeeze() + off_next.squeeze()
        target = torch.tensor(cost_dt, dtype=torch.float32) + V_next_pred.detach()
        loss = (V_pred - target) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Critic loss = {total_loss/len(train_data):.3e}")
