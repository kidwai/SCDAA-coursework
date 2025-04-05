import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from soft_lqr_problem import SoftLQRProblem

# 1) Environment Setup (same as Exercise 2 for the "soft LQR" problem)
H = torch.tensor([[0.5, 0.5],
                  [0.0, 0.5]], dtype=torch.float32)
M = torch.tensor([[1.0, 1.0],
                  [0.0, 1.0]], dtype=torch.float32)
C = torch.tensor([[1.0, 0.1],
                  [0.1, 1.0]], dtype=torch.float32)
D = torch.tensor([[1.0, 0.1],
                  [0.1, 1.0]], dtype=torch.float32) * 0.1
R = torch.tensor([[1.0, 0.3],
                  [0.3, 1.0]], dtype=torch.float32) * 10.0

T = 0.5
time_grid = np.linspace(0, T, 101)
tau = 0.1
gamma = 10.0

env = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
env.set_noise(torch.eye(2)*0.5)
env.solve_riccati()  

# 2) Critic Network: Approximating the Value Function v(t, x; eta)

class CriticNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)   # returns scalar v_eta(t, x)

    def forward(self, t, x):
        """
        t: (batch_size,) or (batch_size,1)
        x: (batch_size,2)
        returns: (batch_size,) = value function
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)  # shape: (batch_size,1)
        inp = torch.cat([t, x], dim=1)  # shape: (batch_size,3)
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        v = self.out(h)  # shape: (batch_size,1)
        return v.view(-1)  # flatten to (batch_size,)

# 3) Critic-Only Algorithm


def train_critic_only(
    num_episodes=50,
    N_sim=100,
    lr=1e-3,
    seed=42
):
    """
    Implements the 'critic-only' approach:
    - Fix the policy = env.optimal_action(t, x)
    - Parametrize value function v(t, x; eta)
    - Minimizes the MSE between v(t_n, x_n) and the Monte Carlo return from n -> N.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    critic = CriticNet(hidden_dim=64)
    optim_critic = optim.Adam(critic.parameters(), lr=lr)

    dt = T / N_sim

    for episode in range(num_episodes):
        # Sample an initial state in [-2, 2]^2
        x0 = np.random.rand(2) * 4 - 2.0
        t0 = 0.0

        # We'll store (tn, xn), plus the "ground-truth return" from n -> N
        time_list = []
        state_list = []
        returns_list = []

        # Accumulate trajectory
        t = t0
        x = x0
        cost_history = []
        logp_history = []
        state_history = [(t, x)]
        for n in range(N_sim):
            # Convert to torch for env
            t_tensor = torch.tensor([t], dtype=torch.float32)
            x_tensor = torch.tensor([x], dtype=torch.float32)
            # The environment's known *optimal action distribution*
            # We can sample from it
            a_sample = env.optimal_action(t_tensor, x_tensor)[0]  # shape: (2,)
            a_np = a_sample.numpy()

            # Compute cost_n, including entropic reg
            cost_n = x @ (env.C @ x) + a_np @ (env.D @ a_np)


            # store cost
            cost_history.append(cost_n)

            # Euler-Maruyama to get x_{n+1}
            noise = np.sqrt(dt)*np.random.randn(2)
            x_next = x + (env.H @ x + env.M @ a_np)*dt + (0.5*np.eye(2))@noise

            state_history.append((t+dt, x_next))
            x = x_next
            t = t + dt

        # terminal cost g(XT)
        x_final = state_history[-1][1]
        final_cost = x_final @ (env.R @ x_final)  # g(x) = x^T R x

        # Return_n = sum_{k=n}^{N-1} cost_history[k]*dt + final_cost
        returns = []
        running_sum = final_cost
        # We'll do a backwards accumulation
        for i in reversed(range(N_sim)):
            running_sum += cost_history[i] * dt
            returns.append(running_sum)
        returns.reverse()


        for i in range(N_sim):
            (tn, xn) = state_history[i]
            time_list.append(tn)
            state_list.append(xn)
            returns_list.append(returns[i])

        # Convert to torch
        t_torch = torch.tensor(time_list, dtype=torch.float32)
        x_torch = torch.tensor(state_list, dtype=torch.float32)
        G_torch = torch.tensor(returns_list, dtype=torch.float32)

        # We want to minimize MSE:
        # L(eta) = sum_{n=0..N-1} [v(tn, xn; eta) - Gn]^2
        v_pred = critic(t_torch, x_torch)
        loss_critic = torch.mean((v_pred - G_torch)**2)

        optim_critic.zero_grad()
        loss_critic.backward()
        optim_critic.step()

        if episode % 10 == 0:
            # Print progress
            print(f"Episode {episode:2d} | Critic MSE Loss = {loss_critic.item(): .4e}")

    return critic


if __name__ == "__main__":
    trained_critic = train_critic_only(num_episodes=90, N_sim=100, lr=1e-3, seed=42)
