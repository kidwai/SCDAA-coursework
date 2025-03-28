import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from soft_lqr_problem import SoftLQRProblem

###########################################################################
# 1) Environment: Soft LQR
###########################################################################
# Same environment from previous exercises
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

###############################################################################
# Create the environment
###############################################################################
env = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
env.set_noise(torch.eye(2)*0.5)  # diffusion = 0.5 I
# We do not call env.solve_riccati() for actor-critic, because we do NOT
# want to rely on the known optimal solution in the training process.

###############################################################################
# 2) Actor & Critic Networks
###############################################################################
# Policy (Actor): param -> mean action (2D). We fix covariance = tau * I
class ActorNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)  # mean action in R^2

    def forward(self, t, x):
        """
        t: (batch_size,) or (batch_size,1)
        x: (batch_size,2)
        -> returns (batch_size,2) mean action
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([t, x], dim=1)  # (batch_size,3)
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        mean_a = self.out(h)  # (batch_size,2)
        return mean_a

# Critic: approximates value function v_eta(t,x)
class CriticNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, t, x):
        """
        t: (batch_size,) or (batch_size,1)
        x: (batch_size,2)
        -> returns (batch_size,) a scalar value
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([t, x], dim=1)
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        val = self.out(h)  # (batch_size,1)
        return val.view(-1)

###############################################################################
# 3) Actor-Critic Training
###############################################################################
def train_actor_critic(
    num_epochs=50,
    N_sim=100,
    n_rollouts_per_epoch=5,
    lr_actor=1e-3,
    lr_critic=1e-3,
    seed=42
):
    """
    Implements the actor-critic approach from relaxed_control_and_pol_grad.pdf

    - Actor: param -> action distribution (Gaussian with mean=actor(t,x), cov=tau I)
    - Critic: param -> v_eta(t,x)
    - Each epoch, gather trajectories using the current actor, do:
      (a) Critic update: MSE between v(tn,xn) and [cost_n*dt + v(t_{n+1}, x_{n+1})]
      (b) Actor update: gradient wrt log_prob * advantage,
          where advantage ~ cost_n*dt + v(t_{n+1}, x_{n+1}) - v(tn,xn)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor = ActorNet(hidden_dim=64)
    critic = CriticNet(hidden_dim=64)

    optim_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optim_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    dt = T / N_sim
    cov_mat = tau * torch.eye(2)  # fixed covariance

    for epoch in range(num_epochs):
        # We'll collect data from multiple rollouts
        traj_data = []  # to store (tn, xn, an, cost_n, tn+1, x_{n+1})

        for _ in range(n_rollouts_per_epoch):
            # random initial state
            x = np.random.rand(2)*4 - 2.0
            t = 0.0

            for step in range(N_sim):
                t_tensor = torch.tensor([t], dtype=torch.float32)
                x_tensor = torch.tensor([x], dtype=torch.float32)

                mean_a = actor(t_tensor, x_tensor)[0]  # shape (2,)

                dist = torch.distributions.MultivariateNormal(mean_a, covariance_matrix=cov_mat)
                a = dist.rsample()  # shape(2,)
                logp = dist.log_prob(a)

                a_np = a.detach().numpy()

                # immediate cost
                cost = x @ (env.C @ x) + a_np @ (env.D @ a_np)

                # Next state via Euler-Maruyama
                noise = np.sqrt(dt)*np.random.randn(2)
                x_next = x + (env.H @ x + env.M @ a_np)*dt + (0.5*np.eye(2))@noise
                t_next = t + dt

                # Store in trajectory
                traj_data.append((t, x, a, cost, t_next, x_next))

                x = x_next
                t = t_next

        # 3.1) Critic Update
        # ------------------------------------
        # We'll do a single (or multiple) gradient step(s) on the MSE objective:
        # v(tn, xn) -> cost_n*dt + v(t_{n+1}, x_{n+1})
        # for each transition
        t_list = []
        x_list = []
        target_list = []

        for (tn, xn, an, costn, tn1, xn1) in traj_data:
            t_list.append(tn)
            x_list.append(xn)
            # 1-step TD target: cost_n dt + v(t_{n+1}, x_{n+1})
            with torch.no_grad():
                # Evaluate critic(t_{n+1}, x_{n+1})
                vt1 = critic(
                    torch.tensor([tn1], dtype=torch.float32),
                    torch.tensor([xn1], dtype=torch.float32)
                )[0]
                
                # target
                y = costn * dt + vt1
            target_list.append(y.item())

        # Convert to torch
        t_torch = torch.tensor(t_list, dtype=torch.float32)
        x_torch = torch.tensor(x_list, dtype=torch.float32)
        G_torch = torch.tensor(target_list, dtype=torch.float32)

        # Evaluate critic
        V_pred = critic(t_torch, x_torch)
        # Critic Loss = MSE
        loss_critic = torch.mean((V_pred - G_torch)**2)

        optim_critic.zero_grad()
        loss_critic.backward()
        optim_critic.step()

        # 3.2) Actor Update
        # ------------------------------------
        # We do gradient ascent on sum( log_prob * advantage )
        # advantage_n = cost_n dt + v(t_{n+1}, x_{n+1}) - v(tn, xn)
        logps = []
        advantages = []

        for (tn, xn, a, costn, tn1, xn1) in traj_data:
            # Recompute distribution
            t_torch_s = torch.tensor([tn], dtype=torch.float32)
            x_torch_s = torch.tensor([xn], dtype=torch.float32)
            mean_a_s = actor(t_torch_s, x_torch_s)[0]
            dist_s = torch.distributions.MultivariateNormal(mean_a_s, covariance_matrix=cov_mat)

            a_torch = torch.tensor(a, dtype=torch.float32)
            logp_s = dist_s.log_prob(a_torch)

            with torch.no_grad():
                # advantage = cost_n dt + V(t_{n+1}, x_{n+1}) - V(tn, xn)
                vt1 = critic(
                    torch.tensor([tn1], dtype=torch.float32),
                    torch.tensor([xn1], dtype=torch.float32)
                )[0]
                vt0 = critic(t_torch_s, x_torch_s)[0]
                adv_s = costn*dt + vt1 - vt0

            logps.append(logp_s)
            advantages.append(adv_s)

        # sum over the entire data buffer
        logp_tensor = torch.stack(logps)
        adv_tensor = torch.stack(advantages)

        # Actor Loss = - mean( logp * advantage )
        loss_actor = -torch.mean(logp_tensor * adv_tensor)

        optim_actor.zero_grad()
        loss_actor.backward()
        optim_actor.step()

        # Logging
        if (epoch % 10) == 0:
            print(f"Epoch {epoch:3d} | Critic Loss = {loss_critic.item():.4e}"
                  f" | Actor Loss = {loss_actor.item():.4e}")

    return actor, critic

###############################################################################
# 4) Example Usage
###############################################################################
if __name__ == "__main__":
    actor, critic = train_actor_critic(
        num_epochs=100,
        N_sim=100,
        n_rollouts_per_epoch=5,
        lr_actor=1e-3,
        lr_critic=1e-3,
        seed=42
    )
