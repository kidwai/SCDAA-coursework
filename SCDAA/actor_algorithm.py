import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from soft_lqr_problem import SoftLQRProblem

# Problem setup for the actor-only algorithm using Soft LQR.
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
tau = 0.1
gamma = 10.0
time_grid = np.linspace(0, T, 101)

# Create the soft LQR environment.
env = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
env.set_noise(torch.eye(2) * 0.5)
env.solve_riccati()

# Define a policy network that outputs a 2x2 feedback matrix φ(t).
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 32)
        self.phi_out = nn.Linear(32, 4)  # 4 outputs to form a 2x2 matrix.
    def forward(self, t):
        # t: tensor of shape [batch, 1]
        h = torch.relu(self.hidden(t))
        phi_flat = self.phi_out(h)  # shape [batch, 4]
        phi = phi_flat.view(-1, 2, 2)
        return phi

policy_net = PolicyNet()
optim_actor = optim.Adam(policy_net.parameters(), lr=1e-3)

# Helper function: analytic value function from the environment.
def V_opt(t, x):
    # t: scalar, x: numpy array of shape (2,)
    t_tensor = torch.tensor([t], dtype=torch.float32)
    x_tensor = torch.tensor([x], dtype=torch.float32)
    return env.value(t_tensor, x_tensor)[0].item()

# Simulation parameters.
N_sim = 100   # Number of simulation steps per episode.
dt = T / N_sim
num_epochs = 50

for epoch in range(num_epochs):
    # Initialize state uniformly in [-2, 2] for each dimension.
    x = np.random.rand(2) * 4 - 2.0
    t = 0.0
    traj_log_probs = []
    traj_advantages = []
    
    for n in range(N_sim):
        # Convert current time and state to tensors.
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # Get feedback matrix from policy network (do NOT detach so gradients flow).
        phi = policy_net(t_tensor)[0]  # Shape: (2, 2)
        mean_action = phi @ x_tensor    # Shape: (2,)
        
        # Define covariance tensor (convert constant cov to torch tensor).
        cov_tensor = torch.tensor(tau * env.D_eff_inv, dtype=torch.float32)
        
        # Create a multivariate normal distribution with reparameterization.
        dist = torch.distributions.MultivariateNormal(mean_action, covariance_matrix=cov_tensor)
        a = dist.rsample()  # Differentiable sample.
        logp = dist.log_prob(a)
        traj_log_probs.append(logp)
        
        # Convert the action to numpy for simulation.
        a_np = a.detach().cpu().numpy()
        
        # Simulate next state using Euler–Maruyama.
        noise = np.sqrt(dt) * np.random.randn(2)
        x_next = x + (env.H @ x + env.M @ a_np) * dt + (0.5 * np.eye(2)) @ noise
        
        # Compute immediate cost and advantage.
        cost = x.dot(env.C @ x) + a_np.dot(env.D @ a_np)
        advantage = cost * dt + V_opt(t + dt, x_next) - V_opt(t, x)
        traj_advantages.append(advantage)
        
        # Update state and time.
        x = x_next
        t += dt
    
    # Compute the average policy loss over the trajectory.
    # We maximize expected advantage, so we minimize negative of (logp * advantage).
    loss_actor = sum([lp * adv for lp, adv in zip(traj_log_probs, traj_advantages)]) / len(traj_advantages)
    
    optim_actor.zero_grad()
    (-loss_actor).backward()  # Gradient ascent.
    optim_actor.step()
    
    if epoch % 10 == 0:
        avg_adv = np.mean(traj_advantages)
        print(f"Epoch {epoch}: avg advantage = {avg_adv:.3f}, loss_actor = {loss_actor.item():.3e}")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from soft_lqr_problem import SoftLQRProblem

# Problem setup for the actor-only algorithm using Soft LQR.
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
tau = 0.1
gamma = 10.0
time_grid = np.linspace(0, T, 101)

# Create the soft LQR environment.
env = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
env.set_noise(torch.eye(2) * 0.5)
env.solve_riccati()

# Define a policy network that outputs a 2x2 feedback matrix φ(t).
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 32)
        self.phi_out = nn.Linear(32, 4)  # 4 outputs to form a 2x2 matrix.
    def forward(self, t):
        # t: tensor of shape [batch, 1]
        h = torch.relu(self.hidden(t))
        phi_flat = self.phi_out(h)  # shape [batch, 4]
        phi = phi_flat.view(-1, 2, 2)
        return phi

policy_net = PolicyNet()
optim_actor = optim.Adam(policy_net.parameters(), lr=1e-3)

# Helper function: analytic value function from the environment.
def V_opt(t, x):
    # t: scalar, x: numpy array of shape (2,)
    t_tensor = torch.tensor([t], dtype=torch.float32)
    x_tensor = torch.tensor([x], dtype=torch.float32)
    return env.value(t_tensor, x_tensor)[0].item()

# Simulation parameters.
N_sim = 100   # Number of simulation steps per episode.
dt = T / N_sim
num_epochs = 50

for epoch in range(num_epochs):
    # Initialize state uniformly in [-2, 2] for each dimension.
    x = np.random.rand(2) * 4 - 2.0
    t = 0.0
    traj_log_probs = []
    traj_advantages = []
    
    for n in range(N_sim):
        # Convert current time and state to tensors.
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # Get feedback matrix from policy network (do NOT detach so gradients flow).
        phi = policy_net(t_tensor)[0]  # Shape: (2, 2)
        mean_action = phi @ x_tensor    # Shape: (2,)
        
        # Define covariance tensor (convert constant cov to torch tensor).
        cov_tensor = torch.tensor(tau * env.D_eff_inv, dtype=torch.float32)
        
        # Create a multivariate normal distribution with reparameterization.
        dist = torch.distributions.MultivariateNormal(mean_action, covariance_matrix=cov_tensor)
        a = dist.rsample()  # Differentiable sample.
        logp = dist.log_prob(a)
        traj_log_probs.append(logp)
        
        # Convert the action to numpy for simulation.
        a_np = a.detach().cpu().numpy()
        
        # Simulate next state using Euler–Maruyama.
        noise = np.sqrt(dt) * np.random.randn(2)
        x_next = x + (env.H @ x + env.M @ a_np) * dt + (0.5 * np.eye(2)) @ noise
        
        # Compute immediate cost and advantage.
        cost = x.dot(env.C @ x) + a_np.dot(env.D @ a_np)
        advantage = cost * dt + V_opt(t + dt, x_next) - V_opt(t, x)
        traj_advantages.append(advantage)
        
        # Update state and time.
        x = x_next
        t += dt
    
    # Compute the average policy loss over the trajectory.
    # We maximize expected advantage, so we minimize negative of (logp * advantage).
    loss_actor = sum([lp * adv for lp, adv in zip(traj_log_probs, traj_advantages)]) / len(traj_advantages)
    
    optim_actor.zero_grad()
    (-loss_actor).backward()  # Gradient ascent.
    optim_actor.step()
    
    if epoch % 10 == 0:
        avg_adv = np.mean(traj_advantages)
        print(f"Epoch {epoch}: avg advantage = {avg_adv:.3f}, loss_actor = {loss_actor.item():.3e}")
