import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from soft_lqr_problem import SoftLQRProblem

# =============================================
# Problem Setup (from Exercise 2)
# =============================================
H = torch.tensor([[0.5, 0.5], [0.0, 0.5]], dtype=torch.float32)
M = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
C = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float32)
D = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float32) * 0.1
R = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float32) * 10.0
T = 0.5
tau = 0.5  # Increased from 0.1 for better exploration
gamma = 10.0
time_grid = np.linspace(0, T, 101)

# Initialize environment
env = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
env.set_noise(torch.eye(2) * 0.5)
env.solve_riccati()

# =============================================
# Policy Network (Actor)
# =============================================
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 64)  # Increased width
        self.phi_out = nn.Linear(64, 4)  # Outputs 2x2 matrix
        
    def forward(self, t):
        h = torch.relu(self.hidden(t))
        phi = self.phi_out(h).view(-1, 2, 2)
        return phi

policy_net = PolicyNet()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)  # Reduced learning rate

# =============================================
# Training Loop
# =============================================
def V_opt(t, x):
    """Optimal value function from Exercise 2."""
    t_tensor = torch.tensor([t], dtype=torch.float32)
    x_tensor = torch.tensor([x], dtype=torch.float32)
    return env.value(t_tensor, x_tensor)[0].item()

N_sim = 100
dt = T / N_sim
num_epochs = 200

for epoch in range(num_epochs):
    # Initialize state
    x = np.random.rand(2) * 4 - 2.0
    t = 0.0
    log_probs = []
    advantages = []
    
    for _ in range(N_sim):
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # Get policy action
        phi = policy_net(t_tensor)[0]
        mean_action = phi @ x_tensor
        cov = tau * env.D_eff_inv
        dist = torch.distributions.MultivariateNormal(mean_action, covariance_matrix=cov)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        
        # Simulate next state
        action_np = action.detach().numpy()
        noise = np.sqrt(dt) * np.random.randn(2)
        x_next = x + (env.H @ x + env.M @ action_np) * dt + (0.5 * np.eye(2)) @ noise
        
        # Compute advantage
        cost = x.dot(env.C @ x) + action_np.dot(env.D @ action_np)
        advantage = cost * dt + V_opt(t + dt, x_next) - V_opt(t, x)
        
        log_probs.append(log_prob)
        advantages.append(advantage)
        x = x_next
        t += dt
    
    # Normalize advantages
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Policy gradient loss
    loss = -(torch.stack(log_probs) * advantages).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # Gradient clipping
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.3e}, Avg Advantage = {advantages.mean().item():.3f}")

# =============================================
# Verification (Exercise 4.1)
# =============================================
def verify_policy():
    """Compare learned policy vs optimal policy on test points."""
    test_times = [0, T/6, 2*T/6, T/2]
    test_states = [np.array([x, y]) for x in np.linspace(-3, 3, 5) for y in np.linspace(-3, 3, 5)]
    
    for t in test_times:
        for x in test_states:
            # Learned policy
            phi = policy_net(torch.tensor([[t]], dtype=torch.float32))[0]
            a_learned = phi @ torch.tensor(x, dtype=torch.float32)
            
            # Optimal policy
            a_opt = env.optimal_action(
                torch.tensor([t], dtype=torch.float32),
                torch.tensor([x], dtype=torch.float32)
            )[0]
            
            error = np.linalg.norm(a_learned.detach().numpy() - a_opt.numpy())
            print(f"t={t:.3f}, x={x}: Error = {error:.3f}")

verify_policy()