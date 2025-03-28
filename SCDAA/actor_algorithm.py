import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from soft_lqr_problem import SoftLQRProblem

##############################################################################
# 1) Environment & Setup
##############################################################################
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
env.solve_riccati()  # So env.value() uses the correct (S(t), b(t)) from the modified Riccati eqn

##############################################################################
# 2) Policy Network
##############################################################################
# We'll parametrize a Gaussian policy with mean = ActorNet(t,x) and fixed covariance = tau * I.

class ActorNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # We'll feed in (t, x1, x2) => 3 inputs
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)  # mean action in R^2

    def forward(self, t, x):
        """
        t: (batch_size,) or (batch_size,1)    -> time
        x: (batch_size,2)                    -> state
        returns: (batch_size,2) -> mean action
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)  # (batch_size,1)

        inp = torch.cat([t, x], dim=1)  # shape: (batch_size, 3)
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        mean_action = self.out(h)  # shape: (batch_size, 2)
        return mean_action


##############################################################################
# 3) Function to Evaluate the Policy
##############################################################################
def evaluate_policy(actor, env, N_sim=100, n_rollouts=5):
    """
    Evaluate how close the policy is to the known optimal solution.
    We'll compare the total cost of the rollout vs. the 'optimal' cost.

    Args:
      actor:      trained ActorNet
      env:        SoftLQRProblem with env.value(...) known
      N_sim:      # of steps per simulated trajectory
      n_rollouts: how many independent rollouts to average over

    Returns:
      avg_rel_cost_diff: the average difference from the optimal cost as fraction
                         or some measure of absolute difference
    """
    dt = env.T / N_sim
    costs_diffs = []

    for _ in range(n_rollouts):
        # Random initial state
        x = np.random.rand(2) * 4.0 - 2.0
        t = 0.0

        # We'll track total cost from rolling out the learned policy
        rollout_cost = 0.0

        # We'll also track the "optimal" cost for that same (t, x) as if we did nothing right away,
        # i.e. v_opt(0, x). That’s what we compare to.
        # Actually, let's define exact_opt_cost = env.value(t, x)[0].item()  # but note we sum costs over time

        # We'll do the partial sums approach:
        # cost_0 = env.value(t, x). The difference between integrated cost and the
        # "value function difference" should match the integral cost. We'll do a direct integral approach.
        cost_init = env.value(torch.tensor([t]), torch.tensor([x], dtype=torch.float32))[0].item()

        for _ in range(N_sim):
            # Convert to torch
            t_tensor = torch.tensor([t], dtype=torch.float32)
            x_tensor = torch.tensor([x], dtype=torch.float32)

            # Mean action from actor
            mean_action = actor(t_tensor, x_tensor)[0]
            # Sample from Gaussian with covariance = tau * I
            cov_mat = tau * torch.eye(2)
            dist = torch.distributions.MultivariateNormal(mean_action, covariance_matrix=cov_mat)
            a = dist.sample()  # not rsample, because we are not updating policy in evaluate
            a_np = a.detach().numpy()

            # One-step cost
            cost_step = x @ (env.C @ x) + a_np @ (env.D @ a_np)
            rollout_cost += cost_step * dt

            # Euler-Maruyama
            noise = np.sqrt(dt) * np.random.randn(2)
            x_next = x + (env.H @ x + env.M @ a_np)*dt + (0.5 * np.eye(2)) @ noise

            x = x_next
            t += dt

        # Compare to the "optimal" cost we *would have had* starting from same initial (0, x):
        # That is env.value(0, x) if we interpret cost from t=0 to T. But we must define
        # "exact_opt_cost" = v(0, x).
        # Then difference = rollout_cost - exact_opt_cost
        exact_opt_cost = env.value(torch.zeros(1), 
                                   torch.tensor([x_tensor[0].numpy()], dtype=torch.float32))[0].item()

        # The difference "rollout_cost - (v_opt(0,x))" might be a measure of how close
        # we are to the optimum. Because (v_opt(0,x)) is the minimal possible cost from 0 to T,
        # if our policy was truly optimal, we'd get ~ the same cost. Usually you expect
        # rollout_cost >= v_opt(0, x).
        cost_diff = rollout_cost - exact_opt_cost
        costs_diffs.append(cost_diff)

    avg_cost_diff = np.mean(costs_diffs)
    return avg_cost_diff


##############################################################################
# 4) Actor-Only Training with Verification
##############################################################################
def train_actor_only(
    num_epochs=50,
    N_sim=100,
    n_rollouts_per_epoch=5,
    lr=1e-3,
    seed=42,
    evaluate_every=10
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor = ActorNet(hidden_dim=64)
    optim_actor = optim.Adam(actor.parameters(), lr=lr)

    dt = T / N_sim
    cov_mat = tau * torch.eye(2)  # fixed covariance

    for epoch in range(num_epochs):
        epoch_logps = []
        epoch_advs = []

        for _ in range(n_rollouts_per_epoch):
            # Random initial state in [-2,2]^2
            x = np.random.rand(2)*4 - 2.0
            t = 0.0

            traj_logps = []
            traj_advs = []

            for _ in range(N_sim):
                # Torchify
                t_tensor = torch.tensor([t], dtype=torch.float32)
                x_tensor = torch.tensor([x], dtype=torch.float32)

                # Actor forward -> mean action
                mean_action = actor(t_tensor, x_tensor)[0]  # shape (2,)

                # Sample from policy
                dist = torch.distributions.MultivariateNormal(mean_action, covariance_matrix=cov_mat)
                a = dist.rsample()
                logp = dist.log_prob(a)

                # Simulate environment step
                a_np = a.detach().numpy()
                noise = np.sqrt(dt)*np.random.randn(2)
                x_next = x + (env.H @ x + env.M @ a_np)*dt + (0.5*np.eye(2))@noise

                # cost for advantage
                cost = x @ (env.C @ x) + a_np @ (env.D @ a_np)
                adv = cost*dt + env.value(torch.tensor([t+dt]), 
                                          torch.tensor([x_next], dtype=torch.float32))[0].item() \
                                - env.value(torch.tensor([t]),
                                            torch.tensor([x], dtype=torch.float32))[0].item()

                traj_logps.append(logp)
                traj_advs.append(adv)

                # Update state/time
                x = x_next
                t += dt

            epoch_logps.extend(traj_logps)
            epoch_advs.extend(traj_advs)

        # Compute the policy gradient loss (negative for gradient ascent on -cost)
        loss_actor = -sum(lp*ad for lp, ad in zip(epoch_logps, epoch_advs)) / len(epoch_advs)

        optim_actor.zero_grad()
        loss_actor.backward()
        optim_actor.step()

        if epoch % evaluate_every == 0:
            avg_adv = np.mean(epoch_advs)
            avg_cost_diff = evaluate_policy(actor, env, N_sim=N_sim, n_rollouts=10)
            print(f"Epoch {epoch:3d}:  avg advantage = {avg_adv: .4f},  loss_actor = {loss_actor.item(): .3e},  "
                  f"avg rollout cost diff = {avg_cost_diff: .4f}")

    return actor


if __name__ == "__main__":
    trained_actor = train_actor_only(num_epochs=100, N_sim=100, n_rollouts_per_epoch=5, lr=1e-3, seed=42)
