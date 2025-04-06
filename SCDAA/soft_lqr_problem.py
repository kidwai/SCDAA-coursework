import numpy as np
import torch
from lqr_problem import LQRProblem
from scipy.integrate import solve_ivp

class SoftLQRProblem(LQRProblem):
    def __init__(self, H, M, C, D, R, T, time_grid, tau, gamma):
        """
        Initialize the soft LQR problem with additional parameters:
          tau  : strength of entropic regularization,
          gamma: variance of the prior normal density.
        """
        super().__init__(H, M, C, D, R, T, time_grid)
        self.tau = tau
        self.gamma = gamma
        self.D_eff = self.D + (tau/(2*(gamma**2))) * np.eye(self.dim)
        self.D_eff_inv = np.linalg.inv(self.D_eff)
        m = self.M.shape[1]
        Sigma = self.D_eff_inv
        # Constant term from entropy: C_const = - tau * (0.5*m*ln(tau) + 0.5*ln(det(Sigma)) - m*ln(gamma))
        self.C_const = - tau * (0.5*m*np.log(tau) + 0.5*np.log(np.linalg.det(Sigma)) - m*np.log(gamma))
    
    def solve_riccati(self):
        """Solve the modified Riccati ODE for the soft LQR."""
        n = self.dim
        y_T = np.concatenate([self.R.reshape(-1), [0.0]])
        def riccati_ode(t, y):
            S = y[:n*n].reshape((n, n))
            # Modified Riccati ODE with D_eff:
            dS_dt = -(self.H.T @ S + S @ self.H - S @ self.M @ self.D_eff_inv @ self.M.T @ S + self.C)
            dS_dt_flat = dS_dt.reshape(-1)
            db_dt = - (np.trace(self.sigma @ self.sigma.T @ S) if self.sigma is not None else 0.0)
            db_dt -= self.C_const
            return np.concatenate([dS_dt_flat, [db_dt]])
        sol = solve_ivp(riccati_ode, [self.T, 0], y_T, t_eval=self.time_grid[::-1],
                        rtol=1e-8, atol=1e-8)
        S_sol = sol.y[:-1, :].T[::-1]
        b_sol = sol.y[-1, :][::-1]
        self.S_t = S_sol.reshape(-1, n, n)
        self.b_t = b_sol

    def optimal_action_mean(self, t_query, x_query):
        """
        Compute the mean of the optimal action distribution:
        a_mean(t,x) = - (D + tau/(2gamma²) I)⁻¹ Mᵀ S(t)x.
        """
        t_np = t_query.cpu().numpy().reshape(-1)
        x_np = x_query.cpu().numpy().reshape(-1, self.dim)
        actions_mean = []
        for tt, xx in zip(t_np, x_np):
            idx = np.searchsorted(self.time_grid, tt, side='right') - 1
            idx = max(0, min(idx, len(self.time_grid)-1))
            S = self.S_t[idx]
            a_mean = - self.D_eff_inv @ (self.M.T @ (S @ xx))
            actions_mean.append(a_mean)
        return torch.tensor(np.array(actions_mean), dtype=torch.float32)
    
    def optimal_action(self, t_query, x_query):
        """
        Sample an optimal action from the Gaussian distribution with:
          mean = optimal_action_mean(t,x)
          covariance = tau * D_eff_inv.
        """
        mean = self.optimal_action_mean(t_query, x_query).numpy()
        actions = []
        cov = self.tau * self.D_eff_inv
        for mu in mean:
            a_sample = np.random.multivariate_normal(mu, cov)
            actions.append(a_sample)
        return torch.tensor(np.array(actions), dtype=torch.float32)

if __name__ == '__main__':
    # Example usage for soft LQR
    H = torch.tensor([[0.5, 0.5],[0.0, 0.5]])
    M = torch.tensor([[1.0, 1.0],[0.0, 1.0]])
    C = torch.tensor([[1.0, 0.1],[0.1, 1.0]])
    D = torch.tensor([[1.0, 0.1],[0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3],[0.3, 1.0]]) * 10.0
    T = 0.5
    time_grid = np.linspace(0, T, 201)
    tau = 0.1
    gamma = 10.0
    soft_lqr = SoftLQRProblem(H, M, C, D, R, T, time_grid, tau, gamma)
    soft_lqr.set_noise(torch.eye(2)*0.5)
    soft_lqr.solve_riccati()
    t_query = torch.tensor([0.0])
    x_query = torch.tensor([[2.0, 2.0]])
    print("Soft LQR value at (0, [2,2]):", soft_lqr.value(t_query, x_query))
    print("Soft LQR optimal action (sample) at (0, [2,2]):", soft_lqr.optimal_action(t_query, x_query))