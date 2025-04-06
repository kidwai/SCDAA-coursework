import numpy as np
import torch
from scipy.integrate import solve_ivp

class LQRProblem:
    def __init__(self, H, M, C, D, R, T, time_grid):
        """
        Initialize the LQR problem with given matrices (numpy or torch), 
        time horizon T, and a specified time grid.
        """
        self.T = float(T)
        self.time_grid = np.array(time_grid, dtype=float)
        self.dim = H.shape[0]
        self.H = H.cpu().numpy() if isinstance(H, torch.Tensor) else np.array(H, dtype=float)
        self.M = M.cpu().numpy() if isinstance(M, torch.Tensor) else np.array(M, dtype=float)
        self.C = C.cpu().numpy() if isinstance(C, torch.Tensor) else np.array(C, dtype=float)
        self.D = D.cpu().numpy() if isinstance(D, torch.Tensor) else np.array(D, dtype=float)
        self.R = R.cpu().numpy() if isinstance(R, torch.Tensor) else np.array(R, dtype=float)
        self.sigma = None
        self.S_t = None  # To store S(t) on the grid
        self.b_t = None  # To store b(t) on the grid

    def set_noise(self, sigma):
        """Set the diffusion matrix sigma."""
        self.sigma = sigma.cpu().numpy() if isinstance(sigma, torch.Tensor) else np.array(sigma, dtype=float)

    def solve_riccati(self):
        """Solve the Riccati ODE backward in time to compute S(t) and b(t) on the grid."""
        n = self.dim
        D_inv = np.linalg.inv(self.D)
        # Initial condition: S(T)=R, b(T)=0
        y_T = np.concatenate([self.R.reshape(-1), [0.0]])
        
        # Define the Riccati ODE system
        def riccati_ode(t, y):
            S = y[:n*n].reshape((n, n))
            b = y[n*n:]
            # Riccati ODE: dS/dt = -[Hᵀ S + S H - S M D⁻¹ Mᵀ S + C]
            dS_dt = -(self.H.T @ S + S @ self.H
                      - S @ self.M @ D_inv @ self.M.T @ S
                      + self.C)
            dS_dt_flat = dS_dt.reshape(-1)
            # b derivative: db/dt = - trace(σσᵀS)
            if self.sigma is not None:
                db_dt = - np.trace(self.sigma @ self.sigma.T @ S)
            else:
                db_dt = 0.0
            return np.concatenate([dS_dt_flat, [db_dt]])
        
        # Solve ODE from T down to 0
        sol = solve_ivp(
            riccati_ode,
            [self.T, 0],
            y_T,
            t_eval=self.time_grid[::-1],  # note we reverse the time grid
            rtol=1e-8,
            atol=1e-8
        )
        # Flip solution in time to get ascending order
        S_solution = sol.y[:-1, :].T[::-1]  # all rows except the last are S
        b_solution = sol.y[-1, :][::-1]     # the last row is b
        self.S_t = S_solution.reshape(-1, n, n)
        self.b_t = b_solution

    def value(self, t_query, x_query):
        """
        Compute the value function v*(t,x) = xᵀS(t)x + b(t) for given t and x.
        t_query: 1D torch tensor of times.
        x_query: 2D torch tensor of states.
        """
        t_np = t_query.cpu().numpy().reshape(-1)
        x_np = x_query.cpu().numpy().reshape(-1, self.dim)
        vals = []
        for tt, xx in zip(t_np, x_np):
            # Find index on the time grid
            idx = np.searchsorted(self.time_grid, tt, side='right') - 1
            idx = max(0, min(idx, len(self.time_grid)-1))
            S = self.S_t[idx]
            b = self.b_t[idx]
            v = xx @ S @ xx + b
            vals.append(v)
        return torch.tensor(vals, dtype=torch.float32)

    def optimal_action(self, t_query, x_query):
        """
        Compute the optimal control a*(t,x) = -D⁻¹MᵀS(t)x.
        t_query: 1D torch tensor of times.
        x_query: 2D torch tensor of states.
        """
        t_np = t_query.cpu().numpy().reshape(-1)
        x_np = x_query.cpu().numpy().reshape(-1, self.dim)
        actions = []
        D_inv = np.linalg.inv(self.D)
        for tt, xx in zip(t_np, x_np):
            idx = np.searchsorted(self.time_grid, tt, side='right') - 1
            idx = max(0, min(idx, len(self.time_grid)-1))
            S = self.S_t[idx]
            a_opt = - D_inv @ (self.M.T @ (S @ xx))
            actions.append(a_opt)
        return torch.tensor(actions, dtype=torch.float32)


if __name__ == '__main__':
    # Example usage for strict LQR
    H = torch.tensor([[0.5, 0.5],
                      [0.0, 0.5]])
    M = torch.tensor([[1.0, 1.0],
                      [0.0, 1.0]])
    C = torch.tensor([[1.0, 0.1],
                      [0.1, 1.0]])
    D = torch.tensor([[1.0, 0.1],
                      [0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3],
                      [0.3, 1.0]]) * 10.0
    T = 0.5
    time_grid = np.linspace(0, T, 201)

    # Create instance of LQRProblem
    lqr = LQRProblem(H, M, C, D, R, T, time_grid)

    # Set noise
    lqr.set_noise(torch.eye(2)*0.5)

    # Solve Riccati
    lqr.solve_riccati()

    # Testing at t=0 for x=[1,1], [2,2]
    t_query = torch.tensor([0.0, 0.0])
    x_query = torch.tensor([[1.0, 1.0],
                            [2.0, 2.0]])
    
    # Print value function
    val = lqr.value(t_query, x_query)
    print("Values at t=0:")
    print(f"  For x=[1,1]: {val[0].item():.4f}")
    print(f"  For x=[2,2]: {val[1].item():.4f}")

    # Print optimal action
    act = lqr.optimal_action(t_query, x_query)
    print("Optimal actions at t=0:")
    print(f"  For x=[1,1]: {act[0].numpy()}")
    print(f"  For x=[2,2]: {act[1].numpy()}")
