import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

class LQR_Solver:
    """
    A class to solve the strict LQR problem in 2 spatial and 2 control dimensions.
    
    It:
      1. Accepts the matrices specifying the LQR problem, final time T, and a time grid.
      2. Solves the Riccati ODE backwards in time:
         S'(t) = S(t) M D⁻¹ Mᵀ S(t) - Hᵀ S(t) - S H - C, with S(T)=R.
      3. Provides a method to compute the value function
         v(t, x) = xᵀ S(t)x + ∫ₜᵀ tr(σσᵀ S(r)) dr.
      4. Provides a method to compute the optimal Markov control:
         a(t,x) = -D⁻¹ Mᵀ S(t)x.
    """
    def __init__(self, H, M, sigma, C, D, R, T, time_grid):
        # All matrices are expected to be torch tensors.
        self.H = H
        self.M = M
        self.sigma = sigma
        self.C = C
        self.D = D
        self.R = R
        self.T = T
        self.time_grid = time_grid  # Assumed sorted in increasing order.
        self.device = H.device
        self.S_sol = None  # To store solution S(t) at grid points.
        self.integral = None  # To store the cumulative integral term.

    def riccati_ode(self, t, S_flat):
        # Reshape the flat vector to a 2x2 matrix.
        S = S_flat.reshape((2, 2))
        # Invert D using numpy.
        D_inv = np.linalg.inv(self.D.cpu().numpy())
        # Compute S'(t) = S M D⁻¹ Mᵀ S - Hᵀ S - S H - C.
        term1 = S @ self.M.cpu().numpy() @ D_inv @ (self.M.cpu().numpy().T) @ S
        term2 = self.H.cpu().numpy().T @ S
        term3 = S @ self.H.cpu().numpy()
        S_dot = term1 - term2 - term3 - self.C.cpu().numpy()
        return S_dot.flatten()

    def solve_riccati(self):
        # Terminal condition S(T) = R.
        S_T = self.R.cpu().numpy().flatten()
        # Solve the ODE backwards from T to 0. Note that we reverse the time grid.
        t_eval = self.time_grid.cpu().numpy()[::-1]
        sol = solve_ivp(fun=self.riccati_ode, t_span=(self.T, 0), y0=S_T, t_eval=t_eval, method='RK45')
        # Reverse the solution to get increasing time order and reshape.
        S_vals = sol.y[:, ::-1].T  # S_vals shape: (len(time_grid), 4)
        # Create a contiguous copy to avoid negative stride issues.
        S_vals = np.ascontiguousarray(S_vals.reshape((-1, 2, 2)))
        self.S_sol = torch.tensor(S_vals, dtype=self.R.dtype, device=self.device)
        
        # Precompute the cumulative integral I(t) = ∫ₜᵀ tr(σσᵀ S(r)) dr using the trapezoidal rule.
        sigma_sigmaT = self.sigma @ self.sigma.T  # 2x2 matrix.
        traces = []
        for i in range(len(self.time_grid)):
            S_i = self.S_sol[i]
            traces.append(torch.trace(sigma_sigmaT @ S_i).item())
        traces = np.array(traces)
        cum_integral = np.zeros_like(traces)
        for i in range(len(self.time_grid)):
            t_sub = self.time_grid[i:].cpu().numpy()
            y_sub = traces[i:]
            cum_integral[i] = np.trapz(y_sub, t_sub)
        self.integral = torch.tensor(cum_integral, dtype=self.R.dtype, device=self.device)
    
    def get_S_at_time(self, t_query):
        """
        For a given time t_query, return S(t) at the largest grid time ≤ t_query.
        """
        idx = (self.time_grid <= t_query).nonzero()
        if len(idx) == 0:
            i = 0
        else:
            i = idx[-1].item()
        return self.S_sol[i], i

    def compute_value(self, t, x):
        """
        Computes the value function v(t, x) = xᵀ S(t)x + ∫ₜᵀ tr(σσᵀ S(r)) dr.
        
        Parameters:
          t: 1D torch tensor of times.
          x: 2D torch tensor where each row is a state vector.
          
        Returns:
          A 1D torch tensor of values.
        """
        values = []
        for i in range(len(t)):
            t_i = t[i].item()
            S_i, idx = self.get_S_at_time(t_i)
            x_i = x[i].unsqueeze(-1)  # Convert to column vector.
            val = (x_i.t() @ S_i @ x_i).squeeze() + self.integral[idx]
            values.append(val)
        return torch.stack(values)
    
    def compute_control(self, t, x):
        """
        Computes the optimal control a(t, x) = -D⁻¹ Mᵀ S(t)x.
        
        Parameters:
          t: 1D torch tensor of times.
          x: 2D torch tensor where each row is a state vector.
          
        Returns:
          A 2D torch tensor where each row is the control vector.
        """
        controls = []
        D_inv = torch.inverse(self.D)
        for i in range(len(t)):
            t_i = t[i].item()
            S_i, _ = self.get_S_at_time(t_i)
            x_i = x[i].unsqueeze(-1)
            u_i = - D_inv @ (self.M.t() @ S_i) @ x_i
            controls.append(u_i.squeeze())
        return torch.stack(controls)


def simulate_LQR_MC(lqr_solver, t0, x0, N, n_samples, scheme="explicit"):
    """
    Simulate the LQR controlled SDE:
       dX = (H X + M a(t, X)) dt + sigma dW,
    using the optimal control a(t, x) from the solved Riccati ODE.
    
    Parameters:
      lqr_solver: Instance of LQR_Solver (with solved Riccati).
      t0: Initial time (float).
      x0: Initial state (torch tensor of shape (2,)).
      N: Number of time steps.
      n_samples: Number of Monte Carlo trajectories.
      scheme: "explicit" (implicit scheme not implemented here).
    
    Returns:
      avg_cost: Monte Carlo estimate of the cost.
      X: Simulated trajectories (tensor of shape (n_samples, N+1, 2)).
      t_grid: Time grid used for the simulation.
    """
    T = lqr_solver.T
    tau = (T - t0) / N
    t_grid = torch.linspace(t0, T, N+1, device=lqr_solver.device, dtype=lqr_solver.R.dtype)
    X = torch.zeros(n_samples, N+1, 2, device=lqr_solver.device, dtype=lqr_solver.R.dtype)
    X[:, 0, :] = x0
    # Generate Brownian increments (dW ~ N(0, tau)).
    dW = torch.randn(n_samples, N, 2, device=lqr_solver.device, dtype=lqr_solver.R.dtype) * np.sqrt(tau)
    for n in range(N):
        t_n = t_grid[n].repeat(n_samples)
        x_n = X[:, n, :]
        u_n = lqr_solver.compute_control(t_n, x_n)
        # Drift: H*x + M*u.
        drift = (lqr_solver.H @ x_n.unsqueeze(-1)).squeeze(-1) + (lqr_solver.M @ u_n.unsqueeze(-1)).squeeze(-1)
        noise = (lqr_solver.sigma @ dW[:, n, :].unsqueeze(-1)).squeeze(-1)
        X[:, n+1, :] = x_n + tau * drift + noise
    # Compute cost: running cost ∫ₜᵀ (xᵀ C x + uᵀ D u) dt plus terminal cost x(T)ᵀ R x(T).
    cost = torch.zeros(n_samples, device=lqr_solver.device, dtype=lqr_solver.R.dtype)
    for n in range(N):
        x_n = X[:, n, :]
        t_n = t_grid[n].repeat(n_samples)
        u_n = lqr_solver.compute_control(t_n, x_n)
        running_cost = (x_n.unsqueeze(1) @ lqr_solver.C @ x_n.unsqueeze(-1)).squeeze() + \
                       (u_n.unsqueeze(1) @ lqr_solver.D @ u_n.unsqueeze(-1)).squeeze()
        cost += running_cost * tau
    x_T = X[:, -1, :]
    terminal_cost = (x_T.unsqueeze(1) @ lqr_solver.R @ x_T.unsqueeze(-1)).squeeze()
    cost += terminal_cost
    avg_cost = cost.mean().item()
    return avg_cost, X, t_grid


if __name__ == "__main__":
    # ----------------------
    # Problem constants (from Figure 1)
    # ----------------------
    dtype = torch.float64
    device = "cpu"
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=dtype, device=device) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=dtype, device=device)
    sigma = torch.eye(2, dtype=dtype, device=device) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=dtype, device=device) * 1.0
    D = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=dtype, device=device) * 0.1
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=dtype, device=device) * 10.0
    T = 0.5  # Final time for the simulation experiments.
    
    # ----------------------
    # Solve the Riccati ODE (Exercise 1.1)
    # ----------------------
    time_grid = torch.linspace(0, T, 100, dtype=dtype, device=device)
    lqr_solver = LQR_Solver(H, M, sigma, C, D, R, T, time_grid)
    
    print("Solving the Riccati ODE...")
    start = time.time()
    lqr_solver.solve_riccati()
    end = time.time()
    print(f"Riccati ODE solved in {end - start:.3f} seconds.")
    
    # Test the value function and control at t = 0 for states [1,1] and [2,2].
    t_test = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
    x_test = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=dtype, device=device)
    v_vals = lqr_solver.compute_value(t_test, x_test)
    u_vals = lqr_solver.compute_control(t_test, x_test)
    print("Value function at test points:", v_vals)
    print("Optimal control at test points:", u_vals)
    
    # ----------------------
    # Monte Carlo simulation (Exercise 1.2)
    # ----------------------
    # Compute the exact value at t = 0 and x = [1,1] using the solver.
    exact_value = lqr_solver.compute_value(torch.tensor([0.0], dtype=dtype, device=device),
                                             torch.tensor([[1.0, 1.0]], dtype=dtype, device=device)).item()
    '''
    # 1. Vary the number of time steps with a fixed number of samples.
    n_time_steps_list = [2**k for k in range(1, 12)]  # e.g., 2, 4, 8, ..., 2048.
    n_samples = 10000
    errors = []
    print("\nMonte Carlo simulation: varying number of time steps")
    for N in n_time_steps_list:
        avg_cost, _, _ = simulate_LQR_MC(lqr_solver, 0.0, torch.tensor([1.0, 1.0], dtype=dtype, device=device), N, n_samples)
        error = abs(avg_cost - exact_value)
        errors.append(error)
        print(f"N = {N:4d}, Monte Carlo cost = {avg_cost:10.4f}, Exact value = {exact_value:10.4f}, Error = {error:10.4f}")
    
    plt.figure()
    plt.loglog(n_time_steps_list, errors, marker='o')
    plt.xlabel("Number of time steps")
    plt.ylabel("Error")
    plt.title("Convergence vs. Number of Time Steps")
    plt.grid(True, which="both", ls="--")
    plt.show()
    plt.savefig("time_step_convergence.png")
    '''
    # 2. Vary the number of Monte Carlo samples with a fixed large number of time steps.
    N_fixed = 10000
    sample_list = [2*4**k for k in range(0, 6)]
    errors_samples = []
    print("\nMonte Carlo simulation: varying number of samples")
    for n in sample_list:
        avg_cost, _, _ = simulate_LQR_MC(lqr_solver, 0.0, torch.tensor([1.0, 1.0], dtype=dtype, device=device), N_fixed, n)
        error = abs(avg_cost - exact_value)
        errors_samples.append(error)
        print(f"n_samples = {n:8d}, Monte Carlo cost = {avg_cost:10.4f}, Exact value = {exact_value:10.4f}, Error = {error:10.4f}")
    
    plt.figure()
    plt.loglog(sample_list, errors_samples, marker='o')
    plt.xlabel("Number of Monte Carlo Samples")
    plt.ylabel("Error")
    plt.title("Convergence vs. Number of MC Samples")
    plt.grid(True, which="both", ls="--")
    plt.show()
    plt.savefig("mc_sample_convergence.png")
