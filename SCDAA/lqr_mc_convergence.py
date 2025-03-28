import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union, Optional

###############################################################################
# Exercise 1.1: A class to solve the LQR Riccati ODE, then provide value v(t,x)
#               and optimal Markov control alpha(t,x).
###############################################################################
class LQRSolver:
    """
    This class encapsulates:
      1) Initialization with matrices for the LQR problem + time grid.
      2) A method to solve (approx.) the Riccati ODE on that grid.
      3) A method to return v(t,x) for given times t, states x.
      4) A method to return alpha(t,x) = -D^{-1} M^T S(t) x, for given t,x.
    """

    def __init__(
        self,
        H: torch.Tensor,
        M: torch.Tensor,
        sigma: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        R: torch.Tensor,
        T: float,
        time_grid: Union[np.ndarray, torch.Tensor]
    ):
        """
        :param H, M, sigma, C, D, R: torch.Tensors specifying the LQR.
        :param T: terminal time.
        :param time_grid: sorted 1D array or tensor of time points in [0,T].
        """
        self.H = H      # shape (2,2) in the typical 2d problem
        self.M = M      # shape (2,2) typically
        self.sigma = sigma  # shape (2,2') e.g. (2,2)
        self.C = C
        self.D = D
        self.R = R
        self.T = T

        # Convert time_grid to a torch tensor (descending order might help for backward solves).
        if not isinstance(time_grid, torch.Tensor):
            time_grid = torch.tensor(time_grid, dtype=torch.float32)
        self.time_grid = time_grid

        # We will store the solution to the Riccati ODE, S(t), at each grid point.
        # S(t) is a 2x2 matrix, so we store them in an array S_list of shape (n_steps, 2, 2).
        self.S_list = None

        # Also store the integral of trace( sigma sigma^T S(r) ) from r to T if needed
        # We can store partial integrals or compute on the fly.

    def solve_riccati(self):
        """
        Solve (backward) the matrix ODE:
            dS/dt = S(t) * M * D^{-1} * M^T * S(t) - H^T * S(t) - S(t)*H - C
        with terminal condition S(T) = R.
        
        We do a simple backward stepping or call an ODE solver. Here is a demo
        using a simple backward Euler or explicit dt approach for clarity.
        (One can also call scipy.integrate.solve_ivp for better accuracy.)
        """
        # Sort time_grid in ascending order, but we solve from T backward to 0
        # so we'll reverse iteration or reorder inside the solver.
        tg = self.time_grid.detach().cpu().numpy()
        # Ensure ascending order:
        if not np.all(np.diff(tg) >= 0):
            tg = np.sort(tg)
        ngrid = len(tg)
        # We'll store S in a python list, then turn into a tensor.  shape = (ngrid,2,2)
        S_array = [torch.zeros((self.H.shape[0], self.H.shape[1])) for _ in range(ngrid)]

        # Terminal condition
        S_array[-1] = self.R.clone()

        # define D^{-1} once
        Dinv = torch.inverse(self.D)

        # Step backward in time:
        for i in reversed(range(ngrid - 1)):
            dt = tg[i+1] - tg[i]  # positive step
            # We'll do a single-step Euler scheme backwards
            # ODE is dS/dt = F(S), we have S(t_{i+1}), want S(t_i).
            # So S(t_i) = S(t_{i+1}) - dt * F(S(t_{i+1})) (backward euler could be a bit more involved)
            Stp1 = S_array[i+1]
            # define F(S):
            # S(t)*M*D^{-1}*M^T*S(t) - H^T*S(t) - S(t)*H - C
            term1 = Stp1 @ self.M @ Dinv @ self.M.T @ Stp1
            term2 = - self.H.T @ Stp1 - Stp1 @ self.H - self.C
            dS = term1 + term2
            # simple explicit Euler backward:
            S_array[i] = Stp1 - dt * dS

        # Convert to a single tensor
        self.S_list = torch.stack(S_array, dim=0)  # shape = (ngrid,2,2)
        # Also store the sorted times in ascending order
        self.time_grid = torch.tensor(tg, dtype=torch.float32)

    def _nearest_S(self, t: torch.Tensor):
        """
        Find S(tn) for tn in the grid that is nearest to the given t in [0,T].
        For the exercise's instructions, they want the biggest grid time <= t
        (or nearest if you prefer). We'll do 'largest t_n <= t'.
        """
        # clamp t into [0,T]
        t_clamped = torch.clamp(t, self.time_grid[0], self.time_grid[-1])
        # We'll do a simple search.  If many t, we do it for each.
        # For clarity, assume t is scalar or 1D. We'll vectorize if needed.
        # Return an index into self.S_list, plus S(tn).
        idxs = []
        for ti in t_clamped:
            # find largest grid point <= ti
            # np.searchsorted does that; but let's do a manual torch approach:
            ind = (self.time_grid <= ti).nonzero()[-1]  # last True
            idxs.append(ind.item())
        # gather from self.S_list
        # shape will be (len(t),2,2)
        return torch.stack([self.S_list[i] for i in idxs], dim=0)

    def get_value(self, t: torch.Tensor, x: torch.Tensor):
        """
        Return v(t,x) = x^T S(t) x + integral_t^T trace(sigma sigma^T S(r)) dr
        using the precomputed S on the grid. We do a discrete approximation
        or store partial integrals. For demonstration, we'll do a simple approach.
        
        x: shape (N, d), t: shape (N,)  => returns shape (N,) for the values
        """
        # We first find S(tn). For the trace integral, we'll approximate
        #  int_{t}^{T} tr(...) dr by a sum from grid
        # The instructions say we only *need* the function value at the nearest grid t_n or so.
        # We'll do exactly that: v(t, x) ~ x^T S(t_n) x + sum_{m=n}^{end-1} tr(...) * dt
        # for the discrete times.  Then you can improve if you like.

        if x.dim() == 1:
            x = x.unsqueeze(0)  # shape => (1,dim)
            t = t.unsqueeze(0)  # shape => (1,)

        S_sel = self._nearest_S(t)  # shape (N,2,2)
        # x^T S x
        # (N,2) * (N,2,2) * (N,2) -> we can do a small loop or a batch matmul
        val_quad = []
        for i in range(x.shape[0]):
            xi = x[i].view(1, -1)             # (1,2)
            Si = S_sel[i]                    # (2,2)
            val_quad.append((xi @ Si @ xi.t()).item())
        val_quad = torch.tensor(val_quad)

        # Next approximate the integral of tr(sigma sigma^T S(r)) from r=t to T.
        # We'll do a simple sum from the index n up to the last.  Let's re-use the same index approach:
        # For each sample i, we find the index of t_i. Then sum forward:
        integral_vals = []
        sigmaSigT = self.sigma @ self.sigma.transpose(0,1)  # shape(2,2)
        for i,ti in enumerate(t):
            # find the index in the grid
            ind = (self.time_grid <= ti).nonzero()[-1].item()
            # sum from ind to end-1
            sum_ = 0.0
            for j in range(ind, len(self.time_grid)-1):
                dt = (self.time_grid[j+1] - self.time_grid[j]).item()
                S_j = self.S_list[j]
                val_ = (sigmaSigT @ S_j).trace().item()
                sum_ += val_ * dt
            integral_vals.append(sum_)
        integral_vals = torch.tensor(integral_vals)

        return val_quad + integral_vals

    def get_control(self, t: torch.Tensor, x: torch.Tensor):
        """
        Return alpha(t,x) = - D^{-1} M^T S(t) x
        For batch input (N,2) we do a vectorized operation with the same approach
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            t = t.unsqueeze(0)
        S_sel = self._nearest_S(t)  # (N,2,2)
        Dinv = torch.inverse(self.D)
        controls = []
        for i in range(x.shape[0]):
            # alpha_i = -D^{-1} M^T S_sel[i] x[i]
            tmp = -Dinv @ self.M.T @ S_sel[i] @ x[i]
            controls.append(tmp)
        controls = torch.stack(controls, dim=0)
        return controls


###############################################################################
# Exercise 1.2: Monte Carlo checks
#   - We'll do a function that, for a given number of time steps N, number of MC
#     samples M, we approximate E[ ... ] and compare with the exact v(0, x).
#   - We'll do the two log-log plots:
#       (1) Fix large #samples, vary N = 2^1,...,2^11
#       (2) Fix large N, vary #samples = 2^4,...,2^9    # this one is weird right now
###############################################################################

def simulate_explicit_euler(H, M, sigma, S_control_func, DinvMTS_func,
                            x0: np.ndarray, dt: float, N: int, seed=42):
    """
    One possible explicit scheme: 
        X_{n+1} = X_n + dt [H X_n + M alpha(t_n,X_n)] + sigma * dW_n
    alpha(t_n, X_n) = -D^{-1} M^T S(t_n) X_n
    We'll pass in a function S_control_func(t, x) returning S(t) x so we can do alpha quickly.
    
    Alternatively, we can pass the entire alpha(t,x). We'll do alpha(t_n,x_n).
    """
    np.random.seed(seed)
    d = x0.shape[0]
    X = np.zeros((N+1, d))
    X[0,:] = x0
    # Brownian increments
    for n in range(N):
        tn = n*dt
        xn = X[n,:]
        # alpha(tn, xn) = - D^{-1} M^T S(tn) xn
        alpha_n = DinvMTS_func(tn, xn)  # or S_control_func + formula
        drift = (H @ xn) + (M @ alpha_n)
        dW = np.sqrt(dt)*np.random.randn(sigma.shape[1])
        X[n+1,:] = xn + dt*drift + (sigma.detach().cpu().numpy() @ dW)
    return X

def main():
    # Matrices from the example in the SCDAA doc:
    # or as in the problem statement, e.g. (2x2)
    H = torch.tensor([[1.0, 1.0],
                      [0.0, 1.0]]) * 0.5
    M = torch.tensor([[1.0, 1.0],
                      [0.0, 1.0]])
    sigma = torch.eye(2)*0.5
    C = torch.tensor([[1.0, 0.1],
                      [0.1, 1.0]]) * 1.0
    D = torch.tensor([[1.0, 0.1],
                      [0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3],
                      [0.3, 1.0]]) * 10.0
    T = 0.5

    # Create a time grid for the solver for Exercise 1.1:
    time_grid = np.linspace(0.0, T, 200)  # for instance 200 points

    # 1) Solve the Riccati ODE and build the LQR solver object
    lqr = LQRSolver(H, M, sigma, C, D, R, T, time_grid)
    lqr.solve_riccati()

    # Check v(0, x) for a couple of points x = (1,1) or x=(2,2)
    xA = torch.tensor([1.0,1.0])
    valA = lqr.get_value(torch.tensor([0.0]), xA)
    print("Value v(0,(1,1)) = ", valA.item())
    xB = torch.tensor([2.0,2.0])
    valB = lqr.get_value(torch.tensor([0.0]), xB)
    print("Value v(0,(2,2)) = ", valB.item())

    # 2) Monte Carlo checks:
    # Part 1: fix #samples large, vary N in powers of two => log-log error plot
    # We define a function to do MC for a single N, return average cost
    def mc_estimate_value_explicit(x0, N, nSamples, seed=1234):
        """
        For each sample we do an explicit Euler scheme with N steps of size dt=T/N,
        we use the optimal control alpha(t,x). Then we compute the realized cost
           int_0^T [X_s^T C X_s + alpha_s^T D alpha_s ] ds + X_T^T R X_T
        using e.g. trapezoidal or left Riemann sum in discrete time.
        Then average over samples => MC estimate of v(0,x0).
        """
        dt = T/N
        rng = np.random.RandomState(seed)

        # We can precompute -D^{-1}M^T * S(t) for each t if we want,
        # but let's just define a helper that calls the LQR get_control
        # but we need it in numpy. So let's do a small wrapper:
        def alpha_opt(t, x):
            # returns alpha(t,x) as numpy array
            tt = torch.tensor([t], dtype=torch.float32)
            xx = torch.tensor(x, dtype=torch.float32)
            ctrl = lqr.get_control(tt, xx)[0]
            return ctrl.detach().cpu().numpy()

        # We'll accumulate costs
        total_costs = []
        for sample_i in range(nSamples):
            # simulate
            x_path = np.zeros((N+1, 2))
            x_path[0,:] = x0.detach().cpu().numpy()
            cost_path = 0.0
            for n in range(N):
                tn = n*dt
                xn = x_path[n,:]
                alpha_n = alpha_opt(tn, xn)
                # immediate cost ~ X_n^T C X_n + alpha_n^T D alpha_n
                cost_inst = xn @ C.detach().cpu().numpy() @ xn + alpha_n @ D.detach().cpu().numpy() @ alpha_n
                cost_path += cost_inst*dt
                # do one euler step
                drift = (H.detach().cpu().numpy() @ xn) + (M.detach().cpu().numpy() @ alpha_n)
                dW = np.sqrt(dt)*rng.randn(sigma.shape[1])
                x_next = xn + dt*drift + (sigma.detach().cpu().numpy() @ dW)
                x_path[n+1,:] = x_next
            # terminal cost
            xT = x_path[-1,:]
            cost_term = xT @ R.detach().cpu().numpy() @ xT
            cost_total = cost_path + cost_term
            total_costs.append(cost_total)
        return np.mean(total_costs)

    # (a) fix nSamples=10000, vary N = 2^1..2^9 or 2^10, do log-log of error
    nSamples_fixed = 10000
    Ns_list = [2**k for k in range(1, 12)]  # e.g. up to 2^11
    x0A = torch.tensor([1.0, 1.0], dtype=torch.float32)
    true_valA = lqr.get_value(torch.tensor([0.0]), x0A).item()
    errs_vs_N = []
    for N_ in Ns_list:
        est_val = mc_estimate_value_explicit(x0A, N_, nSamples_fixed, seed=1234)
        err = abs(est_val - true_valA)
        errs_vs_N.append(err)
        print(f"N={N_}, MC val= {est_val:.5f}, true= {true_valA:.5f}, err= {err:.4e}")

    plt.figure()
    plt.title("Error vs N (log-log) for fixed #samples=10000, x=(1,1)")
    plt.loglog(Ns_list, errs_vs_N, 'o--', label="MC error")
    plt.xlabel("N (number of time steps)")
    plt.ylabel("Absolute error")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig("exercise1_2_plot_part1.png", dpi=150)
    # plt.close()

    # this one is weird right now
    # (b) fix a large N, vary #samples => log-log
    # fix N=2000 or so
    N_fixed = 2000
    samples_list = [2**(4 + k) for k in range(6)]  # 2^4..2^9
    errs_vs_M = []
    for M_ in samples_list:
        est_val = mc_estimate_value_explicit(x0A, N_fixed, M_, seed=999)
        err = abs(est_val - true_valA)
        errs_vs_M.append(err)
        print(f"Samples={M_}, MC val= {est_val:.5f}, true= {true_valA:.5f}, err= {err:.4e}")

    plt.figure()
    plt.title(f"Error vs #samples (log-log), N={N_fixed}, x=(1,1)")
    plt.loglog(samples_list, errs_vs_M, 'o--', label="MC error")
    plt.xlabel("Number of Monte Carlo samples")
    plt.ylabel("Absolute error")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig("exercise1_2_plot_part2.png", dpi=150)
    # plt.close()


if __name__ == "__main__":
    main()
