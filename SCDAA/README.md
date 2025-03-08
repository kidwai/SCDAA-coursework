# SCDAA Coursework 2024-25: Actor-Critic Implementation for LQR Problems

## Overview

This repository contains the code for solving the following exercises:
1. **Strict LQR Problem (Exercises 1.1 & 1.2):** 
   - Implementation of a class `LQRProblem` to solve the Riccati ODE.
   - Monte Carlo simulation to verify convergence.
2. **Soft LQR Problem (Exercise 2.1):**
   - Extension to incorporate entropic regularization via `SoftLQRProblem`.
3. **Critic-Only Algorithm (Exercise 3.1):**
   - A critic network (`OnlyLinearValueNN`) learns the value function.
4. **Actor-Only Algorithm (Exercise 4.1):**
   - A policy network (`PolicyNet`) learns the optimal policy using the true value function as a baseline.
5. **Actor-Critic Algorithm (Exercise 5.1):**
   - Joint learning of actor (policy) and critic (value function).

## Files

- `lqr_problem.py`: Implements the strict LQR solver.
- `soft_lqr_problem.py`: Implements the soft LQR solver.
- `monte_carlo_simulation.py`: Runs Monte Carlo simulations and plots log–log convergence graphs.
- `trajectory_plot.py`: Simulates and plots trajectories for both strict and soft LQR controllers.
- `critic_algorithm.py`: Implements the critic-only algorithm.
- `actor_algorithm.py`: Implements the actor-only algorithm.
- `actor_critic.py`: Implements the full actor-critic algorithm.
- `README.md`: This file.
- `requirements.txt`: Lists required Python packages.
- `report.md`: Detailed report on methodology, experiments, and results.

## Requirements

- Python 3.x
- numpy
- scipy
- matplotlib
- torch
