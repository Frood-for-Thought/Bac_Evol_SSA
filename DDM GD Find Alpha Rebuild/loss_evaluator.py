import torch
from typing import Optional
from macro_stats import MacroStats
from finite_difference_tracker import FiniteDifferenceTracker

"""
loss_evaluator.py

This module defines the LossEvaluator class, which computes the scalar loss L(α) and its derivative dL/dα
based on sampled macroscopic statistics and finite-difference approximations. It supports optimization
over a stochastic, non-explicit function v_j(α) where the function expectation is not known analytically
but is instead approximated through repeated sampling.

Key features:
    - Computes loss:       L(α) = (μ(α) - v_d)^2 + λ ⋅ s²(α)
    - Computes gradient:   dL/dα = 2(μ(α) - v_d) ⋅ dμ/dα + λ ⋅ ds²/dα
    - Uses:
        - MacroStats: provides μ(α), s²(α)
        - FiniteDifferenceTracker: provides dμ/dα, ds²/dα via finite differences
    - Supports confidence-weighted or variance-penalized landscape tracking
"""


class LossEvaluator:
    def __init__(self, stats: MacroStats, fd_tracker: FiniteDifferenceTracker, v_d: float, sample_func, lambda_var: float = 0.0):
        """
        Initializes the loss evaluator with references to statistical records and finite differences.
        Parameters:
            stats (MacroStats): The macro statistics tracker (contains μ, var, stderr for each α)
            fd_tracker (FiniteDifferenceTracker): Contains finite differences dμ/dα and dvar/dα
            v_d (float): Target value for the optimization objective
            lambda_var (float): Penalty coefficient for variance in the loss function
        """
        self.stats = stats
        self.fd_tracker = fd_tracker
        self.v_d = v_d
        self.lambda_var = lambda_var
        self.sample_func = sample_func  # Injected sampling function

    def loss(self, alpha: float, decimals: int = 6) -> Optional[float]:
        """
        Computes scalar loss at α:
            L(α) = (μ(α) - v_d)^2 + λ⋅ s²(α)
        Parameters:
            alpha (float): Current α
            decimals (int): Matching precision for stored values
        Returns:
            float or None: Loss value if α found, else None
        """
        α_rounded = round(float(alpha), decimals)
        match = next((r for r in self.stats.records if round(float(r["alpha"]), decimals) == α_rounded), None)
        if match is None:
            return None

        mu = match["mu"]
        var = match["var"]
        loss_val = (mu - torch.tensor(self.v_d, dtype=mu.dtype)) ** 2 + torch.tensor(self.lambda_var, dtype=var.dtype) * var
        return loss_val.item()

    def dloss_dalpha(self, alpha: float, decimals: int = 6) -> Optional[float]:
        """
        Computes the derivative of the loss function at α:
            dL/dα = 2(μ - v_d) ⋅ dμ/dα + λ ⋅ dvar/dα
        Parameters:
            alpha (float): The α at which to evaluate the gradient
            decimals (int): Matching precision for lookup
        Returns:
            float or None: Estimated loss gradient at α
        """
        α_rounded = round(float(alpha), decimals)
        match = next((r for r in self.stats.records if round(float(r["alpha"]), decimals) == α_rounded), None)
        if match is None:
            return None
        mu = match["mu"].item()
        dmu_dα = self.fd_tracker.estimate_derivative_at(alpha=alpha, kind="mu", decimals=decimals)
        dvar_dα = self.fd_tracker.estimate_derivative_at(alpha=alpha, kind="var", decimals=decimals)
        if dmu_dα is None or dvar_dα is None:
            return None  # Not enough data to compute gradient
        grad = 2 * (mu - torch.tensor(self.v_d, dtype=torch.float32)) * dmu_dα + torch.tensor(self.lambda_var, dtype=torch.float32) * dvar_dα
        return grad

    def decide_next_alpha(self,
                      alpha_k: float,
                      gamma: float,
                      n: int,
                      min_step_size: float,
                      fitness_thresh: float,
                      slope_thresh: float,
                      alpha_left: float,
                      alpha_right: float,
                      use_gradient_override: Optional[bool] = None,  # Manual override (None = automatic mode)
                      decimals: int = 6) -> Optional[float]:
        """
        Decide the next alpha (α_k+1) using a hybrid step selection strategy:
        Manual override:
            - If use_gradient_override is True:
                ▸ Take full gradient step even if it overshoots, as long as it points in the bracketed direction.
                ▸ If step would move α away from bracketed region [α_left, α_right], reverse it.
        Automatic mode:
            - If slope is collapsed (|dμ/dα| < slope_thresh) and we're far from target (loss_k > fitness_thresh):
                ▸ Likely stuck on a plateau → take a cautious RM-like probe step (α + h)
            - Else, evaluate tentative α_k+1 = α_k - γ ∂L/∂α:
                ▸ If it improves loss: accept the gradient step
                ▸ Else (gradient worsens loss): fallback to RM-like probe step (α + h)
        Inputs:
            alpha_k            : Current α
            gamma              : Learning rate multiplier
            n                  : Sample count for tentative evaluation
            min_step_size      : Minimum allowable step size
            fitness_thresh     : Threshold to consider solution “close enough”
            slope_thresh       : Collapse threshold for |dμ/dα|
            alpha_left         : Left edge of bracket (from sign transition)
            alpha_right        : Right edge of bracket (from sign transition)
            use_gradient_override : If True, always apply gradient step if direction is valid (manual override)
            decimals           : Rounding precision for alpha lookup
        Returns:
            alpha_next (float): Suggested next step (α_k+1)
        """
        # Round α to avoid floating-point issues
        alpha_k = round(float(alpha_k), decimals)

        # Current loss and gradient
        loss_k = self.loss(alpha=alpha_k)
        grad_k = self.dloss_dalpha(alpha=alpha_k)

        if loss_k is None or grad_k is None:
            raise ValueError(f"Missing loss or gradient at α = {alpha_k}")

        match = next((r for r in self.stats.records if round(float(r["alpha"]), decimals) == alpha_k), None)
        mu_k = match["mu"]
        dmu_dα = self.fd_tracker.estimate_derivative_at(alpha=alpha_k, kind="mu", decimals=decimals)
        fitness_error = torch.abs(mu_k - torch.tensor(self.v_d, dtype=mu_k.dtype))

        # Gradient step: α_next = α_k - γ * grad.
        alpha_k_tensor = torch.tensor(alpha_k, dtype=torch.float32)
        step = -gamma * grad_k
        alpha_kp1 = alpha_k_tensor + step

        if use_gradient_override is True:
            # Determine bracket direction, target_direction ∈ {+1, -1, 0}
            if alpha_k < alpha_left:
                target_direction = +1  # Must move right
            elif alpha_k > alpha_right:
                target_direction = -1  # Must move left
            else:
                target_direction = 0  # Inside bracket; direction constraint lifted

            # Check if direction of update matches intended bracket direction
            step_sign = torch.sign(alpha_kp1 - alpha_k_tensor).item()
            if target_direction != 0 and step_sign != target_direction:
                step = -step
                alpha_kp1 = alpha_k_tensor + step
            return alpha_kp1.item()
        # REVERT TO STEP SIZE ENFORCEMENT, NOT LOSS FUNCTION.
        else:
            # Evaluate temporary loss at α_k+1 (without storing permanently)
            samples_next = self.sample_func(alpha=alpha_kp1.item(), n=n)
            mu_next = samples_next.mean()
            var_next = samples_next.var(unbiased=True)
            # Compute loss at α_k+1 using same loss formula:
            #     L(α) = (μ(α) - v_d)^2 + λ ⋅ Var(α)
            # This is used to evaluate whether the gradient step was helpful.
            loss_kp1 = (mu_next - torch.tensor(self.v_d, dtype=mu_next.dtype)) ** 2 + \
                       torch.tensor(self.lambda_var, dtype=var_next.dtype) * var_next

            # If fitness_error > fitness_thresh, likely stuck at a false plateau and forward probe is justified.
            # If fitness_error < fitness_thresh, likely at convergence. Forward probe might push past the solution.
            if abs(dmu_dα) < slope_thresh and fitness_error > fitness_thresh:
                # Get probing step size.
                h = self.stats.suggest_step_size(alpha=alpha_k, v_d=self.v_d, min_h=min_step_size)
                return (alpha_k_tensor + h).item()  # Try small forward probe
            # If gradient step improved the loss → accept it.
            elif loss_kp1 < loss_k:
                return alpha_kp1.item()
            # Gradient made loss worse but slope is not collapsed → likely overshoot.
            # Fallback to a small RM-like probe step.
            else:
                h = self.stats.suggest_step_size(alpha=alpha_k, v_d=self.v_d, min_h=min_step_size)
                return (alpha_k_tensor + h).item()
