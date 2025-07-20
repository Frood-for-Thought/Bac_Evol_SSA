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
    def __init__(self, stats: MacroStats, fd_tracker: FiniteDifferenceTracker, v_d: float, lambda_var: float = 0.0):
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
