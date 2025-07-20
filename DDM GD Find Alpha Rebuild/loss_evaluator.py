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

    def decide_next_alpha(self,
                          alpha_k: float,
                          h: float,
                          slope_tol: float = 1e-3,
                          fitness_tol: float = 0.1,
                          gamma: float = 1.0,
                          clip_max_step: float = 1.0,
                          decimals: int = 6) -> Optional[float]:
        """
        Decide next α using hybrid criteria:
          - If slope ≈ 0 and far from target → try α + h
          - If slope contradicts fitness direction → step forward
          - Else → gradient step via ∂L/∂α
        Parameters:
            alpha_k (float): Current α.
            h (float): Step size (usually from suggest_step_size).
            slope_tol (float): Threshold for detecting a flat slope.
            fitness_tol (float): When to consider mu far from v_d.
            gamma (float): Learning rate for gradient.
            clip_max_step (float): Max allowed α step.
            decimals (int): Matching precision for α lookup.
        Returns:
            alpha_next (float): Updated α_k+1
        """
        α_k_rounded = round(float(alpha_k), decimals)
        α_kp1 = round(float(alpha_k + h), decimals)

        rec_k = next((r for r in self.stats.records if round(float(r["alpha"]), decimals) == α_k_rounded), None)
        rec_kp1 = next((r for r in self.stats.records if round(float(r["alpha"]), decimals) == α_kp1), None)

        if rec_k is None or rec_kp1 is None:
            print(f"Missing macro records at α = {α_k_rounded} or α + h = {α_kp1}")
            return None

        mu_k = rec_k["mu"].item()
        mu_kp1 = rec_kp1["mu"].item()
        fitness_k = abs(mu_k - self.v_d)
        fitness_kp1 = abs(mu_kp1 - self.v_d)

        dmu_dα = self.fd_tracker.estimate_derivative_at(alpha=alpha_k, kind="mu", decimals=decimals)
        dvar_dα = self.fd_tracker.estimate_derivative_at(alpha=alpha_k, kind="var", decimals=decimals)

        if dmu_dα is None or dvar_dα is None:
            print(f"Missing finite differences at α = {alpha_k}")
            return None

        dL_dα = self.dloss_dalpha(alpha=alpha_k, decimals=decimals)
        if dL_dα is None:
            print(f"Missing ∂L/∂α at α = {alpha_k}")
            return None

        print(f"dμ/dα = {dmu_dα:.6f}, |μ(α) - v_d| = {fitness_k:.6f}")
        print(f"μ(α_k+1) = {mu_kp1:.6f}, fitness_k+1 = {fitness_kp1:.6f}")
        print(f"∂L/∂α = {dL_dα:.6f}")

        # Rule 1: Slope is too flat and alpha_k is far from target.
        if abs(dmu_dα) < slope_tol and fitness_k > fitness_tol:
            print("Slope too small and far from target → stepping forward")
            return alpha_k + h

        # Rule 2: Slope points in wrong way of fitness direction → override
        slope_sign = torch.sign(torch.tensor(dmu_dα)).item()
        fitness_sign = torch.sign(torch.tensor(mu_k - mu_kp1)).item()
        if slope_sign != fitness_sign:
            print("Slope contradicts fitness → overriding gradient")
            return alpha_k + h

        # Rule 3: Normal gradient step
        step = -gamma * dL_dα
        print(f"🔷 Gradient step → α_k+1 = α_k + {step:.6f}")
        return alpha_k + step
