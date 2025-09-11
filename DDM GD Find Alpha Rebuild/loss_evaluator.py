import torch
import math
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
        if dmu_dα is None or not math.isfinite(float(dmu_dα)):
            return None  # Not enough data to compute gradient
        if (dvar_dα is None) or (not math.isfinite(float(dvar_dα))):
            dvar_dα = 0.0

        # Prefer PR slope m_k once the linear gate AND PR has stabilized
        mk = self.fd_tracker.m_k
        if (mk is not None) and self.fd_tracker.linear_slope_ready and math.isfinite(float(mk)):
            # Residual gate
            n_k = int(match.get("n", 0))
            resid_ok, _, _ = self.fd_tracker.residual_gate(
                alpha=match["alpha"], mu_val=match["mu"], var_val=match["var"], n=n_k
            )
            if bool(resid_ok):
                dmu_dα = float(mk)  # <-- USE m_k AS THE SLOPE

        grad = 2 * (mu - torch.tensor(self.v_d, dtype=torch.float32)) * dmu_dα + torch.tensor(self.lambda_var, dtype=torch.float32) * dvar_dα
        return grad

    def ensure_record(self, alpha: float, n: int, decimals: int = 6):
        """
        Ensure that a MacroStats record exists at the given α (rounded to `decimals`).

        If a record at α is already present in `self.stats.records`, it is returned.
        Otherwise this method:
          1) draws `n` samples at α using `self.sample_func`,
          2) logs them via `self.stats.macro_observations(alpha, samples)`, and
          3) recomputes finite differences with `self.fd_tracker.compute_all_differences(self.stats)`,
        then looks up and returns the (now) existing record.

        :param alpha: The α value whose macro statistics are required.
        :param n: Number of samples to draw if a new record must be created.
        :param decimals: Rounding precision used to match/insert α in the records table (default: 6).
        :return: The record dict for α (e.g., with keys 'alpha', 'mu', 'var', ...), or None if creation failed.
        """
        alpha = round(float(alpha), decimals)
        rec = next((r for r in self.stats.records
                    if round(float(r["alpha"]), decimals) == alpha), None)
        if rec is None:
            samples = self.sample_func(alpha=alpha, n=n)
            self.stats.macro_observations(alpha=alpha, samples=samples)
            self.fd_tracker.compute_all_differences(self.stats)
            rec = next((r for r in self.stats.records
                        if round(float(r["alpha"]), decimals) == alpha), None)
        return rec

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
        # Ensure data is at α_k
        rec_k = self.ensure_record(alpha_k, n=n, decimals=decimals)

        # Try normal loss/grad
        loss_k = self.loss(alpha=alpha_k)
        grad_k = self.dloss_dalpha(alpha=alpha_k)
        # FAIL FAST on bad gradient/loss.
        if grad_k is None or not math.isfinite(float(grad_k)):
            raise RuntimeError(
                f"[GD-ERROR] Non-finite or missing gradient at α={alpha_k}. "
                f"loss_k={loss_k}, grad_k={grad_k}"
            )
        if loss_k is None or not math.isfinite(float(loss_k)):
            raise RuntimeError(
                f"[GD-ERROR] Non-finite or missing loss at α={alpha_k}. "
                f"loss_k={loss_k}, grad_k={grad_k}"
            )

        # If missing/bad, rebuild loss from rec_k
        if (loss_k is None) or (not math.isfinite(loss_k)):
            mu_k, var_k = rec_k["mu"], rec_k["var"]
            loss_k = (mu_k - torch.tensor(self.v_d, dtype=mu_k.dtype))**2 + \
                     torch.tensor(self.lambda_var, dtype=var_k.dtype) * var_k

        match = next((r for r in self.stats.records if round(float(r["alpha"]), decimals) == alpha_k), None)
        mu_k = match["mu"]
        dmu_dα = self.fd_tracker.estimate_derivative_at(alpha=alpha_k, kind="mu", decimals=decimals)
        fitness_error = torch.abs(mu_k - torch.tensor(self.v_d, dtype=mu_k.dtype))

        # Gradient step: α_next = α_k - γ * grad.
        alpha_k_tensor = torch.tensor(alpha_k, dtype=torch.float32)
        # -----------------------------------------------------------------------------
        # THIS IS NOW CALCULATED IN Calc_Alpha_ML_Function
        # Adaptive gamma cap from linear analysis:
        # (alpha_{k+1} - alpha_star) = (1 - 2 * gamma * m_k^2) * (alpha_k - alpha_star).
        # Convergence requires 0 < gamma < 1 / m_k^2. To avoid oscillations and blow-ups,
        # cap gamma at (1 - eta) / (2 * m_k^2), with a tiny margin eta in (0, 0.1].
        # This keeps |1 - 2 * gamma * m_k^2| < 1 and avoids the huge jumps away from convergence.
        # mk = self.fd_tracker.m_k
        # So the only ways that scaling would be “not used” in practice are:
        # The guard set dmu_dalpha = 1.0 (e.g., slope missing/tiny/non-finite), which makes gamma = 1.0.
        # The gamma cap is enabled and use_gradient_override=False, so LossEvaluator reduces gamma.
        # if (mk is not None) and self.fd_tracker.linear_slope_ready:
        #     mk_val = float(mk.item() if hasattr(mk, "item") else mk)
        #     mk2 = mk_val * mk_val
        #     if mk2 > 0.0 and math.isfinite(mk2):
        #         eta = 0.05  # small safety margin
        #         gamma_cap = (1.0 - eta) / (2.0 * mk2)
        #         if gamma > gamma_cap:
        #             gamma = gamma_cap

        step = -gamma * grad_k
        alpha_kp1 = alpha_k_tensor + step

        # FAIL FAST on non-finite step/α_{k+1}
        mk = self.fd_tracker.m_k
        mk_val = (float(mk.item()) if (mk is not None and hasattr(mk, "item")) else
                  (float(mk) if mk is not None else None))
        if not math.isfinite(float(step)) or not math.isfinite(float(alpha_kp1)):
            raise RuntimeError(
                "[GD-ERROR] Non-finite update detected.\n"
                f"  alpha_k={alpha_k}\n"
                f"  gamma={gamma}\n"
                f"  grad_k={grad_k}\n"
                f"  step={step}\n"
                f"  alpha_kp1={alpha_kp1}\n"
                f"  m_k={mk_val}"
            )

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

            # Override-only safety: if α_{k+1} is non-finite, snap to bracket edge in the step direction.
            if not torch.isfinite(alpha_kp1):
                br_left  = float(min(alpha_left, alpha_right))
                br_right = float(max(alpha_left, alpha_right))
                eps = torch.finfo(alpha_kp1.dtype).eps  # tiny nudge to stay inside
                # choose edge consistent with intended direction (or actual step sign if inside bracket)
                dir_sign = target_direction if target_direction != 0 else step_sign
                if dir_sign > 0:
                    return br_right - eps
                elif dir_sign < 0:
                    return br_left + eps
                else:
                    # zero/undefined direction: no-op to avoid NaN
                    return alpha_k_tensor.item()

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
