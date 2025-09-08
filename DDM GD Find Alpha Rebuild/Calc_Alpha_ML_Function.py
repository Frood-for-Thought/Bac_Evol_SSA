# Import the Tumble Angle Module
import torch
import math
import logging
from abc import ABC, abstractmethod
import logging
from loss_evaluator import LossEvaluator  # L(α), dL/dα, and decide_next_alpha


# Abstract Base Class for Data Generators
class BaseDataGenerator(ABC):
    @abstractmethod
    def generate_data(self, alpha, max_iter):
        pass

    def sample(self, alpha, n):
        return self.generate_data(alpha, n)


class Dynamic_Data_Evolving_Mean_Estimator:
    """
    Dynamic_Data_Evolving_Mean_Estimator, pronounced 'deme'.

    Instead of finding the mean of clusters, this algorithm generates dynamically changing data for each iteration of
    training using a stochastic function with an independent variable alpha.  This in turn produces a stochastic mean
    which is then compared to an optimal theoretical target value.  Gradient descent guides the ML algorithm to adjust
    the alpha accordingly so the dependent variable of the stochastic function will have an evolving mean that becomes
    more aligned with the theoretical value.

    The intrinsic limitation of dynamically generating normalized data are their mean's standard error.
    Therefore, the ML algorithm iteratively refines the learning_rate and max_iter, (no. data points generated),
    every number of step_size iterations in order to reduce the standard error on two fronts.  The learning_rate is
    reduced by learning_rate_gamma and the max_iter is increased by max_iter_factor so that the standard error is
    decreased by σ⁄√(max_iter_factor).  These hyperparameters will have to be balanced with the total number of
    epoch iterations in order to best minimize the standard error. The process is similar to how simulated annealing
    tries to escape a local minima, however, in this case the step_size schedules an adjustment of hyperparameters to
    minimize the intrinsic standard error.

    :param: data_generator: Has to follow the same format of the BaseDataGenerator to make sure that the object used
    has a generate_data() method.
    :param: max_iter: The number of initial training iterations (data points generated) used by the ML model.
    The generate_data() method must have max_iter as an argument.
    :param: max_iter_limit: The upper limit for max_iter to prevent it from growing indefinitely.
    :param: max_iter_factor: The factor by which max_iter is multiplied each step (e.g., 2 to double).
    :param: learning_rate: The learning rate for the ML algorithm during gradient descent.
    :param: learning_rate_gamma: The factor by which the learning rate is multiplied each step_size.
    learning_rate_gamma = 0.85 gives ~ 1/2, 1/3, 1/4.
    :param: step_size: The number of iteration the training loop goes through before it adjusts the parameters max_iter
    and learning_rate.
    :param: num_epochs: The total number of G.D. training iterations used by the ML algorithm.
    :param: theoretical_val: The target the model is trying to predict using the normalized data mean produced by
    generate_data().
    :param: alpha: The independent variable the ML model is optimizing for a stochastic function whose mean
    """
    def __init__(self, data_generator: BaseDataGenerator, num_epochs, learning_rate, theoretical_val,
                 alpha, max_iter, eta_for_gamma_cap=0.05, step_size=20, max_iter_limit=20000, max_iter_factor=2,
                 learning_rate_gamma=0.7, stats=None, fd_tracker=None, bracket=None, use_gradient_override=True):

        self.data_generator = data_generator  # class BaseDataGenerator(ABC)
        self.max_iter = max_iter
        self.max_iter_limit = max_iter_limit
        self.max_iter_factor = max_iter_factor
        # learning_rate = a global scale set once (and decay with the scheduler).
        # Not to be confused with gamma = the actual per-epoch step size, computed from the current data and then
        # scaled by the current learning_rate, and possibly capped for stability.
        self.learning_rate = learning_rate
        self.theoretical_val = theoretical_val  # theoretical_val is also incorporated into LossEvaluator.
        # Safety margin for the gamma cap derived from m_k (keeps |1-2*gamma*m_k^2| < 1)
        self.eta_for_gamma_cap = eta_for_gamma_cap
        self.learning_rate_gamma = learning_rate_gamma
        self.step_size = step_size
        self.num_epochs = num_epochs

        # Enable Polyak–Ruppert slope estimation over the first bracket; start fresh so slope_history is clean.
        # When you call run_linear_estimation(..., enabled=True, ...), it sets linear_mode_enabled = True and
        # runs the Polyak–Ruppert (PR) slope update.
        # Enabled=False turns off linear mode (no PR update runs on that call).
        # Enabled=None leaves the previous toggle as-is.
        # Window=[α_left=a_min, α_right=a_max] sets the active interval for PR estimation,
        # whereby new estimations are only inside that window.
        # reset=True clears prior PR state, (slope_history, m_k, ᾱ, μ̄, readiness flags), start fresh in the new window
        self.stats = stats
        self.fd_tracker = fd_tracker
        self.bracket = bracket
        if (self.stats is not None) and (self.fd_tracker is not None) and (self.bracket is not None):
            a_min, a_max = map(float, self.bracket)
            # Guarding for two points avoids a zero-denominator slope and premature “readiness” noise.
            if a_min > a_max: # In case the values are reversed.
                a_min, a_max = a_max, a_min
            in_window = [
                r for r in self.stats.records
                if (a_min - 1e-6) <= float(r["alpha"].item() if hasattr(r["alpha"], "item") else r["alpha"]) <= (
                            a_max + 1e-6)
            ]
            if len(in_window) >= 2:
                self.fd_tracker.run_linear_estimation(self.stats, enabled=True, window=[a_min, a_max], reset=True)

        # Importing loss functions.
        self.loss_eval = None  # Inactive by default.
        if (self.stats is not None) and (self.fd_tracker is not None):
            # Initialize class from loss_evaluator.py.
            self.loss_eval = LossEvaluator(
                stats=self.stats,
                fd_tracker=self.fd_tracker,
                v_d=float(theoretical_val),  # v_d is the theoretical target.
                lambda_var=0.0,  # Penalty coefficient for variance in the loss function in loss_evaluator.py.
                # LossEvaluator just stores the callable,
                # return self.generate_data(alpha, n) are only provided later when LossEvaluator needs samples.
                sample_func=self.data_generator.sample
            )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Convert the alpha integer to a tensor to be optimized.
        alpha_value = float(alpha)  # Convert to float first.
        self.alpha = torch.tensor(alpha_value, requires_grad=True, dtype=torch.float32, device=self.device)
        # use_gradient_override a class switch in alpha_next = self.loss_eval.decide_next_alpha() in training.
        self.use_gradient_override = bool(use_gradient_override)

        # Initialize the optimizer with the model parameters and learning rate.
        # This module DOES NOT optimize α with autograd.
        # α is updated by LossEvaluator.decide_next_alpha(...) (the evaluator-driven step).
        # It still keeps a tiny optimizer + scheduler to reuse PyTorch’s LR decay logic as a scalar multiplier
        # for gamma (γ). It only reads its LR each epoch:
        #     current_lr_scale = optimizer.param_groups[0]['lr']
        # The optimizer will handle the update of alpha based on the computed gradients.
        self.optimizer = torch.optim.SGD([self.alpha], lr=self.learning_rate)

        # The learning rate scheduler will reduce the learning rate by learning_rate_reduction every step_size epochs.
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.step_size,
            gamma=self.learning_rate_gamma
        )

    def train(self):
        final_loss = None  # Store the final loss to return
        for epoch in range(self.num_epochs):
            # Ensure MacroStats has a record at current α_k and recompute FDs as needed.
            alpha_k = float(self.alpha.detach().item())
            if self.loss_eval is None:
                raise RuntimeError("LossEvaluator not initialized (stats/fd_tracker missing).")
            # Ensure αlpha_k is recorded, observe and update finite differences.
            # Draw n=self.max_iter samples using data_generator, logs MacroStats,
            # and recomputes finite differences internally.
            # Within ensure_record "samples = self.sample_func(alpha=alpha, n=n)"
            #   --> sample_func=self.data_generator.sample --> data_generator: MacroStats.
            # MacroStats generates new epoch of data for every instance of the training loop, sample = 1/n * ∑vj(α)
            # The data point generator is external to PyTorch's computational graph and PyTorch cannot connect
            # alpha using the chain rule because for this model ∂vj(α)/∂α is unknown.
            rec_k = self.loss_eval.ensure_record(alpha=alpha_k, n=self.max_iter)
            # Call path:
            # self.loss_eval.ensure_record(...)
            # → self.loss_eval.sample_func(alpha, n)
            # → self.data_generator.sample(alpha, n) (from BaseDataGenerator)
            # → self.data_generator.generate_data(alpha, n)
            if rec_k is None:
                raise RuntimeError(f"Failed to record macro stats at α={alpha_k}.")

            # Using residual_gate in finite_difference_tracker.py:
            # Centered-residual gate (LOG ONLY; does not change behavior) and compare it against a stderr-based bound.
            #         Returns (resid_ok, r_cent, bound) for the centered-residual gate:
            #             |r_cent(α)| ≤ stderr_tol * (2 * s(α) / sqrt(n))
            #         where r_cent(α) = (μ(α) - μ̄) - m_k * (α - ᾱ).
            # Small centered residuals indicate the local linearity required by the γ-cap analysis.
            # Pull the μ, var at the *current* alpha_k (we just ensured it exists).
            # declare “linear-agreement OK” if ∣r_cent(α)∣ ≤ stderr_tol⋅( 2s(α)/sqrt(n) )
            resid_ok, r_cent, bound = self.fd_tracker.residual_gate(
                alpha=rec_k["alpha"],  # use the recorded α directly
                mu_val=rec_k["mu"],
                var_val=rec_k["var"],
                n=self.max_iter,
            )

            # Loss and gradient now come from LossEvaluator (μ, var, finite-difference dL/dα),
            # no linear layer or proxy scaling needed.
            # Computing the loss of the dynamic data point mean compared to the theoretical_val (now via LossEvaluator).
            # L(α) = (μ(α) − v_d)^2 + λ·var(α)
            loss_val = self.loss_eval.loss(alpha=alpha_k)
            if (loss_val is None) or (not torch.isfinite(torch.tensor(loss_val))):
                raise RuntimeError(f"Loss unavailable/non-finite at α={alpha_k}.")
            # Keep track of the latest loss for the return value
            final_loss = torch.tensor(float(loss_val), dtype=self.alpha.dtype, device=self.alpha.device)

            # Update PR slope each iteration (no reset), then report readiness.
            self.fd_tracker.run_linear_estimation(self.stats, enabled=True, window=None, reset=False)
            # -----------------------------------------------------------------------------
            # Clarification of the “linear recursion” and the bounds:
            #
            # 1) Define the tracking error:
            #      e_k := alpha_k - alpha_star
            #    Using the linear model above, the gradient near alpha_star is:
            #      dL/dalpha |_alpha_k  ~=  2 * m_k^2 * e_k
            #
            # 2) One GD step:
            #      alpha_{k+1} = alpha_k - gamma * (2 * m_k^2 * e_k)
            #    Subtract alpha_star from both sides:
            #      e_{k+1} = (alpha_{k+1} - alpha_star)
            #              = (alpha_k - alpha_star) - 2 * gamma * m_k^2 * e_k
            #              = (1 - 2 * gamma * m_k^2) * e_k
            #    This is the linear recursion with multiplier q := (1 - 2 * gamma * m_k^2).
            #
            #    Intuition: the recursion says each new error e_{k+1} is just the old
            #    error e_k multiplied by a constant factor q. So the whole behavior
            #    depends on |q|:
            #       - if |q| < 1 → errors shrink
            #       - if |q| = 1 → errors persist
            #       - if |q| > 1 → errors grow
            #
            # 3) What “convergence” means here:
            #      We need |e_{k+1}| < |e_k| for errors to shrink.
            #      That requires |q| < 1  <=>  |1 - 2 * gamma * m_k^2| < 1.
            #      Solving gives:  0 < gamma < 1 / m_k^2.
            #
            #    Detailed steps:
            #      - Start:  |1 - 2 * gamma * m_k^2| < 1
            #      - Equivalent to: -1 < 1 - 2 * gamma * m_k^2 < 1
            #      - Left inequality:  -1 < 1 - 2γm_k^2  ⇒  -2 < -2γm_k^2  ⇒  γ < 1/m_k^2
            #      - Right inequality: 1 - 2γm_k^2 < 1   ⇒  -2γm_k^2 < 0   ⇒  γ > 0
            #      - Combined: 0 < γ < 1/m_k^2
            #
            # 4) Behavior by gamma range (assume m_k^2 > 0; the square handles m_k < 0 too):
            #      - 0 < gamma < 1/(2 m_k^2):       q in (0, 1)  → monotone convergence (no sign flips)
            #      - gamma = 1/(2 m_k^2):           q = 0        → one-step to alpha_star in ideal linear/noiseless case
            #      - 1/(2 m_k^2) < gamma < 1/m_k^2: q in (-1, 0) → convergent but oscillatory (e_k flips sign each step)
            #      - gamma = 1/m_k^2:               q = -1       → no contraction; persistent large oscillation
            #      - gamma > 1/m_k^2:               |q| > 1      → divergence (errors grow)
            #
            # 5) Why this matches the intuition:
            #      - When gamma is “too big” relative to the local curvature scale m_k^2,
            #        the step overshoots, flips the sign, and if |q| >= 1 the amplitude does not decay
            #        (oscillates or explodes).
            #      - Setting gamma ~ 1 normalizes by nothing; if m_k ~ 1 (common in linear patches),
            #        then q ~ 1 - 2*1*1 = -1, i.e., the problematic oscillation factor.
            #      - The square m_k^2 is why the condition depends only on the *magnitude* of slope,
            #        not its sign — negative slopes behave the same.
            #
            # 6) Effect of the variance term (lambda > 0):
            #      - The gradient gains + lambda * d(s^2)/dalpha. If that term is small near the target
            #        or comparatively flat, the m_k^2-driven analysis dominates. If it is not small,
            #        it perturbs q slightly; the same form still holds locally with m_k replaced by the
            #        effective local slope factor of the full gradient.
            # -----------------------------------------------------------------------------
            m_k = self.fd_tracker.m_k
            # self.fd_tracker.last_delta_m = Abs diff between last two m_k’s within slope_history,|Δm|,(None initially)

            # Estimate dμ/dα externally
            dmu_dalpha = self.fd_tracker.estimate_derivative_at(alpha=alpha_k, kind="mu")
            # Prevent blow-ups from dμ/dα
            if (dmu_dalpha is None) or (not math.isfinite(dmu_dalpha)) or (abs(dmu_dalpha) < 1e-8):
                dmu_dalpha = 1.0  # safe default scale
            # Slope-normalized step scaling:
            # dmu_dalpha ≈ local sensitivity μ'(α_k). We set γ = 1 / |μ'| so that the raw GD step.
            #   Δα = −γ · dL/dα  ≈  −(1/|μ'|) · 2(μ − v_d) · μ'  =  −2 · (μ − v_d) · sign(μ').
            # The step size depends on the error (μ − v_d) but is ≈ invariant to the local slope magnitude.
            # This avoids huge Δα on steep regions and vanishing Δα on flat regions.
            # Any stability/convergence cap on γ is applied inside LossEvaluator.decide_next_alpha().
            gamma = 1.0 / abs(dmu_dalpha)  # scale only; LossEvaluator handles the safety cap.
            # Multiply by scheduler-controlled scale (decays every 'step_size_epochs').
            # Scale gamma by the decayed LR from the dummy optimizer.
            current_lr_scale = self.optimizer.param_groups[0]['lr']
            gamma *= float(current_lr_scale)

            # Adaptive stability cap on gamma (ONLY when PR slope is ready).
            # From linear analysis: convergence needs |1 - 2*gamma*m_k^2| < 1 ⇒ gamma < 1/m_k^2.
            # Best non-oscillatory contraction at gamma = 1/(2*m_k^2).
            if self.fd_tracker.linear_slope_ready and (m_k is not None):
                mk_val = float(m_k.item()) if hasattr(m_k, "item") else float(m_k)
                if math.isfinite(mk_val):
                    mk2 = mk_val * mk_val
                    if mk2 > 0.0:
                        gamma_cap = (1.0 - self.eta_for_gamma_cap) / (2.0 * mk2)
                        # Fail fast if cap is nonsensical
                        if not math.isfinite(gamma_cap) or gamma_cap <= 0.0:
                            raise RuntimeError(
                                f"[GD-ERROR] Bad gamma_cap from m_k: m_k={mk_val}, gamma_cap={gamma_cap}"
                            )
                        # Cap gamma (keep original scaling but make it safe).
                        if gamma > gamma_cap:
                            gamma = gamma_cap

            if self.bracket is None:
                raise RuntimeError("Bracket (α_min, α_max) not set.")
            a_min, a_max = map(float, self.bracket)

            alpha_next = self.loss_eval.decide_next_alpha(
                alpha_k=alpha_k,
                n=self.max_iter,
                gamma=gamma,
                min_step_size=0.1,
                fitness_thresh=0.1,
                slope_thresh=1e-3,
                alpha_left=a_min,
                alpha_right=a_max,
                use_gradient_override=self.use_gradient_override
            )

            # This does NOT optimize `alpha` via autograd/optimizer in this module.
            # # Alpha is updated explicitly by LossEvaluator.decide_next_alpha(...).
            # (self.optimizer.zero_grad()) Reset previous gradient to prevent incorrect update.

            # Compute the gradient of the loss function with respect to the parameters with requires_grad=True,
            # in this case the alpha value.  The function the model hopes to optimize, vj(α), is quite complex
            # and non-differentiable by PyTorch, so loss.backward() cannot be used because
            # within ∂L(α)/∂α = (2/n)∑(vj(α)−vd) * ∂vj(α)/∂α, ∂vj(α)/∂α is unknown.
            # Since the function is not differentiable w.r.t. alpha, loss.backward() cannot compute the true gradient.
            # Apply the update (without optimizer.step) because α is updated externally by
            # LossEvaluator.decide_next_alpha(...). The optimizer tracks a dummy parameter
            # to reuse PyTorch’s LR scheduler as a scalar decay for γ --> # α_k_+_1 = α_k - γ*∂L(α)/∂α.
            with torch.no_grad():
                self.alpha.fill_(float(alpha_next))  # update α directly

            # Print out the gradient of alpha after backpropagation.
            logging.info(f"\nEpoch: {epoch}")
            logging.info(f"alpha_k = {alpha_k:.6f}  →  alpha_next = {float(alpha_next):.6f}")
            logging.info(f"Effective Learning Rate, γ' = {gamma}")

            logging.info(
                f"gamma = {float(gamma):.6f}  "
                f"(lr_scale={float(current_lr_scale):.6f})"
            )
            logging.info(f"dmu/dalpha={float(dmu_dalpha):.6f}")

            # Safe formatting for possibly-None values
            r_str = f"{float(r_cent):.6f}" if r_cent is not None else "None"
            b_str = f"{float(bound):.6f}" if bound is not None else "None"
            mk_str = (
                f"{float(m_k.item() if hasattr(m_k, 'item') else m_k):.6f}"
                if (m_k is not None) else "None"
            )
            logging.info(
                f"[LIN-CHK] ready={self.fd_tracker.linear_slope_ready} "
                f"resid_ok={bool(resid_ok)} r_cent={r_str} bound={b_str} m_k={mk_str}"
            )
            logging.info(f"loss = {float(loss_val):.6f}")

            # From the MacroStats record:
            mu_k = float(rec_k["mu"].item()) if "mu" in rec_k else float("nan")
            n_k = int(rec_k["n"].item()) if "n" in rec_k else int(self.max_iter)
            # std is either provided or computed from var
            if "std" in rec_k:
                std_k = float(rec_k["std"].item() if hasattr(rec_k["std"], "item") else rec_k["std"])
            elif "var" in rec_k:
                _var = float(rec_k["var"].item() if hasattr(rec_k["var"], "item") else rec_k["var"])
                std_k = (max(_var, 0.0)) ** 0.5
            else:
                std_k = float("nan")
            logging.info(f"mu(α_k)={mu_k:.6f}")
            logging.info(f"n={n_k}")
            logging.info(f"std={std_k:.6f}")

            # Logging every step_size epochs.
            if (epoch % self.step_size == 0) and (epoch > 0):
                # Update max_iter.
                new_max_iter = self.max_iter * self.max_iter_factor
                # Prevent max_iter from going over the max_iter_limit.
                self.max_iter = min(new_max_iter, self.max_iter_limit)
                # Scheduler step: Adjust the learning rate according to the schedule, γ decays over epochs.
                self.scheduler.step()

        # Return the final optimized alpha and the final loss value
        return self.alpha.item(), final_loss.item()
