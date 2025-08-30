import torch
import math
from macro_stats import MacroStats
from finite_difference_tracker import FiniteDifferenceTracker
from landscape_analysis import detect_sign_transitions
from loss_evaluator import LossEvaluator


def sample_v_func_NU(alpha: float, n: int, m1: float = 1.0, h: float = 8.0, sigma: float = 0.1) -> torch.Tensor:
    """
    PyTorch-compatible stochastic function with piecewise expectation.
    This function has no globally uniform expectation function, and may cause unpredictable convergence.

    PyTorch-compatible stochastic function with piecewise expectation:
    E[v_j(α)] = -(α - 5)^2 + 6 for α < h
    E[v_j(α)] = m1*α - 11 for α ≥ h

    Parameters:
        alpha : float       # Current parameter value
        m1    : float       # Linear slope in region α > h
        h     : float       # Breakpoint separating regions
        sigma : float       # Standard deviation of Gaussian noise

    Returns:
        v_j : torch.Tensor  # One sample from v_j(α)
    """
    if isinstance(alpha, torch.Tensor):
        # If alpha is already a tensor, use clone().detach() to avoid warning; else convert from float.
        alpha_tensor = alpha.clone().detach().to(dtype=torch.float32)
    else:
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32)  # ensure it's a float

    if alpha < h:
        mean = -((alpha_tensor - 5) ** 2) + 6
    else:
        mean = m1 * alpha_tensor - 11

    noise = torch.normal(mean=0.0, std=sigma, size=(n,))
    samples = mean + noise  # broadcasting scalar mean over n noise samples
    return samples  # Return a tensor of a batch of n stochastic samples with torch.Size([1000]).
    # Each sample is: vj​(α) = mean(α) + Gaussian noise



def main():
    # Initialize macro stats tracker and finite difference tracker
    stats = MacroStats()
    # slope_tol is the convergence threshold for m_k, to measure when ∣m(α_(k+1) )-m(α_k )∣ < slope_tol.
    fd_tracker = FiniteDifferenceTracker(epsilon=1e-8, slope_tol=1e-5, stderr_tol=2.0)

    # Define sampling parameters
    α_min = 0
    α_max = 22
    alpha_range = torch.arange(α_min, α_max, step=1.0, dtype=torch.float32)
    n = 1000000
    v_d = 7.0
    lambda_var = 1.0
    # Safety margin for the gamma cap derived from m_k (keeps |1-2*gamma*m_k^2| < 1)
    eta_for_gamma_cap = 0.05  # 5% margin

    # Collect macro observations for each α
    for α in alpha_range:
        samples = sample_v_func_NU(alpha=α, n=n)
        stats.macro_observations(alpha=α, samples=samples)

    # Insert another value in between
    α_insert = 10.5
    samples_insert = sample_v_func_NU(alpha=α_insert, n=n)
    stats.macro_observations(alpha=α_insert, samples=samples_insert)

    # Compute all forward finite differences
    fd_tracker.compute_all_differences(stats)

    # Display finite differences near the insertion point
    print("\nFinite differences near α = 10.0 and α = 10.5:")
    for fd in fd_tracker.fd_records:
        if abs(fd["alpha_k"] - 10.0) < 1e-6 or abs(fd["alpha_k"] - 10.5) < 1e-6:
            print(fd)

    # Display all computed finite differences (optional)
    print("\nAll finite differences:")
    for fd in fd_tracker.fd_records:
        print(fd)

    # Detect μ(α) - v_d sign changes.
    transitions = detect_sign_transitions(stats, v_d=v_d)
    if not transitions:
        raise RuntimeError("No sign transitions detected.")
    else:
        print(f"\nDetected sign transitions in μ(α) - v_d for v_d = {v_d}")
        α_left, α_right = transitions[0]  # Select the first transition interval
        print(f"Between α = {α_left} and α = {α_right}")

    # Ensure α_left is recorded.
    if not any(abs(r["alpha"].item() - α_left) < 1e-6 for r in stats.records):
        samples_left = sample_v_func_NU(alpha=α_left, n=n)
        stats.macro_observations(alpha=α_left, samples=samples_left)

    # Enable Polyak–Ruppert slope estimation over the first bracket; start fresh so slope_history is clean.
    fd_tracker.run_linear_estimation(stats, enabled=True, window=[α_left, α_right], reset=True)

    # Initialize LossEvaluator with injected sampling function
    loss_eval = LossEvaluator(
        stats=stats,
        fd_tracker=fd_tracker,
        v_d=v_d,
        lambda_var=lambda_var,
        sample_func=sample_v_func_NU  # Inject the sampling function
    )

    # Loop through two optimization iterations
    alpha_k = α_left  # Start from the left of the first detected transition

    for i in range(300):
        # Ensure αlpha_k is recorded, observe and update finite differences.
        if not any(abs(r["alpha"].item() - alpha_k) < 1e-6 for r in stats.records):
            samples = sample_v_func_NU(alpha=alpha_k, n=n)
            stats.macro_observations(alpha=alpha_k, samples=samples)
            fd_tracker.compute_all_differences(stats)

        # Centered-residual gate (LOG ONLY; does not change behavior)
        # Pull the μ, var at the *current* alpha_k (we just ensured it exists).
        rec_k = next(r for r in stats.records if abs(r["alpha"].item() - alpha_k) < 1e-6)
        resid_ok, r_cent, bound = fd_tracker.residual_gate(
            alpha=rec_k["alpha"],    # tensor is fine
            mu_val=rec_k["mu"],      # tensor
            var_val=rec_k["var"],    # tensor
            n=n
        )
        print(f"[LIN-CHK] ready={fd_tracker.linear_slope_ready}, "
              f"resid_ok={resid_ok}, "
              f"r_cent={float(r_cent) if r_cent is not None else None}, "
              f"bound={float(bound) if bound is not None else None}")

        # Update PR slope each iteration (no reset), then report readiness.
        fd_tracker.run_linear_estimation(stats, enabled=True, window=None, reset=False)

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
        #      - 0 < gamma < 1/(2 m_k^2):       q in (0, 1)     → monotone convergence (no sign flips).
        #      - gamma = 1/(2 m_k^2):           q = 0           → one-step to alpha_star in  ideal linear/noiseless case
        #      - 1/(2 m_k^2) < gamma < 1/m_k^2: q in (-1, 0)    → convergent but oscillatory (e_k flips sign each step).
        #      - gamma = 1/m_k^2:               q = -1          → no contraction; persistent large oscillation.
        #      - gamma > 1/m_k^2:               |q| > 1         → divergence (errors grow).
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
        m_k = fd_tracker.m_k
        delta_m = fd_tracker.last_delta_m
        if fd_tracker.last_delta_m is not None:
            print(f"[PR] m_k={float(m_k) if m_k is not None else None}, "
                  f"|Δm|={float(delta_m) if delta_m is not None else None}, "
                  f"ready={fd_tracker.linear_slope_ready}")

        else:
            print("[PR] collecting slopes…")

        # Estimate dμ/dα externally
        dmu_dalpha = fd_tracker.estimate_derivative_at(alpha=alpha_k, kind="mu")
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

        # Adaptive stability cap on gamma (ONLY when PR slope is ready).
        # From linear analysis: convergence needs |1 - 2*gamma*m_k^2| < 1 ⇒ gamma < 1/m_k^2.
        # Best non-oscillatory contraction at gamma = 1/(2*m_k^2).
        if fd_tracker.linear_slope_ready and (m_k is not None):
            mk_val = float(m_k.item()) if hasattr(m_k, "item") else float(m_k)
            if math.isfinite(mk_val):
                mk2 = mk_val * mk_val
                if mk2 > 0.0:
                    gamma_cap = (1.0 - eta_for_gamma_cap) / (2.0 * mk2)
                    # Fail fast if cap is nonsensical
                    if not math.isfinite(gamma_cap) or gamma_cap <= 0.0:
                        raise RuntimeError(
                            f"[GD-ERROR] Bad gamma_cap from m_k: m_k={mk_val}, gamma_cap={gamma_cap}"
                        )
                    # Cap gamma (keep original scaling but make it safe).
                    if gamma > gamma_cap:
                        gamma = gamma_cap

        # Decide next alpha based on current α_left
        alpha_next = loss_eval.decide_next_alpha(
            alpha_k=alpha_k,
            n=n,
            gamma=gamma,
            min_step_size=0.1,
            fitness_thresh=0.1,
            slope_thresh=1e-3,
            alpha_left=α_left,
            alpha_right=α_right,
            use_gradient_override=True  # Or True / False to control override behavior
        )

        print(f"\nIteration {i+1}")
        print(f"  alpha_k    = {alpha_k}")
        print(f"  gamma      = {gamma}")
        print(f"  alpha_next = {alpha_next}")

        # Prepare for next iteration
        alpha_k = alpha_next

    # Display all computed finite differences (optional)
    print("\nAll finite differences:")
    print("\nAll finite differences (with macro observations):")
    atol = 1e-6

    for fd in fd_tracker.fd_records:
        # Show the finite-difference record itself
        print(f"\nFD: {fd}")

        a_k = fd["alpha_k"]
        a_kp1 = fd["alpha_kp1"]

        # Find matching macro records for α_k and α_{k+1}
        def _get_alpha_val(x):
            return x.item() if hasattr(x, "item") else float(x)

        rec_k = next((r for r in stats.records if abs(_get_alpha_val(r["alpha"]) - a_k) < atol), None)
        rec_kp1 = next((r for r in stats.records if abs(_get_alpha_val(r["alpha"]) - a_kp1) < atol), None)

        def _show(label, rec):
            if rec is None:
                print(f"  {label}: <no record>")
                return
            a = _get_alpha_val(rec["alpha"])
            mu = _get_alpha_val(rec["mu"])
            var = _get_alpha_val(rec["var"])
            n = rec.get("n", None)
            std = rec.get("std", None)
            ci = rec.get("stderr_ci", None)

            base = f"  {label} α={a:.6f}, μ={mu:.6f}, var={var:.6f}"
            if n is not None: base += f", n={int(n)}"
            if std is not None: base += f", std={_get_alpha_val(std):.6f}"
            if ci is not None: base += f", stderr_ci={_get_alpha_val(ci):.6f}"
            print(base)

        _show("alpha_k   ", rec_k)
        _show("alpha_k+1 ", rec_kp1)

    # After showing all finite differences + macro observations,
    # this displays latest Polyak–Ruppert status from the tracker.
    print("\n[PR] m_k={}, |Δm|={}, ready={}".format(
        float(fd_tracker.m_k) if fd_tracker.m_k is not None else None,
        float(fd_tracker.last_delta_m) if fd_tracker.last_delta_m is not None else None,
        fd_tracker.linear_slope_ready
    ))

if __name__ == "__main__":
    main()
