import torch
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
    fd_tracker = FiniteDifferenceTracker(epsilon=1e-8, slope_tol=0.01, stderr_tol=2.0)

    # Define sampling parameters
    α_min = 0
    α_max = 22
    alpha_range = torch.arange(α_min, α_max, step=1.0, dtype=torch.float32)
    n = 1000000
    v_d = 7.0
    lambda_var = 1.0

    # Collect macro observations for each α
    for α in alpha_range:
        samples = sample_v_func_NU(alpha=α, n=n)
        stats.macro_observations(alpha=α, samples=samples)

    # Compute all forward finite differences
    fd_tracker.compute_all_differences(stats)

    # Insert another value in between
    α_insert = 10.5
    samples_insert = sample_v_func_NU(alpha=α_insert, n=n)
    stats.macro_observations(alpha=α_insert, samples=samples_insert)

    # Recompute all finite differences after insertion
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

    # Initialize LossEvaluator with injected sampling function
    loss_eval = LossEvaluator(
        stats=stats,
        fd_tracker=fd_tracker,
        v_d=v_d,
        lambda_var=lambda_var,
        sample_func=sample_v_func_NU  # Inject the sampling function
    )

    # Estimate dμ/dα externally
    dmu_dalpha = fd_tracker.estimate_derivative_at(alpha=α_left, kind="mu")
    gamma = 1.0/dmu_dalpha

    # Decide next alpha based on current α_left
    alpha_next = loss_eval.decide_next_alpha(
        alpha_k=α_left,
        n=n,
        gamma=gamma,
        min_step_size=0.1,
        fitness_thresh=0.1,
        slope_thresh=1e-3,
        alpha_left=α_left,
        alpha_right=α_right,
        use_gradient_override=True  # Or True / False to control override behavior
    )

    print(f"alpha_left = {α_left}")
    print(f"alpha_next = {alpha_next}")


if __name__ == "__main__":
    main()
