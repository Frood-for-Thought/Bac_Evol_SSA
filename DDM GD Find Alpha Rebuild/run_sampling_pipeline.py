import torch
from macro_stats import MacroStats
from finite_difference_tracker import FiniteDifferenceTracker
from landscape_analysis import detect_sign_transitions


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

    # Detect μ(α) - v_d sign changes
    transitions = detect_sign_transitions(stats, v_d=0.0)
    print("\nDetected sign transitions in μ(α) - v_d:")
    for α_left, α_right in transitions:
        print(f"Between α = {α_left} and α = {α_right}")


if __name__ == "__main__":
    main()
