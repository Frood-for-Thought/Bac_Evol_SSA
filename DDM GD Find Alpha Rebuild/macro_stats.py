import torch
"""
This module defines the MacroStats class, which manages macroscopic statistical observations
for a stochastic sampling-based optimization algorithm. Each observation corresponds to a parameter
value α and a batch of independently sampled outputs {v_j(α)}.

Responsibilities:
    - Track and store sample mean, variance, standard deviation, and standard error at each α.
    - Support replacement and update of prior records for re-sampled α.
    - Suggest adaptive finite difference step sizes based on signal-to-noise ratio, fitness error,
      and local sample statistics, useful for exploring the expectation landscape μ(α) near a target v_d.

Designed to be used in conjunction with finite difference modules and loss evaluators.
"""


class MacroStats:
    def __init__(self, epsilon=1e-8, slope_tol=0.01, stderr_tol=2.0):
        # Store records as list of dicts with each α and its statistics
        self.records = []  # Each item: {"alpha": α, "mu": μ, "var": s², "stderr": δ}

    def macro_observations(self, alpha: float, samples: torch.Tensor, decimals: int = 6):
        """
        Register a new observation (α, {v_j(α)}), and compute macroscopic statistics.
        Given:
            - α: The current parameter value
            - {v_j(α)}: A batch of n sampled stochastic values at α
        Computes:
            - Sample mean:
                μ(α) ≈ (1/n) ∑_{j=1}^n v_j(α)
            - Sample variance (unbiased estimator):
                s²(α) ≈ (1 / (n - 1)) ∑_{j=1}^n (v_j(α) - μ(α))²
            - Approximate standard error bound (95% confidence):
                This is a confidence bound, based on the approximation that ~95% of sample means (if resampled many times) fall within this range under the Central Limit Theorem as:
                μ ± 2⋅SE(α)
                δ(α) ≈ (2s(α)) / √n
        Stores:
            {"alpha": α, "mu": μ, "var": s², "stderr": δ} as a dict in self.records
        Returns:
            - μ(α), s²(α), δ(α)
        """
        # Round α to consistent float precision
        alpha_rounded = round(float(alpha), decimals)
        alpha_tensor = torch.tensor(alpha_rounded, dtype=torch.float32)
        mu = samples.mean() # Sample mean.
        var = samples.var(unbiased=True) # Sample variance.
        std = torch.sqrt(var) # Sample standard deviation.
        stderr_ci = 2 * std / torch.sqrt \
            (torch.tensor(len(samples), dtype=torch.float32)) # Standard error bound (95% CI).

        # Create the record as a dict of torch scalars
        record = {
            "alpha": alpha_tensor,  # Alpha value.
            "n": len(samples),  # Number of samples.
            "mu": mu,  # Sample average.
            "var": var,  # Sample variance.
            "std": std,  # Sample standard deviation
            "stderr_ci": stderr_ci  # Sample standard error bound (95% CI).
        }

        # Check if alpha already exists. If so, replace it.
        for i, r in enumerate(self.records):
            if round(float(r["alpha"]), decimals) == alpha_rounded:
                self.records[i] = record
                break
        else:
            # Update the record within the function.
            self.records.append(record)

        return record  # Optionally return information for inspection.

    def suggest_step_size(self, alpha: float, v_d: float, d: int = 1, kappa: float = 100.0, min_h: float = 1e-3) -> float:
        """
        Suggest a finite difference step size h for probing μ(α) with adaptive scaling based on fitness error |μ(α) - v_d|.

        The step size is computed as:
            h = max(sqrt(d + κ) ⋅ s(α) / √n, dynamic_min_h)
        where:
            - d is the dimensionality of the input (default 1).
            - κ is a spread control parameter.
            - σ_α is the estimated standard deviation of the input parameter α.
            - dynamic_min_h increases based on how far μ(α) is from v_d.

        Since σ_α is not directly known, it is approximated from the observed output variance
        as:
            σ_α ≈ s(α) / √n
        where s(α) is the sample standard deviation of {v_j(α)}, and n is the sample size.
        Now, the step size is computed as:
            h ≈ sqrt(d + κ) ⋅ s(α) / √n
        Parameters:
            - alpha (float): The parameter location α to retrieve step size for.
            - d (int): Dimension of α, (default 1).
            - kappa (float): Spread parameter controlling sigma point scaling.
            - min_h (float): Floor value to prevent degenerately small step size, (default 1e-3).
        Returns:
            - h (float): Recommended probing step size for finite difference or sigma point generation
        """
        # Specified alpha.
        alpha_tensor = alpha.clone().detach().to(dtype=torch.float32) if isinstance(alpha, torch.Tensor) else torch.tensor \
            (alpha, dtype=torch.float32)
        # Search for matching α
        match = next((r for r in self.records if torch.isclose(r["alpha"], alpha_tensor, atol=1e-6)), None)
        if match is None:
            raise ValueError(f"No statistics found for α = {alpha} in recorded macroscopic observations.")

        s = match["std"]  # Sample standard deviation of v_j(α).
        n = match["n"]  # Sample size at this α (Python int).
        mu = match["mu"]  # Sample average at this α.

        # Dynamically increase floor for step size if far from target
        fitness_error = abs(mu.item() - v_d)

        if fitness_error > 1.0:
            dynamic_min_h = 0.5
        elif fitness_error > 0.1:
            dynamic_min_h = 0.05
        else:
            dynamic_min_h = min_h

        sigma_alpha = s / torch.sqrt(torch.tensor(n, dtype=torch.float32))
        h = torch.sqrt(torch.tensor(d + kappa, dtype=torch.float32)) * sigma_alpha
        h = torch.maximum(h, torch.tensor(dynamic_min_h))

        print(f"sigma_alpha = {sigma_alpha}, fitness_error: |μ(α) - v_d| = {fitness_error}, h = {h}, dynamic_min_h = {dynamic_min_h}")
        return h.item()
