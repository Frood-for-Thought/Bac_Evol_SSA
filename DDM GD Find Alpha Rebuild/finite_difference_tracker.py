import torch
from macro_stats import MacroStats
"""
finite_difference_tracker.py
This module defines the FiniteDifferenceTracker class for managing first-order finite difference
derivative estimates between sampled α values in a stochastic optimization setting.
Responsibilities:
    - Compute forward finite differences for sample mean μ(α) and sample variance s²(α).
    - Maintain a clean record of pairwise slopes between α_k and α_{k+1}.
    - Estimate the derivative at a specific α using available surrounding intervals, averaging
      left and right-sided slopes if both are available.
    - Support complete recomputation of finite difference structure when α insertions or reordering occur.
This module assumes macroscopic statistics are managed separately via the MacroStats class.
"""


class FiniteDifferenceTracker:
    def __init__(self, epsilon, slope_tol, stderr_tol):
        # fd_records is a *history of adjacent-α finite differences*, rebuilt as needed.
        # Each entry contains the forward differences for μ and s² between consecutive α’s.
        self.fd_records = []  # Each: {"alpha_k": α_k,"alpha_k+1": α_k+1,"dmu_dalpha": ...,"dvar_dalpha": ..., "h": ...}
        # Store Polyak–Ruppert slope history as torch tensor list
        # PR slope history stores successive m_k estimates so we can check stabilization (|Δm| < slope_tol).
        self.slope_history = []  # list of torch scalars

        # Parameters for stability, tolerance, confidence
        # Numeric/config parameters.
        self.epsilon = epsilon  # small positive guard added to denominators for stability (denominator guard)
        self.slope_tol = slope_tol  # tolerance for declaring PR slope stabilized, convergence threshold for m_k
        # multiplier used in residual_gate threshold (stderr-style bound), for stderr-based agreement.
        self.stderr_tol = stderr_tol

        # Most recent slope estimate; m_k denotes the slope estimate of the expectation function μ(α) at iteration k.
        self.m_k = None  # type: torch.Tensor or None

        # Linear-estimation mode toggle & bookkeeping
        self.linear_mode_enabled: bool = False  # flag: Record if True, (_linear_estimation_function() runs on call)
        # record_start_index in stats.records to begin slope estimation,
        # reserved (not used here) for future incremental scans.
        self.record_start_index = 0
        self.linear_windows = {}  # holds the active [α_min, α_max] window (tuple)

        # Linear Slope Check, Stabilization flags & deltas
        self.linear_slope_ready = False  # becomes True iff the last two m_k’s differ by |m_k - m_{k-1}| < slope_tol.
        self.last_delta_m = None  # Abs diff between last two m_k’s within slope_history list (None initially), |Δm|
        # _last_inrange_count track how many α samples are currently inside the sign-change interval.
        # Used to ensure the function only append/update slope readiness when a *new* point enters.
        self._last_inrange_count = 0

        # Per-window means (used by centered residual and gate); set each time PR sloped is recomputed.
        self.alpha_bar = None  # mean α in the current window
        self.mu_bar = None  # mean μ in the current window

    @staticmethod
    def get_alpha_pair(stats: MacroStats, alpha_k: float, atol: float = 0.01):
        """
        Retrieves the RECORDS for α_k and α_k+1 (i.e. α_k + h), assuming stats.records is sorted.

        Parameters:
            stats (MacroStats): The statistics object
            alpha_k (float): The α_k value
            atol (float): Tolerance for float comparison

        Returns:
            rec_k (dict): record at α_k
            rec_kp1 (dict): record at α_k + h
        """
        # Convert query α to tensor for torch.isclose comparisons.
        alpha_k_tensor = torch.tensor(alpha_k, dtype=torch.float32)

        # Ensure records are sorted by alpha before indexing neighbors.
        stats.records.sort(key=lambda r: r["alpha"].item())

        # Find the index of the exact (within atol) match for α_k.
        index = next(
            (i for i, r in enumerate(stats.records) if torch.isclose(r["alpha"], alpha_k_tensor, atol=atol)),
            None
        )
        if index is None:
            raise ValueError(f"No record found for alpha_k = {alpha_k}")

        rec_k = stats.records[index]

        # Get the next record to the right in the sorted list
        if index + 1 >= len(stats.records):
            raise ValueError(f"No α_k+1 found after α_k = {alpha_k}")
        rec_kp1 = stats.records[index + 1]

        return rec_k, rec_kp1

    def compute_and_store(self, rec_k: dict, rec_kp1: dict, decimals: int = 6):
        """
        Computes finite difference estimates between two macro records:
            - dμ/dα ≈ (μ_k+1 - μ_k) / (α_k+1 - α_k)
            - ds²/dα ≈ (s²_k+1 - s²_k) / (α_k+1 - α_k)

        This function also stores the result in self.fd_records for later reference.

        Parameters:
            rec_k (dict): Record at α_k
            rec_kp1 (dict): Record at α_k + h
            decimals (int): Rounding precision for α identifiers

        Returns:
            dict with:
                - 'alpha_pair': tuple of (α_k, α_k+1)
                - 'dmu_dalpha': float
                - 'dvar_dalpha': float
                - 'h': float (step size used)
        """
        # Round α identifiers for stable dictionary keys/printing.
        α_k = round(float(rec_k["alpha"]), decimals)
        α_kp1 = round(float(rec_kp1["alpha"]), decimals)
        h = α_kp1 - α_k

        # Guard: must have a non-negligible step.
        if abs(h) < 1e-12:
            raise ValueError("Finite difference step h is effectively zero.")

        # Forward differences for mean and variance.
        dmu = (rec_kp1["mu"] - rec_k["mu"]) / h
        dvar = (rec_kp1["var"] - rec_k["var"]) / h

        # Persist the computed differences for downstream use (e.g., estimate_derivative_at).
        fd_record = {
            "alpha_k": α_k,
            "alpha_kp1": α_kp1,
            "dmu_dalpha": dmu.item(),
            "dvar_dalpha": dvar.item(),
            "h": h
        }

        self.fd_records.append(fd_record)
        return fd_record

    def compute_all_differences(self, stats: 'MacroStats', decimals: int = 6):
        """
        Fully recomputes all finite differences between adjacent α values in stats.records.
        Clears fd_records and repopulates it cleanly.
        Uses function compute_and_store(self) to populate fd_records.

        Parameters:
            stats (MacroStats): The macro-level statistics tracker.
            decimals (int): Rounding precision for α comparison.
        """
        # Start from a clean slate.
        self.fd_records.clear()  # Clear out old
        # Sort stats.records by α (in-place) before pairing consecutive entries.
        stats.records.sort(key=lambda r: round(float(r["alpha"]), decimals))

        # Walk consecutive pairs and compute forward differences.
        for i in range(len(stats.records) - 1):
            rec_k = stats.records[i]
            rec_kp1 = stats.records[i + 1]
            try:
                self.compute_and_store(rec_k, rec_kp1, decimals=decimals)
            except ValueError as e:
                # Skip degenerate pairs; keep going so the rest of the structure remains usable.
                print(f"Skipping invalid α pair ({rec_k['alpha']}, {rec_kp1['alpha']}): {e}")

    def estimate_derivative_at(self, alpha: float, kind: str = "mu", decimals: int = 6):
        """
        Estimates the derivative (mean or variance) at a specific α using surrounding finite differences.
        If both left and right intervals exist, average them. Otherwise, use whichever exists.
        Parameters:
            alpha (float): The α value at which to estimate the derivative.
            kind (str): Either "mu" or "var".
            decimals (int): Precision for float matching.
        Returns:
            float or None: Estimated slope at α
        """
        # Round query α for stable matching against stored keys.
        alpha = round(alpha, decimals)
        left = None
        right = None

        # Locate the two FD entries that “touch” α from left and right.
        for fd in self.fd_records:
            a_k = round(fd["alpha_k"], decimals)
            a_kp1 = round(fd["alpha_kp1"], decimals)

            if a_kp1 == alpha:
                left = fd
            elif a_k == alpha:
                right = fd

        key = "dmu_dalpha" if kind == "mu" else "dvar_dalpha"

        # If both sides exist, average for a centered estimate; else use what’s available.
        if left and right:
            return 0.5 * (left[key] + right[key])
        elif right:
            return right[key]
        elif left:
            return left[key]
        else:
            return None  # No estimate available

    def centered_residual(self, alpha, mu_val):
        """
        Compute r_cent(α) = (μ(α) - μ̄) - m_k * (α - ᾱ).
        # ------------------------------------------------------------------
        # Centered Residual
        # Testing |μ(α) − m_k α| against a stderr bound is biased if the local line has an intercept b ≠ 0.
        # That uncentered residual is approximately b + noise,
        # so it won’t go to zero as sampling noise shrinks.
        # The centered residual removes the intercept automatically:
        #   r_cent(α) = (μ(α) − μ̄) − m_k (α − ᾱ).
        # If the local model is linear (μ(α) ≈ m_k α + b), then
        #   (μ − μ̄) − m_k (α − ᾱ) ≈ 0 + sampling noise.
        # So r_cent behaves like “pure noise” and is the right thing to compare to a stderr-type threshold.
        # Gate: declare “linear-agreement OK” if
        #   |r_cent(α)| ≤ stderr_tol · 2 s(α) / √n
        # (Note: the bound uses 2*s(α)/sqrt(n), not 2*s(α)/n.)
        # Store the window means here so other code can compute r_cent later without
        # touching the slope math below. (No behavior change to the slope calculation.)
        # ------------------------------------------------------------------
        Returns None if m_k or the window means are not set yet.
        """
        # Requires: m_k and (ᾱ, μ̄) already computed by run_linear_estimation().
        if self.m_k is None or self.alpha_bar is None or self.mu_bar is None:
            return None
        # Convert inputs to plain floats.
        mk = float(self.m_k.item() if hasattr(self.m_k, "item") else self.m_k)
        a = float(alpha.item() if hasattr(alpha, "item") else alpha)
        mu = float(mu_val.item() if hasattr(mu_val, "item") else mu_val)
        # Centered residual, unbiased for intercept under local linearity.
        return (mu - self.mu_bar) - mk * (a - self.alpha_bar)

    def residual_gate(self, alpha, mu_val, var_val, n: int):
        """
        Returns (resid_ok, r_cent, bound) for the centered-residual gate:
            |r_cent(α)| ≤ stderr_tol * (2 * s(α) / sqrt(n))
        where r_cent(α) = (μ(α) - μ̄) - m_k * (α - ᾱ).
        """
        # Compute centered residual; if unavailable (no m_k or means yet), return a safe False
        r = self.centered_residual(alpha, mu_val)
        if r is None or self.stderr_tol is None:
            return False, r, None
        # s(α) = sqrt(var); the bound uses an approximate 95% CI width 2*s/sqrt(n) scaled by stderr_tol
        s = float(torch.sqrt(var_val).item() if hasattr(var_val, "item") else (var_val ** 0.5))
        bound = self.stderr_tol * (2.0 * s / (n ** 0.5))
        # resid_ok says whether the centered residual is within the tolerance band
        return (abs(r) <= bound), r, bound

    def run_linear_estimation(self,
                              stats: 'MacroStats',
                              enabled: bool = None,
                              window: list = None,
                              reset: bool = False):
        """
        Public entry: toggles linear mode and runs PR slope estimation if enabled.
        """
        # Toggle master switch of the “linear mode”.
        if enabled is not None:
            self.linear_mode_enabled = bool(enabled)

        # Only run the estimator if enabled.
        if self.linear_mode_enabled:
            self._linear_estimation_function(stats, window=window, reset=reset)

    def _linear_estimation_function(self,
                                    stats: 'MacroStats',
                                    window: list = None,
                                    reset: bool = False
                                    ):

        # Store the current window configuration.
        # If a new window is supplied, store it and (optionally) reset state.
        if window is not None:  # Only do this if a window specification was passed in.
            # Just store the alpha bounds directly, Keep tuple (α_min, α_max) as the active PR region.
            self.linear_windows = tuple(window)  # e.g., (alpha_min, alpha_max)

            # If reset flag is True OR there is no existing window configuration
            if reset or not self.slope_history:
                # Clear Polyak–Ruppert slope tracking state
                self.slope_history.clear()
                self.m_k = None
                self.record_start_index = 0
                self._last_inrange_count = 0
                # Also clear per-window means and flags so no stale values leak across windows.
                self.alpha_bar = None
                self.mu_bar = None
                self.last_delta_m = None
                self.linear_slope_ready = False

        # If bounds exist, scan through stats.records to find matching alphas.
        # If we have an active window, gather all (α, μ) that fall inside it.
        if self.linear_windows:
            alpha_min, alpha_max = self.linear_windows
            # Temporary holders for in-range alpha and mu values.
            # Accumulate in-range α’s and μ’s.
            alphas_in_range = []
            mus_in_range = []

            for rec in stats.records:
                a_val = rec["alpha"].item() if hasattr(rec["alpha"], "item") else float(rec["alpha"])
                if alpha_min <= a_val <= alpha_max:
                    alphas_in_range.append(a_val)
                    # mu is already a tensor — convert to float for regression calculation.
                    mu_val = rec["mu"].item() if hasattr(rec["mu"], "item") else float(rec["mu"])
                    mus_in_range.append(mu_val)

            # Compute Polyak–Ruppert slope m_k using vectorized PyTorch ops over the window if there is data.
            if alphas_in_range and mus_in_range:
                with torch.no_grad():
                    a = torch.as_tensor(alphas_in_range, dtype=torch.float32)
                    mu = torch.as_tensor(mus_in_range, dtype=torch.float32)
                    # ------------------------------------------------------------------
                    # Centered Residual
                    # Testing |μ(α) − m_k α| against a stderr bound is biased if the local line has an intercept b ≠ 0.
                    # That uncentered residual is approximately b + noise,
                    # so it won’t go to zero as sampling noise shrinks.
                    # The centered residual removes the intercept automatically:
                    #   r_cent(α) = (μ(α) − μ̄) − m_k (α − ᾱ).
                    # If the local model is linear (μ(α) ≈ m_k α + b), then
                    #   (μ − μ̄) − m_k (α − ᾱ) ≈ 0 + sampling noise.
                    # So r_cent behaves like “pure noise” and is the right thing to compare to a stderr-type threshold.
                    # Gate: declare “linear-agreement OK” if
                    #   |r_cent(α)| ≤ stderr_tol · 2 s(α) / √n
                    # (Note: the bound uses 2*s(α)/sqrt(n), not 2*s(α)/n.)
                    # Store the window means here so other code can compute r_cent later without
                    # touching the slope math below. (No behavior change to the slope calculation.)
                    # ------------------------------------------------------------------
                    # Store window means so centered residual can be computed elsewhere (e.g., residual_gate).
                    self.alpha_bar = float(a.mean().item())
                    self.mu_bar = float(mu.mean().item())

                    # Centered least-squares slope to remove intercept bias.
                    # m_k = Σ( (α_i − ᾱ)(μ_i − μ̄) ) / [ Σ( (α_i − ᾱ)^2 ) + ε ]
                    a_c = a - a.mean()  # α_i − ᾱ
                    mu_c = mu - mu.mean()  # μ_i − μ̄

                    # Denominator with ε guard for stability.
                    # Σ (α_i − ᾱ)^2 + ε   (denominator).
                    denom = torch.dot(a_c, a_c) + torch.as_tensor(self.epsilon, dtype=a.dtype, device=a.device)

                    # Numerator: covariance-like term between centered α and centered μ.
                    # Σ (α_i − ᾱ)(μ_i − μ̄)   (numerator).
                    num = torch.dot(a_c, mu_c)

                    # If denominator is finite/nonzero, accept the new slope estimate.
                    if torch.isfinite(denom) and denom.item() != 0.0:
                        m_k = num / denom
                        self.m_k = m_k

            # after computing self.m_k, update stabilization metrics only when *new* points have entered the window.
            current_count = len(alphas_in_range)
            if current_count > self._last_inrange_count and self.m_k is not None:
                # Compute m_k slope history, record this slope for stabilization checks.
                self.slope_history.append(self.m_k)  # keep for later stability checks
                # Update readiness, if we have at least two slope estimates, compute |Δm|.
                if len(self.slope_history) >= 2:
                    with torch.no_grad():
                        delta_m = torch.abs(self.slope_history[-1] - self.slope_history[-2])
                        self.last_delta_m = delta_m
                        # Criterion 1: slope stabilization, declare “ready” once |Δm| falls below slope_tol.
                        self.linear_slope_ready = bool(delta_m.item() < self.slope_tol)
                # Remember the new in-range count for next time
                self._last_inrange_count = current_count

            # Return the latest slope estimate and readiness flag to the caller (e.g., for logging).
            return self.m_k, self.linear_slope_ready
