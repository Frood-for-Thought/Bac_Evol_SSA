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
        self.fd_records = []  # Each: {"alpha_k": α_k, "alpha_k+1": α_k+1, "dmu_dalpha": ..., "dvar_dalpha": ..., "h": ...}
        # Store Polyak–Ruppert slope history as torch tensor list
        self.slope_history = []  # list of torch scalars
        # Parameters for stability, tolerance, confidence
        self.epsilon = epsilon                    # numerical stability (denominator guard)
        self.slope_tol = slope_tol                # convergence threshold for m_k
        self.stderr_tol = stderr_tol              # multiplier for stderr-based agreement
        # Most recent slope estimate
        self.m_k = None  # type: torch.Tensor or None
        self.linear_mode_enabled: bool = False  # flag: only record if True
        self.record_start_index = 0  # index in stats.records to begin slope estimation
        self.linear_windows = {}
        # Linear Slope Check
        self.linear_slope_ready = False  # becomes True when |m_k - m_{k-1}| < slope_tol
        self.last_delta_m = None  # tracks the most recent |Δm|

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
        alpha_k_tensor = torch.tensor(alpha_k, dtype=torch.float32)

        # Ensure records are sorted by alpha
        stats.records.sort(key=lambda r: r["alpha"].item())

        # Find the index of the exact match
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
        α_k = round(float(rec_k["alpha"]), decimals)
        α_kp1 = round(float(rec_kp1["alpha"]), decimals)
        h = α_kp1 - α_k

        if abs(h) < 1e-12:
            raise ValueError("Finite difference step h is effectively zero.")

        dmu = (rec_kp1["mu"] - rec_k["mu"]) / h
        dvar = (rec_kp1["var"] - rec_k["var"]) / h

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
        self.fd_records.clear()  # Clear out old
        # Sort stats.records by α (in-place)
        stats.records.sort(key=lambda r: round(float(r["alpha"]), decimals))

        for i in range(len(stats.records) - 1):
            rec_k = stats.records[i]
            rec_kp1 = stats.records[i + 1]
            try:
                self.compute_and_store(rec_k, rec_kp1, decimals=decimals)
            except ValueError as e:
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
        alpha = round(alpha, decimals)
        left = None
        right = None

        for fd in self.fd_records:
            a_k = round(fd["alpha_k"], decimals)
            a_kp1 = round(fd["alpha_kp1"], decimals)

            if a_kp1 == alpha:
                left = fd
            elif a_k == alpha:
                right = fd

        key = "dmu_dalpha" if kind == "mu" else "dvar_dalpha"

        if left and right:
            return 0.5 * (left[key] + right[key])
        elif right:
            return right[key]
        elif left:
            return left[key]
        else:
            return None  # No estimate available

    def run_linear_estimation(self,
                              stats: 'MacroStats',
                              enabled: bool = None,
                              window: list = None,
                              reset: bool = False):
        """
        Public entry: toggles linear mode and runs PR slope estimation if enabled.
        """
        # Toggle master switch
        if enabled is not None:
            self.linear_mode_enabled = bool(enabled)

        # Only run the estimator if enabled
        if self.linear_mode_enabled:
            self._linear_estimation_function(stats, window=window, reset=reset)

    def _linear_estimation_function(self,
                                    stats: 'MacroStats',
                                    window: list = None,
                                    reset: bool = False
                                    ):

        # Store the current window configuration.
        if window is not None:  # Only do this if a window specification was passed in.
            # Just store the alpha bounds directly
            self.linear_windows = tuple(window)  # e.g., (alpha_min, alpha_max)

            # If reset flag is True OR there is no existing window configuration
            if reset or not self.slope_history:
                # Clear Polyak–Ruppert slope tracking state
                self.slope_history.clear()
                self.m_k = None
                self.record_start_index = 0

        # If bounds exist, scan through stats.records to find matching alphas
        if self.linear_windows:
            alpha_min, alpha_max = self.linear_windows
            # Temporary holders for in-range alpha and mu values
            alphas_in_range = []
            mus_in_range = []

            for rec in stats.records:
                a_val = rec["alpha"].item() if hasattr(rec["alpha"], "item") else float(rec["alpha"])
                if alpha_min <= a_val <= alpha_max:
                    alphas_in_range.append(a_val)
                    # mu is already a tensor — convert to float for calculation
                    mu_val = rec["mu"].item() if hasattr(rec["mu"], "item") else float(rec["mu"])
                    mus_in_range.append(mu_val)

            # Compute Polyak–Ruppert slope m_k using vectorized PyTorch ops
            if alphas_in_range and mus_in_range:
                with torch.no_grad():
                    a = torch.as_tensor(alphas_in_range, dtype=torch.float32)
                    mu = torch.as_tensor(mus_in_range, dtype=torch.float32)

                    # Σ α_i μ(α_i)
                    num = torch.dot(a, mu)
                    # Σ α_i^2 + ε
                    denom = torch.dot(a, a) + torch.as_tensor(self.epsilon, dtype=a.dtype, device=a.device)

                    if torch.isfinite(denom) and denom.item() != 0.0:
                        m_k = num / denom
                        self.m_k = m_k
                        self.slope_history.append(m_k)  # keep for later stability checks

            # After: self.slope_history.append(m_k)
            if len(self.slope_history) >= 2:
                with torch.no_grad():
                    delta_m = torch.abs(self.slope_history[-1] - self.slope_history[-2])
                    self.last_delta_m = delta_m
                    # Criterion 1: slope stabilization
                    self.linear_slope_ready = bool(delta_m.item() < self.slope_tol)

            return self.m_k, self.linear_slope_ready
