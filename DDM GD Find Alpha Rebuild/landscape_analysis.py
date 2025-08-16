import torch
from typing import List, Tuple
from macro_stats import MacroStats


def detect_sign_transitions(stats: MacroStats, v_d: float, atol: float = 1e-6):
    """
    Detects intervals in α where the sign of (μ(α) - v_d) changes between consecutive records.

    This function scans through the recorded macro statistics (stats.records) and checks for sign changes
    in the deviation of the sample mean μ(α) from the target value v_d. A sign change indicates
    a crossing point where μ(α) - v_d switches from positive to negative or vice versa.

    This is useful for identifying bracketing intervals around roots or peaks in the expectation
    landscape where the function crosses the desired target level.

    Parameters:
        stats (MacroStats): The macro statistics object containing sampled μ(α) values.
        v_d (float): The target value against which μ(α) is compared.
        atol (float): Absolute tolerance for floating-point equality when comparing α values.

    Returns:
        transitions (List[Tuple[float, float]]): A list of tuples (α_left, α_right) marking
        intervals where the sign of (μ(α) - v_d) changes.
    """
    transitions = []
    records_ordered = sorted(stats.records, key=lambda r: r["alpha"].item())  # ensure correct order

    for i in range(1, len(records_ordered)):
        mu_i, mu_prev = records_ordered[i]["mu"], records_ordered[i-1]["mu"]
        sign_i = torch.sign(mu_i - v_d).item()
        sign_prev = torch.sign(mu_prev - v_d).item()
        if sign_i != sign_prev:
            α_left = records_ordered[i-1]["alpha"].item()
            α_right = records_ordered[i]["alpha"].item()
            transitions.append((α_left, α_right))
    return transitions
