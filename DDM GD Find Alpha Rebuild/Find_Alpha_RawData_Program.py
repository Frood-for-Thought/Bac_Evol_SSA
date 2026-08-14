import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
# Import the Tumble Angle Module
from Tumble_Angle import AngleGenerator_cuda
# Initialize the angle generator class to select from probability distribution.
angle_generator = AngleGenerator_cuda()
# uses .simulate_bacterial_movement_cuda(alpha, max_iter) to generate Run-and-Tumble data points.
from Calc_Alpha_ML_Function import Dynamic_Data_Evolving_Mean_Estimator, BaseDataGenerator
from Generate_Dynamic_Data_Points_Lower_GPU_and_Time import Norm_Vd_Mean_Data_Generator
from macro_stats import MacroStats
from finite_difference_tracker import FiniteDifferenceTracker
from landscape_analysis import detect_sign_transitions

# Initialization and Food Concentration Calculation.
nl = 101
Grad = 0.000405  # µm^-1
Max_Food_Conc = 60000  # µM
DL = 310  # µm
Food_Function = np.zeros(nl)
for Food_Pos in range(nl):
    Food_Function[Food_Pos] = np.exp(Grad * Food_Pos * DL)
Ini_Food_Const = Max_Food_Conc / Food_Function[-1]
xbias = Ini_Food_Const * Food_Function
# The theoretical parameters have been pre-calculated to fit onto.
# These parameters are for:
#     nl = 101
#     Grad = 0.000405  µm^-1
#     Max_Food_Conc = 60000  µM
#     DL = 310  µm
input_parameters = 'input_parameters.xlsx'
parameter_df = pd.read_excel(input_parameters)
vd_chemotaxis = parameter_df.loc[:, 'drift_velocity']  # The theoretical drift velocity per deme.
c_df_over_dc = parameter_df.loc[:, 'c_x_df_l_dc']  # Concentration*df/dc.
Vo_max = parameter_df.loc[1, 'Vo_max']  # The run speed.
# Timed rate of change of the amount of receptor protein bound.
Rtroc = vd_chemotaxis*Grad*c_df_over_dc  # This numpy vector is calculated from the above constant and pandas series.

# Values used for Norm_Vd_Mean_Data_Generator.
alpha = 100
Start_Angle = 90  # degrees
Angle = Start_Angle
diff = 1.16
dt = 0.1

# Plotting
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Position')
ax1.set_ylabel('Drift Velocity, Vd [µm/s]', color=color)
ax1.plot(vd_chemotaxis, color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlim([0, nl])
ax1.set_ylim([0, 5])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Concentration [µM]', color=color)  # we already handled the x-label with ax1
ax2.plot(xbias, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, np.max(xbias)])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Drift Velocity and Concentration')
plt.show()


# Specific Data Generator Implementation for Norm_Vd_Mean_Data_Generator
class NormMeanDataGenerator(BaseDataGenerator):
    """
    This class inherits the format of BaseDataGenerator and is used
    for the data generator 'Norm_Vd_Mean_Data_Generator'.
    Adapter around Norm_Vd_Mean_Data_Generator that matches the BaseDataGenerator ABC.
    """
    def __init__(self, *args, **kwargs):
        # Initialize with parameters specific to Norm_Vd_Mean_Data_Generator
        self.generator = Norm_Vd_Mean_Data_Generator(*args, **kwargs)

    def generate_data(self, alpha, max_iter):
        """
        :param: max_iter: The total number of iterations to run in parallel, (the number of data points generated).
        :return: The datapoints generator specific to this system's generator method.
        """
        return self.generator.simulate_bacterial_movement_cuda(alpha, max_iter)

    def sample(self, alpha, n):
        # BaseDataGenerator contract: sample(alpha, n) -> 1-D CPU tensor
        v = self.generate_data(alpha=alpha, max_iter=n)
        return (v.detach().cpu().reshape(-1)
                if hasattr(v, "detach") else torch.as_tensor(v).cpu().reshape(-1))


def test_sign_transitions(alpha_grid, expand_alpha=False, samples_per_alpha: float = 100):
    """
    Probe μ(α) on a coarse grid and report where μ(α) − v_d changes sign.
    :param alphas : list[float]
        Grid of α values to probe for μ(α). The variables of alpha used for inspection.
    :param expand_alpha : boolean
        Determine if the alpha window of [α_min, α_max] should be expanded.
    :param samples_per_alpha: float - How many samples to draw at each α for estimating μ(α) and s²(α).
        Larger values reduce the standard error of μ(α) and make sign changes more reliable.
        (Kept as float to match existing call sites; it’s used as a count when passed to the generator.)

    :return: α_min, α_max
    """
    # When running a script on CUDA, 'spawn' avoids CUDA multiprocessing issues.
    # Tells Python to start new processes with the 'spawn' method instead of the default (which can be 'fork' on Linux).
    # 'fork' clones the current process, which can cause CUDA context corruption, deadlocks, or crashes,
    # 'spawn' is safer for CUDA because it creates a fresh Python interpreter in each subprocess.
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # start_method may already be set

    # Build the generator reusing already-defined parameters at the start of the module.
    data_generator: BaseDataGenerator = NormMeanDataGenerator(Rtroc, Angle, Vo_max, DL, nl, deme_start, diff, dt)

    # Collect macro stats on the requested α grid 'from macro_stats import MacroStats'.
    # MacroStats.macro_observations will record μ(α), s²(α), n, etc., for later use.
    stats = MacroStats()

    #  Helper function to get python scalars from tensors/numbers.
    _get = lambda x: x.item() if hasattr(x, "item") else float(x)

    for a in alpha_grid:
        # Generate `samples_per_alpha` draws from the stochastic data generator at current α.
        # The generator may return a CUDA tensor, it needs to move to CPU before handing to
        # MacroStats to keep everything uniform.
        v = data_generator.generate_data(alpha=a, max_iter=samples_per_alpha)
        # Ensure samples live on CPU for MacroStats; no changes to MacroStats needed.
        samples = v.detach().cpu() if hasattr(v, "detach") else v
        # Update macro stats to append a record into stats.records.
        stats.macro_observations(alpha=float(a), samples=samples)

        # Find the newly added record (match α with a small tolerance).
        rec = next(r for r in stats.records if abs(_get(r["alpha"]) - float(a)) < 1e-6)

        # Print out an update for each alpha value's record. Only print fields that exist.
        print(f"Macro records from macro_observations for alpha = {int(a)}")
        msg = [f"[α={int(a)}]"]
        if "n" in rec:         msg.append(f"n={int(rec['n'])}")
        if "mu" in rec:        msg.append(f"μ={_get(rec['mu']):.6f}")
        if "std" in rec:       msg.append(f"std={_get(rec['std']):.6f}")
        if "var" in rec:       msg.append(f"var={_get(rec['var']):.6f}")
        if "stderr_ci" in rec: msg.append(f"stderr_ci={_get(rec['stderr_ci']):.6f}")
        print("  " + ", ".join(msg))


    # Print all recorded macro observations directly.
    print("\nMacro records from macro_observations:")
    for rec in stats.records:
        print(rec)

    # Quick linearity sanity: Δμ per +100 α (just prints)
    mus = [float(rec["mu"]) for rec in stats.records]
    deltas = [mus[i + 1] - mus[i] for i in range(len(mus) - 1)]
    print("\nΔμ over +100 α steps:", [f"{d:.3f}" for d in deltas])

    # Run the sign-transition detector.
    v_d = float(theoretical_val)
    transitions = None
    transitions = detect_sign_transitions(stats, v_d=v_d)

    print(f"\nDetected sign transitions in μ(α) - v_d for v_d = {v_d}")
    if not transitions:
        print("\n[No sign transition detected] μ(α) does not cross v_d within sampled α-range.")
        print("→ Proceeding without bracket constraints (free α search).")
        bracket = None  # No range restriction
        return None, None, stats, data_generator
    else:
        # Map detector’s names to the program’s local bracket names.
        # From here on, treat [a_min, a_max] as the *working* window for the ML stage.
        α_min, α_max = transitions[0]  # Select the first bracket
        alpha_vals = sorted(float(r["alpha"]) for r in stats.records)  # Get alpha from records to find min/max domain.
        print(f"Between α = {α_min} and α = {α_max}")

        # Resolution check: if bracket μ-span is too small, widen it
        mu_values = [float(r["mu"]) for r in stats.records]
        mu_range = max(mu_values) - min(mu_values)
        bracket_mu_range = abs(float(next(r for r in stats.records if float(r["alpha"]) == α_max)["mu"]) -
                               float(next(r for r in stats.records if float(r["alpha"]) == α_min)["mu"]))
        if mu_range > 0 and (bracket_mu_range / mu_range) < 0.15 and expand_alpha:
            alpha_span = α_max - α_min
            α_min = max(0.0, α_min - alpha_span)
            α_max = α_max + alpha_span
            print(f"\n[Resolution check] Coarse μ-range detected — "
                  f"expanded bracket to [{α_min}, {α_max}]")

        # Check if α_min is already in records.
        if not any(abs(float(r["alpha"]) - α_min) < 1e-6 for r in stats.records):
            # If missing, generate new data at α_min
            v_left = data_generator.generate_data(alpha=α_min, max_iter=samples_per_alpha)
            # Ensure samples are on CPU.
            samples_left = v_left.detach().cpu() if hasattr(v_left, "detach") else v_left
            # Store sample into records using macro_observations.
            stats.macro_observations(alpha=float(α_min), samples=samples_left)
        # Do the same for α_max.
        if not any(abs(float(r["alpha"]) - α_max) < 1e-6 for r in stats.records):
            v_right = data_generator.generate_data(alpha=α_max, max_iter=samples_per_alpha)
            samples_right = v_right.detach().cpu() if hasattr(v_right, "detach") else v_right
            stats.macro_observations(alpha=float(α_max), samples=samples_right)
        return (
            max(alpha_vals[0], α_min - (α_max - α_min)/2),
            min(alpha_vals[-1], α_max + (α_max - α_min)/2),
            stats,
            data_generator
        )


if __name__ == "__main__":

    # # Modify only deme 51
    # Rtroc[50] = 0.00563  # λ = 1 mm^-1 example

    for deme_start in range(90, 101):
        # Values used for training.
        # deme_start = 5  # Deme 1 is 0 for python.
        num_epochs = 100
        learning_rate = 2 / (100 * Rtroc[deme_start])

        theoretical_val = vd_chemotaxis[deme_start]

        # Provide the number of parallel iterations to run for sampling data points from the data generator algorithm.
        max_iter_start = 20000

        # The variables of alpha used for inspection.
        alphas = list(range(0, 700, 100))

        print(f"\nDEME NUMBER = {deme_start + 1}")
        print(f"\nRtroc = {Rtroc[deme_start]}")
        print(f"\nv_d = {theoretical_val}")

        a_min, a_max, stats, data_generator = test_sign_transitions(alpha_grid=alphas, expand_alpha=False, samples_per_alpha=max_iter_start)
        print(f"\n[a_min, a_max] = [{a_min}, {a_max}]")

        # slope_tol is the convergence threshold for m_k, to measure when ∣m(α_(k+1) )-m(α_k )∣ < slope_tol.
        # Inside finite_difference_tracker.residual_gate(), stderr_tol is a multiplier on the statistical confidence band:
        # bound = stderr_tol * (2 * s(α) / sqrt(n))
        # resid_ok = (abs(r_cent) <= bound)
        # The trade off being the Polyak–Ruppert slope “trusted” and prevents the γ-cap from being disabled unnecessarily,
        # but too large and it might mask real curvature and mis-estimate m_k.
        fd_tracker = FiniteDifferenceTracker(epsilon=1e-8, slope_tol=1e-5, stderr_tol=1.0)

        # Compute all forward differences between adjacent α’s currently in stats.records.
        fd_tracker.compute_all_differences(stats)

        # -------------------------------------------------------
        # Export coarse α sweep (before ML training)
        # -------------------------------------------------------
        def to_scalar(x):
            return x.item() if hasattr(x, "item") else x

        macro_records = [
            {k: to_scalar(v) for k, v in rec.items()}
            for rec in stats.records
        ]
        fd_records = [
            {k: to_scalar(v) for k, v in rec.items()}
            for rec in fd_tracker.fd_records
        ]
        macro_df = pd.DataFrame(macro_records)
        fd_df = pd.DataFrame(fd_records)
        with pd.ExcelWriter(
                f"Deme_{deme_start + 1}_alpha_w_const_Rtroc_data_raw.xlsx",
                engine="openpyxl") as writer:
            macro_df.to_excel(
                writer,
                sheet_name="MacroStats",
                index=False
            )
            fd_df.to_excel(
                writer,
                sheet_name="FiniteDifferences",
                index=False
            )
        print("Saved coarse alpha sweep.")
        # exit()

        print("\nFinite differences touching the detected bracket:")
        for fd in fd_tracker.fd_records:
            print(fd)

        # Prepare the ML estimator before training.
        bracket = (a_min, a_max)
        ddeme = Dynamic_Data_Evolving_Mean_Estimator(
            data_generator=data_generator,  # data_generator = Norm_Vd_Mean_Data_Generator
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            theoretical_val=float(theoretical_val),
            alpha=float(a_min),  # start on the left side of the bracket
            max_iter=max_iter_start,
            # eta_for_gamma_cap,
            # The ML algorithm iteratively refines the learning_rate and max_iter every number of step_size iterations.
            step_size=1,
            max_iter_limit=20000,
            max_iter_factor=2,
            learning_rate_gamma=0.5,
            stats=stats,
            fd_tracker=fd_tracker,
            bracket=bracket,
            use_gradient_override=True,
            alpha_decimals=2
        )

        print("\n[Init] Estimator constructed.")
        print(f"  bracket = {bracket}")

        # exit()

        print(f"  PR ready? {fd_tracker.linear_slope_ready}")
        #--------------------------------------------------------
        # Run the ML algorithm.
        alpha_opt, final_loss, alpha_history = ddeme.train()
        #--------------------------------------------------------
        alpha_opt = float(alpha_opt)

        # --------------------------------------------------------
        # Select the statistically best alpha from MacroStats.
        # --------------------------------------------------------

        eligible = []
        for r in stats.records:
            if r["n"] >= ddeme.max_iter_limit:
                err = abs(float(r["mu"]) - float(theoretical_val))
                eligible.append((err, r))

        print("\n===================================================")
        print("Eligible MacroStats candidates")
        print("===================================================")
        if eligible:
            # Sort by distance from theoretical mean.
            # Sort n based on most negative value so no not set reverse=True
            eligible.sort(key=lambda x: (-int(x[1]["n"]), x[0]))  # x[0] = |μ-v_d|, x[1] = n

            # Put the record of elegible candidates into a dataframe
            eligible_df = pd.DataFrame([
                {
                    "deme_index": int(deme_start + 1),
                    "v_d_target": float(theoretical_val),
                    "alpha_star": float(r["alpha"]),
                    "mu_final": float(r["mu"]),
                    "final_loss": float(final_loss),
                    "mk_slope_final": float(fd_tracker.m_k) if fd_tracker.m_k is not None else None,
                    "pr_slope_ready": bool(fd_tracker.linear_slope_ready),
                    "alpha_min": float(bracket[0]),
                    "alpha_max": float(bracket[1]),
                    "std_final": float(r["std"]),
                    "var_final": float(r["var"]),
                    "std_err_final": float(r["std"]) / np.sqrt(int(r["n"])),
                    "std_err_ci_final": float(r["stderr_ci"]),
                    "abs(mu_minus_vd)": err,
                    "learning_rate": float(ddeme.gamma_last),
                    "num_epochs": int(num_epochs),
                    "max_iter_final": int(r["n"]),
                }
                for err, r in eligible
            ])

            for rank, (err, r) in enumerate(eligible[:10], start=1):
                print(
                    f"{rank:2d}. "
                    f"alpha={float(r['alpha']):10.6f}  "
                    f"mu={float(r['mu']):10.6f}  "
                    f"|μ-v_d|={err:.6f}  "
                    f"CI={float(r['stderr_ci']):.6f}  "
                    f"n={int(r['n'])}"
                )
            # Pick the best candidate.
            closest_rec = eligible[0][1]
            if len(eligible) > 1:
                mu_prev = float(eligible[1][1]["mu"])
            else:
                mu_prev = float(closest_rec["mu"])
            alpha_opt = float(closest_rec["alpha"])
            print("\nSelected MacroStats alpha")
            print(
                f"alpha={alpha_opt:.6f}, "
                f"mu={float(closest_rec['mu']):.6f}, "
                f"|μ-v_d|={eligible[0][0]:.6f}, "
                f"CI={float(closest_rec['stderr_ci']):.6f}, "
                f"n={int(closest_rec['n'])}"
            )
        else:
            print("\nNo alpha reached max_iter_limit.")
            print("Falling back to optimizer result.")
            # Find the record closest to the final alpha_star
            alpha_opt_rounded = round(float(alpha_opt), 6)
            closest_rec = min(
                stats.records,
                key=lambda rec: abs(round(float(rec["alpha"]), 6) - alpha_opt_rounded)
            )

        # -------------------------------------------------------
        # Export results for each drift-velocity calculation
        # -------------------------------------------------------
        mu_final = float(closest_rec["mu"])
        std_final = float(closest_rec["std"])
        n_final = int(closest_rec["n"]) if "n" in closest_rec else int(ddeme.max_iter)
        std_err_final = std_final / np.sqrt(n_final)
        err_mu_final = abs(mu_final - float(theoretical_val))
        within_stderr = bool(err_mu_final <= std_err_final)

        # Create (or append to) a DataFrame summarizing run statistics
        output_record = {
            "deme_index": int(deme_start + 1),  # +1 because list starts at 0
            "v_d_target": float(theoretical_val),
            "alpha_star": float(alpha_opt),
            "mu_final": mu_final,
            "final_loss": float(final_loss),
            "mk_slope_final": float(fd_tracker.m_k) if fd_tracker.m_k is not None else None,
            "pr_slope_ready": bool(fd_tracker.linear_slope_ready),
            "alpha_min": float(bracket[0]),
            "alpha_max": float(bracket[1]),
            "std_final": std_final,
            "var_final": float(closest_rec["var"]),
            "std_err_final": std_err_final,
            "std_err_ci_final": float(closest_rec["stderr_ci"]),
            "abs(mu_minus_vd)": err_mu_final,
            "learning_rate": float(ddeme.gamma_last),
            "num_epochs": int(num_epochs),
            "max_iter_final": n_final
        }

        # Append new results to file (or create one)
        output_filename = f"deme_{int(deme_start + 1)}_alpha_results_summary.xlsx"

        try:
            existing_df = pd.read_excel(output_filename)
            updated_df = pd.concat([existing_df, pd.DataFrame([output_record])], ignore_index=True)
        except FileNotFoundError:
            updated_df = pd.DataFrame([output_record])

        # Write back to Excel
        with pd.ExcelWriter(output_filename, engine="openpyxl", mode="w") as writer:
            updated_df.to_excel(writer, index=False)
            eligible_df.to_excel(writer,
                                 sheet_name="Eligible Candidates",
                                 index=False)

        print(f"\n[Saved] Results written to {output_filename}")
