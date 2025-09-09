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
from Generate_Dynamic_Data_Points import Norm_Vd_Mean_Data_Generator
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
deme_start = 30

# Values used for training.
num_epochs = 100
learning_rate = 2 / (100 * Rtroc[deme_start])
theoretical_val = vd_chemotaxis[deme_start]
# Provide the number of parallel iterations to run for sampling data points from the data generator algorithm.
max_iter_start = 5000

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


def test_sign_transitions(samples_per_alpha: float = 100):
    """
    Probe μ(α) on a coarse grid and report where μ(α) − v_d changes sign.
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
    # The variables of alpha used for inspection.
    alphas = list(range(100, 1000, 100))

    #  Helper function to get python scalars from tensors/numbers.
    _get = lambda x: x.item() if hasattr(x, "item") else float(x)

    for a in alphas:
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
    v_d = float(vd_chemotaxis[deme_start])
    transitions = detect_sign_transitions(stats, v_d=v_d)

    print(f"\nDetected sign transitions in μ(α) - v_d for v_d = {v_d}")
    if not transitions:
        raise RuntimeError("No sign transitions detected.")
    else:
        # Map detector’s names to the program’s local bracket names.
        # From here on, treat [a_min, a_max] as the *working* window for the ML stage.
        α_min, α_max = transitions[0]  # Select the first bracket
        print(f"Between α = {α_min} and α = {α_max}")
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
        return α_min, α_max, stats, data_generator


if __name__ == "__main__":
    a_min, a_max, stats, data_generator = test_sign_transitions(samples_per_alpha=max_iter_start)
    print(f"\n[a_min, a_max] = [{a_min}, {a_max}]")

    # slope_tol is the convergence threshold for m_k, to measure when ∣m(α_(k+1) )-m(α_k )∣ < slope_tol.
    fd_tracker = FiniteDifferenceTracker(epsilon=1e-8, slope_tol=1e-5, stderr_tol=2.0)

    # Compute all forward differences between adjacent α’s currently in stats.records.
    fd_tracker.compute_all_differences(stats)

    print("\nFinite differences touching the detected bracket:")
    for fd in fd_tracker.fd_records:
        print(fd)

    # Prepare the ML estimator before training.
    bracket = (a_min, a_max)
    deme = Dynamic_Data_Evolving_Mean_Estimator(
        data_generator=data_generator,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        theoretical_val=float(theoretical_val),
        alpha=float(a_min),  # start on the left side of the bracket
        max_iter=max_iter_start,
        step_size=20,
        max_iter_limit=20000,
        max_iter_factor=2,
        learning_rate_gamma=0.7,
        stats=stats,
        fd_tracker=fd_tracker,
        bracket=bracket,
        use_gradient_override=True
    )

    print("\n[Init] Estimator constructed.")
    print(f"  bracket = {bracket}")
    print(f"  PR ready? {fd_tracker.linear_slope_ready}")

    alpha_opt, final_loss = deme.train()
    print(f"\n[Result] alpha* ≈ {alpha_opt:.6f}, final_loss = {final_loss:.6f}")
