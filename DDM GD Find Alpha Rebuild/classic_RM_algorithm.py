import numpy as np
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
num_epochs = 150
learning_rate = 2 / (100 * Rtroc[deme_start])
theoretical_val = vd_chemotaxis[deme_start]
# Provide the number of parallel iterations to run for sampling data points from the data generator algorithm.
max_iter_start = 2000

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


def robbins_monro(alpha0, v_d, gamma_schedule, max_iters, sample_fn, verbose=True):
    """
    Classical Robbins-Monro algorithm.
    - alpha0: initial guess
    - v_d: desired expectation (target)
    - gamma_schedule: function giving step size gamma_k at iteration k
    - sample_fn(alpha): returns a single noisy sample v_j(alpha)
    """
    alpha = alpha0
    history = [alpha]

    for k in range(1, max_iters + 1):
        v_sample = sample_fn(alpha)               # one noisy sample
        mu = v_sample                             # single-sample mean
        gamma = gamma_schedule(k)                 # e.g. gamma0 / k
        alpha = alpha - gamma * (mu - v_d)        # Robbins-Monro update
        history.append(alpha)

        if verbose:
            print(f"Iter {k:3d}: alpha = {alpha:.6f}, sample = {v_sample:.6f}, "
                  f"step size = {gamma:.6f}, target = {v_d:.6f}")

    return np.array(history)


# Initialize data generator (adapter class you defined)
gen: BaseDataGenerator = NormMeanDataGenerator(Rtroc, Angle, Vo_max, DL, nl, deme_start, diff, dt)

alpha = 375

# Robbins–Monro run
rm_history = robbins_monro(
    alpha0=alpha,
    v_d=theoretical_val,
    gamma_schedule=lambda k: learning_rate / k,   # decaying step-size
    max_iters=num_epochs,
    sample_fn=lambda a: gen.sample(a, 1).item(),
    verbose=True
)

# Convert to numpy array for plotting
rm_history = np.array(rm_history)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(rm_history, label="Robbins–Monro estimate α_k")
plt.axhline(y=alpha, color="red", linestyle="--", label="Initial α")
plt.axhline(y=theoretical_val, color="green", linestyle="--", label="Target α⋆ (theoretical)")
plt.xlabel("Iteration")
plt.ylabel("α value")
plt.title("Robbins–Monro Convergence")
plt.legend()
plt.grid(True)
plt.show()