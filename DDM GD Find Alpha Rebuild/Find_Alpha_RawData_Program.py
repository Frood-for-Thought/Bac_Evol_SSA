import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
# Import the Tumble Angle Module
from Tumble_Angle import AngleGenerator_cuda
# Initialize the angle generator class to select from probability distribution.
angle_generator = AngleGenerator_cuda()
from Generate_Dynamic_Data_Points import Norm_Vd_Mean_Data_Generator
from Calc_Alpha_ML_Function import BaseDataGenerator, Dynamic_Data_Evolving_Mean_Estimator


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
input_parameters = '../input_parameters.xlsx'
parameter_df = pd.read_excel(input_parameters)
vd_chemotaxis = parameter_df.loc[:, 'drift_velocity']  # The theoretical drift velocity per deme.
c_df_over_dc = parameter_df.loc[:, 'c_x_df_l_dc']  # Concentration*df/dc.
Vo_max = parameter_df.loc[1, 'Vo_max']  # The run speed.
# Timed rate of change of the amount of receptor protein bound.
Rtroc = vd_chemotaxis*Grad*c_df_over_dc  # This numpy vector is calculated from the above constant and pandas series.

# Values used for Norm_Vd_Mean_Data_Generator.
alpha = 500
Start_Angle = 90  # degrees
Angle = Start_Angle
diff = 1.16
dt = 0.1
deme_start = 30

# Values used for Dynamic_Data_Evolving_Mean_Estimator.
num_epochs = 100
learning_rate = 2 / (100 * Rtroc[deme_start])
theoretical_val = vd_chemotaxis[deme_start]
max_iter = 1000

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


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  # Required for CUDA tensors

    # Initialize the data generator.
    data_generator = NormMeanDataGenerator(Rtroc, Angle, Vo_max, DL, nl, deme_start, diff, dt)

    optim_alpha, loss_value = Dynamic_Data_Evolving_Mean_Estimator(data_generator, num_epochs, learning_rate,
                                                                   theoretical_val, alpha, max_iter).train()

    print(optim_alpha, loss_value)

# Pos_Alpha_Array = []
# Ni = 2
# Nj = 100
# for deme_start in range(Ni, Nj + 1):
#     alpha_start = 500
#
#     # Call the ML function to find the most optimum alpha value.
#     Pos_Alpha_Array = Calc_Alpha_ML_Function(
#         Rtroc, F, vd_chemotaxis, alpha_start, deme_start, nl, Angle, Vo_max, xbias, DL, Pos_Alpha_Array)
