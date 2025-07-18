import numpy as np
import pandas as pd
import torch
import time
import logging
# Import the Tumble Angle Module
from Tumble_Angle import Angle_Generator
from Tumble_Angle import AngleGenerator_cuda

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NormMeanMatchDataGenerator:
    def __init__(self, Rtroc, alpha, Angle, Vo_max, DL, nl, deme_start, diff, dt, max_iter):
        self.Rtroc = Rtroc  # Time rate of change of the fractional amount of receptor (protein) bound.
        self.alpha = alpha  # The alpha constant to be found.
        self.Angle = Angle  # Bacterial orientation angle.
        self.Vo_max = Vo_max  # Run Speed
        self.DL = DL  # Deme length [µM]
        self.nl = nl  # Total number of demes.
        self.deme_start = deme_start  # Deme starting position.
        self.d = diff  # Diffusion constant
        self.dt = dt  # Time step
        self.pos = DL * deme_start  # Position variable [µM]
        self.pos_ini = DL * deme_start  # Starting position [µM]
        self.max_iter = max_iter  # Number of data points generated.

        logging.info(f"Initialized NormMeanMatchDataGenerator with: alpha={alpha}, Angle={Angle}, Vo_max={Vo_max}, "
                     f"DL={DL}, nl={nl}, deme_start={deme_start}, diff={diff}, dt={dt}, max_iter={max_iter}")

    def simulate_bacterial_movement_cpu(self):
        pos = self.pos
        Angle = self.Angle
        Calculated_Ave_Vd_Array = []
        angle_generator = Angle_Generator()
        iter = 1

        while iter < self.max_iter:
            for t in np.arange(1, 1000, self.dt):
                i = int(pos // self.DL)
                if (90 <= Angle) and (Angle < 270):
                    Ptum = self.dt * np.exp(-self.d + self.alpha * self.Rtroc[i])
                else:
                    Ptum = self.dt * np.exp(-self.d - self.alpha * self.Rtroc[i])

                R_rt = np.random.rand()
                if R_rt < Ptum:
                    Next_Angle = angle_generator.tumble_angle_function()
                    Angle = (Angle + Next_Angle) % 360
                else:
                    Dot_Product = np.cos(np.radians(Angle))
                    pos += self.dt * self.Vo_max * Dot_Product

                pos = max(0, min(pos, self.nl * self.DL))

                if pos == 0 or pos >= self.nl * self.DL:
                    break

            Calculated_Ave_Vd = (pos - self.pos_ini) / t
            Calculated_Ave_Vd_Array.append(Calculated_Ave_Vd)
            pos = self.DL * self.deme_start
            iter += 1

        return np.array(Calculated_Ave_Vd_Array)

    def simulate_bacterial_movement_cuda(self):
        # Initialize accumulators for sum of velocities and count
        total_velocity_sum = torch.tensor(0.0, device='cuda')
        total_count = torch.tensor(0, device='cuda')

        # Initializing total time steps.
        time_steps = torch.arange(0, 1000, self.dt, device='cuda')
        num_steps = time_steps.size(0)

        # Initialize tensors for pos and ang for all iterations.
        position = torch.zeros((num_steps, self.max_iter), device='cuda')
        ang = torch.zeros((num_steps, self.max_iter), device='cuda')

        # Starting Position.
        position[0, :] = torch.full((self.max_iter,), self.pos, device='cuda')

        # Apply model constraints
        position = torch.clamp(position, 0, self.nl * self.DL)

        # Starting Angle: All angles start at 'self.Angle'.
        ang[0, :] = torch.full((self.max_iter,), self.Angle, device='cuda').float()

        # Calculating the Timed Rate of Change Tensor
        Rtroc_tensor = torch.tensor(self.Rtroc, device='cuda')  # Convert Rtroc to tensor
        Rtroc_tensor_size = Rtroc_tensor.size(0)

        # Initialize the angle generator class to select from probability distribution.
        angle_generator = AngleGenerator_cuda()
        # Calculating the Total Angles needed for the Algorithm.
        total_angles = num_steps * self.max_iter
        Next_Angle = angle_generator.tumble_angle_function_cuda(size=total_angles).view(num_steps, self.max_iter)
        Next_Angle_Size = Next_Angle.size()

        # The random numbers to be used.
        R_rt = torch.rand(total_angles, device='cuda').view(num_steps, self.max_iter)
        R_rt_size = R_rt.size()

        # Initialize Ptum as a 2D tensor with dimensions [num_steps, max_iter].
        Ptum = torch.zeros((num_steps, self.max_iter), device='cuda')
        Ptum_size = Ptum.size()

        # Mask to track active bacteria
        active_mask = torch.ones(self.max_iter, dtype=torch.bool, device='cuda')

        for t_idx, t in enumerate(time_steps):
            # Tensors inside the for loop are vectorized and in parallel.
            if t_idx % 100 == 0:
                logging.info(f"Step {t_idx}/{num_steps}: Current time = {t.item()}")

            if active_mask.any():
                # Filter positions and angles using active_mask
                active_positions = position[t_idx, active_mask]

                # Calculate boundary_mask to identify bacteria reaching the end of the deme
                boundary_mask = active_positions >= (self.pos_ini + self.DL)

                # Handle bacteria that have reached the boundary.
                if boundary_mask.any():
                    # Calculate the distance traveled for these bacteria.
                    distance_travelled = active_positions[boundary_mask] - self.pos_ini
                    total_time = t

                    # Calculate the average velocity for these bacteria.
                    Calculated_Ave_Vd = distance_travelled / total_time

                    # Accumulate sum of velocities and count
                    total_velocity_sum += Calculated_Ave_Vd.sum()
                    total_count += Calculated_Ave_Vd.numel()

                # Update the active mask: remove bacteria that have reached the boundary
                active_mask[active_mask.clone()] = ~boundary_mask

                if active_mask.any():
                    # The .long() method ensures the tensor is of integer type, which is necessary for indexing.
                    i = (position[t_idx, active_mask] // self.DL).long()

                    # Direction for moving up or down gradient.
                    # The boolean is true if it is moving down the gradient.
                    direction_condition = (90 <= ang[t_idx, active_mask]) & (ang[t_idx, active_mask] < 270)

                    # Calculate Ptum using torch.where
                    Ptum[t_idx, active_mask] = torch.where(
                        direction_condition,
                        self.dt * torch.exp(-self.d + self.alpha * Rtroc_tensor[i]),
                        self.dt * torch.exp(-self.d - self.alpha * Rtroc_tensor[i])
                    ).float()  # Convert to float to match Ptum's dtype.

                    # Tumbling condition
                    tumble_mask = R_rt[t_idx, active_mask] < Ptum[t_idx, active_mask]
                    # Update angles based on tumbling condition.
                    # Condition: tumble_mask
                    # If the condition is True, update the angle with Next_Angle.
                    # If the condition is False, the bacteria does not tumble and the angle is left unchanged.
                    ang[t_idx, active_mask] = torch.where(
                        tumble_mask,
                        (ang[t_idx, active_mask] + Next_Angle[t_idx, active_mask]) % 360,
                        ang[t_idx, active_mask]
                    )

                    # Running condition
                    run_mask = ~tumble_mask
                    # Update position based on running condition
                    Dot_Product = torch.cos(ang[t_idx, active_mask] * (torch.pi / 180))  # Convert ang to radians manually
                    # Condition: run_mask
                    # If the condition is True, update the position with Vo_max*Dot_Product*dt.
                    # If the condition is False, the bacteria does not move and the position is left unchanged.
                    position[t_idx, active_mask] = torch.where(
                        run_mask,
                        position[t_idx, active_mask] + self.dt * self.Vo_max * Dot_Product,
                        position[t_idx, active_mask]
                    )

                    if t_idx < num_steps - 1:
                        # Set position and ang for the next time step
                        position[t_idx + 1, active_mask] = position[t_idx, active_mask]
                        ang[t_idx + 1, active_mask] = ang[t_idx, active_mask]

            # Remove finished bacteria from further calculations
            if not active_mask.any():
                logging.info("The break condition is met.")
                break

        # Final calculation for the remaining iterations.
        # Calculate boundary_mask to identify bacteria reaching the end of the deme
        final_remaining_positions = position[-1, active_mask]
        if active_mask.any():
            Calculated_Ave_Vd = (final_remaining_positions - self.pos_ini) / (time_steps[-1])
            if Calculated_Ave_Vd.numel() > 0:
                # Accumulate sum of velocities and count of the remaining iterations.
                total_velocity_sum += Calculated_Ave_Vd.sum()
                total_count += Calculated_Ave_Vd.numel()

        logging.info(f"total_velocity_sum = {total_velocity_sum}\ntotal_count = {total_count}")

        if total_count > 0:
            mean_results = total_velocity_sum / total_count
        else:
            mean_results = float('nan')  # Return NaN if no valid velocities were calculated
            return mean_results

        # Return the results
        return mean_results.item()

    def time_execution(self):
        # # Measure CPU execution time
        # logging.info("Starting CPU execution")
        # start_time = time.time()
        # cpu_data = self.simulate_bacterial_movement_cpu()
        # cpu_time = time.time() - start_time
        # logging.info(f"CPU execution time: {cpu_time:.4f} seconds")

        # Measure CUDA execution time
        logging.info("Starting CUDA execution")
        start_time = time.time()
        cuda_data = self.simulate_bacterial_movement_cuda()
        cuda_time = time.time() - start_time
        logging.info(f"CUDA execution time: {cuda_time:.4f} seconds")

        # Verify that both implementations produce similar results
        logging.info(f"CUDA data: {cuda_data}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  # Required for CUDA tensors

    # Parameters for the simulation
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

    # Generate a new tumble angle.
    angle_generator = Angle_Generator()
    next_tumble_angle = angle_generator.tumble_angle_function()
    Start_Angle = 90  # degrees
    Angle = Start_Angle

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
    Rtroc = vd_chemotaxis * Grad * c_df_over_dc  # This numpy vector is calculated from the above constant and pandas series.

    alpha = 400
    diff = 1.16
    dt = 0.1
    max_iter = 20000
    deme_start = 30

    # Initialize the data generator
    data_generator = NormMeanMatchDataGenerator(Rtroc, alpha, Angle, Vo_max, DL, nl, deme_start, diff, dt, max_iter)

    # torch.autograd provides classes and functions implementing automatic differentiation of arbitrary
    # scalar valued functions. It requires minimal changes to the existing code
    # - you only need to declare Tensor s for which gradients should be computed with the requires_grad=True keyword.
    # Autograd includes a profiler that lets you inspect the cost of different operators inside your model
    # - both on the CPU and GPU.
    # The record_shapes=True option is used to record the shapes of the tensors in the operations being profiled.
    # This can be helpful in understanding how tensor dimensions change throughout your operations and
    # identify potential inefficiencies related to tensor shape manipulations.

    # with torch.autograd.profiler.profile(record_shapes=True, use_cuda=True) as prof:
    #     with torch.autograd.profiler.record_function("simulate_bacterial_movement_cuda"):
    #         cuda_data = data_generator.simulate_bacterial_movement_cuda()
    #
    # print(prof.key_averages().table(sort_by="cuda_time_total"))

    # # Time the execution of both algorithms
    # data_generator.time_execution()

    print(data_generator.time_execution())
