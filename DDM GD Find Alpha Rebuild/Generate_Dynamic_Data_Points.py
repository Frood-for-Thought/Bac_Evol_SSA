# Import the Tumble Angle Module
from Tumble_Angle import AngleGenerator_cuda
import torch
import logging
import pandas as pd


class Norm_Vd_Mean_Data_Generator:
    def __init__(self, Rtroc, Angle, Vo_max, DL, nl, deme_start, diff, dt):
        self.Rtroc = Rtroc  # Time rate of change of the fractional amount of receptor (protein) bound.
        self.Angle = Angle  # Bacterial orientation angle.
        self.Vo_max = Vo_max  # Run Speed.
        self.DL = DL  # Deme length [µM].
        self.nl = nl  # Total number of demes.
        self.deme_start = deme_start  # Deme starting position.
        self.d = diff  # Diffusion constant.
        self.dt = dt  # Time step.
        self.pos = DL * deme_start  # Position variable [µM].
        self.pos_ini = DL * deme_start  # Starting position [µM].
        self.velocities = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logging.info(f"Initialized NormMeanMatchDataGenerator with: Angle={Angle}, Vo_max={Vo_max},"
                     f" DL={DL}, nl={nl}, deme_start={deme_start}, diff={diff}, dt={dt}, device={self.device}")

    def simulate_bacterial_movement_cuda(self, alpha, max_iter):
        """
        Generates the Dataset for one Epoch of the ML algorithm.
        :param: max_iter: The total number of iterations to run in parallel, (the number of data points generated).
        :return: A CUDA tensor containing a normalized distribution of velocity data points.
        """
        # Number of data points generated needs to remain in the method because the ML algorithm will adjust
        # this parameter through the training iterations as it repeatedly calls this method.
        # The max_iter gives a standard error of σ⁄√(max_iter) for the normalized data velocities.
        self.alpha = alpha
        self.max_iter = max_iter

        # Initialize accumulators for sum of velocities and count
        velocities = torch.zeros(self.max_iter, device=self.device)
        velocities_index = 0

        # Initializing total time steps.
        time_steps = torch.arange(0, 1000, self.dt, device=self.device)
        num_steps = time_steps.size(0)

        # Initialize tensors for pos and ang for all iterations.
        position = torch.zeros((num_steps, self.max_iter), device=self.device)
        ang = torch.zeros((num_steps, self.max_iter), device=self.device)

        # Starting Position.
        position[0, :] = torch.full((self.max_iter,), self.pos, device=self.device)

        # Apply model constraints
        position = torch.clamp(position, 0, self.nl * self.DL)

        # Starting Angle: All angles start at 'self.Angle'.
        ang[0, :] = torch.full((self.max_iter,), self.Angle, device=self.device).float()

        # Calculating the Timed Rate of Change Tensor
        Rtroc_tensor = torch.tensor(self.Rtroc, device=self.device)  # Convert Rtroc to tensor

        # Initialize the angle generator class to select from probability distribution.
        angle_generator = AngleGenerator_cuda()
        # Calculating the Total Angles needed for the Algorithm.
        total_angles = num_steps * self.max_iter
        Next_Angle = angle_generator.tumble_angle_function_cuda(size=total_angles).view(num_steps, self.max_iter)

        # The random numbers to be used.
        R_rt = torch.rand(total_angles, device=self.device).view(num_steps, self.max_iter)

        # Initialize Ptum as a 2D tensor with dimensions [num_steps, max_iter].
        Ptum = torch.zeros((num_steps, self.max_iter), dtype=torch.float32, device=self.device)

        # Mask to track active bacteria
        active_mask = torch.ones(self.max_iter, dtype=torch.bool, device=self.device)

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

                    # Append the values in the tensor onto another velocities tensor on 'cuda'.
                    for v in Calculated_Ave_Vd:
                        if velocities_index < self.max_iter:
                            velocities[velocities_index] = v  # Directly assign tensor value on the GPU.
                            velocities_index += 1
                        else:
                            break

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
                # Any remaining values are appended to the tensor 'velocities'.
                for v in Calculated_Ave_Vd:
                    if velocities_index < self.max_iter:
                        velocities[velocities_index] = v  # Directly assign tensor value on the GPU.
                        velocities_index += 1
                    else:
                        pass

        # The class attribute is set for the class when the function is called when using it to return.
        self.velocities = velocities
        return self.velocities  # Return the results

    def calculate_average_velocity(self):
        """
        Calculates the average velocity from the CUDA tensor of velocities.
        This method was only made for running this module as main.
        The velocities average for the machine learning module is calculated within that module.
        :param vel_tensor: Another tensor inserted if
        :param self.velocities: Tensor of velocities.
        :return: Average velocity as a float.
        """
        if self.velocities is None:
            raise ValueError("Velocities have not been computed yet. Call simulate_bacterial_movement_cuda first.")
        return torch.mean(self.velocities).item()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  # Required for CUDA tensors

    # Parameters for the simulation
    Start_Angle = 90  # degrees
    Angle = Start_Angle

    # The theoretical parameters have been pre-calculated to fit onto.
    nl = 101
    Grad = 0.000405  # µm^-1
    DL = 310  # µm
    input_parameters = '../input_parameters.xlsx'
    parameter_df = pd.read_excel(input_parameters)
    vd_chemotaxis = parameter_df.loc[:, 'drift_velocity']  # The theoretical drift velocity per deme.
    c_df_over_dc = parameter_df.loc[:, 'c_x_df_l_dc']  # Concentration*df/dc.
    Vo_max = parameter_df.loc[1, 'Vo_max']  # The run speed.
    # Timed rate of change of the amount of receptor protein bound.
    # This numpy vector is calculated from the above constant and pandas series.
    Rtroc = vd_chemotaxis * Grad * c_df_over_dc

    alpha = 700
    diff = 1.16
    dt = 0.1
    max_iter = 20000
    deme_start = 30

    # Initialize the data generator
    data_generator = Norm_Vd_Mean_Data_Generator(Rtroc, alpha, Angle, Vo_max, DL, nl, deme_start, diff, dt)

    # When calculating vel, the self.velocities attribute is set to the 'data_generator' object.
    # After self.velocities has been set to data_generator, use it to compute 'calculate_average_velocity()'.
    vel = data_generator.simulate_bacterial_movement_cuda(max_iter)
    print(vel)
    print(data_generator.calculate_average_velocity())
