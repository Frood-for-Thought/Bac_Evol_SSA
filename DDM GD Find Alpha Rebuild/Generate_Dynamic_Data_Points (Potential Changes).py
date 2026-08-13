# Import the Tumble Angle Module
from Tumble_Angle import AngleGenerator_cuda
import torch
import logging
import pandas as pd
import time

def cuda_timer(label):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    def stop():
        torch.cuda.synchronize()
        print(f"[TIMING] {label}: {time.perf_counter() - t0:.6f} s")
    return stop


class SectionTimer:
    def __init__(self):
        self._starts = {}
        self.totals = {}

    def start(self, label):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._starts[label] = time.perf_counter()

    def stop(self, label):
        if label not in self._starts:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._starts.pop(label)
        self.totals[label] = self.totals.get(label, 0.0) + dt

    def print_summary(self):
        print("[TIMING SUMMARY]")
        for k, v in self.totals.items():
            print(f"[TIMING] {k}: {v:.6f} s")


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
        self.pos = DL * 50  # Position variable [µM].
        self.pos_ini = DL * 50  # Starting position [µM].
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

        # # Initialize tensors for pos and ang for all iterations.
        # position = torch.zeros((num_steps, self.max_iter), device=self.device)
        # ang = torch.zeros((num_steps, self.max_iter), device=self.device)

        # Starting Position.
        position = torch.full((self.max_iter,), self.pos, device=self.device, dtype=torch.float32)

        # Apply model constraints
        position = torch.clamp(position, 0, self.nl * self.DL)

        # Starting Angle: All angles start at 'self.Angle'.
        ang = torch.full((self.max_iter,), self.Angle, device=self.device, dtype=torch.float32,)

        # Calculating the Timed Rate of Change Tensor
        Rtroc_tensor = torch.tensor(self.Rtroc, device=self.device)  # Convert Rtroc to tensor
        local_Rtroc = Rtroc_tensor[self.deme_start]

        # Initialize the angle generator class to select from probability distribution.
        angle_generator = AngleGenerator_cuda()
        timers = SectionTimer()
        # Calculating the Total Angles needed for the Algorithm.
        total_angles = num_steps * self.max_iter
        timers.start("angle_generation")
        Next_Angle = angle_generator.tumble_angle_function_cuda(size=total_angles).view(num_steps, self.max_iter)
        timers.stop("angle_generation")

        # The random numbers to be used.
        timers.start("R_rt_generation")
        R_rt = torch.rand(total_angles, device=self.device).view(num_steps, self.max_iter)
        timers.stop("R_rt_generation")

        # Initialize Ptum as a 2D tensor with dimensions [num_steps, max_iter].
        Ptum = torch.zeros(self.max_iter, dtype=torch.float32, device=self.device)

        # Mask to track active bacteria
        active_mask = torch.ones(self.max_iter, dtype=torch.bool, device=self.device)

        # Track how long each bacterium has been active
        start_times = torch.zeros(self.max_iter, device=self.device)

        loop_start = timers.start("main_loop")

        for t_idx, t in enumerate(time_steps):
            # Tensors inside the for loop are vectorized and in parallel.
            if t_idx % 100 == 0:
                logging.info(f"Step {t_idx}/{num_steps}: Current time = {t.item()}")

            if active_mask.any():
                # Full-tensor boundary detection for active bacteria
                boundary_full = position >= (self.pos_ini + self.DL * 10)
                mask_boundary = active_mask & boundary_full

                # Handle bacteria that have reached the boundary.
                if mask_boundary.any():
                    # Calculate the distance traveled for these bacteria.
                    distance_travelled = position[mask_boundary] - self.pos_ini
                    # Use per-bacterium elapsed time
                    total_time = torch.clamp(t - start_times[mask_boundary], min=self.dt)

                    # Calculate the average velocity for these bacteria.
                    Calculated_Ave_Vd = distance_travelled / total_time

                    timers.start("ave_v_append_loop")
                    # Vectorized append: write a slice of velocities
                    k = Calculated_Ave_Vd.numel()
                    avail = self.max_iter - velocities_index
                    if k > 0 and avail > 0:
                        take = int(min(k, avail))
                        velocities[velocities_index:velocities_index + take] = Calculated_Ave_Vd[:take]
                        velocities_index += take
                    timers.stop("ave_v_append_loop")

                # Update the active mask: remove bacteria that have reached the boundary
                active_mask = active_mask & (~boundary_full)

                # Reset timers for newly inactive bacteria to avoid reuse
                start_times[~active_mask] = 0.0

                if active_mask.any():
                    # Direction for moving up or down gradient on full tensor
                    direction_condition = (90 <= ang) & (ang < 270)

#-----------------------------------------------------------------------------------------------------------
                    # THIS WAS THE NON-POISSON BROWN AND BERG PROCESS
                    # Calculate Ptum using torch.where and time each sub-operation
                    timers.start("Ptum_and_updates")

                    timers.start("Ptum_compute")
                    Ptum_full = torch.where(
                        direction_condition,
                        self.dt * torch.exp(-self.d + self.alpha * local_Rtroc),
                        self.dt * torch.exp(-self.d - self.alpha * local_Rtroc)
                    ).float()
                    timers.stop("Ptum_compute")

                    # # THIS IS THE POISSON PROCESS HAZARD FORM.
                    # r_plus  = torch.clamp(self.d + self.alpha * local_Rtroc, min=1e-8)
                    # r_minus = torch.clamp(self.d - self.alpha * local_Rtroc, min=1e-8)
                    #
                    # Ptum_full = torch.where(
                    #     direction_condition,
                    #     1 - torch.exp(-r_plus * self.dt),
                    #     1 - torch.exp(-r_minus * self.dt)
                    # ).float()
#----------------------------------------------------------------------------------------------------------

                    # Tumbling condition
                    timers.start("tumble_mask")
                    tumble_mask_full = R_rt[t_idx] < Ptum_full
                    timers.stop("tumble_mask")

                    # Update angles based on tumbling condition.
                    # If True, update the angle with Next_Angle; otherwise unchanged.
                    timers.start("angle_update")
                    ang = torch.where(active_mask & tumble_mask_full,
                                      (ang + Next_Angle[t_idx]) % 360,
                                      ang)
                    timers.stop("angle_update")

                    # Running condition + dot product
                    timers.start("dot_product")
                    run_mask_full = ~tumble_mask_full
                    Dot_Product_full = torch.cos(ang * (torch.pi / 180))  # Convert ang to radians manually
                    timers.stop("dot_product")

                    timers.start("position_update")
                    position = torch.where(active_mask & run_mask_full,
                                           position + self.dt * self.Vo_max * Dot_Product_full,
                                           position)
                    timers.stop("position_update")

                    if t_idx < num_steps - 1:
                        # Set position and ang for the next time step (no-op kept for clarity)
                        position = position
                        ang = ang
                    timers.stop("Ptum_and_updates")

            # Remove finished bacteria from further calculations
            if not active_mask.any():
                logging.info("The break condition is met.")
                break

        timers.stop("main_loop")
        timers.print_summary()

        # Final calculation for the remaining iterations.
        # Calculate boundary_mask to identify bacteria reaching the end of the deme
        final_remaining_positions = position[active_mask]
        if active_mask.any():
            Calculated_Ave_Vd = (final_remaining_positions - self.pos_ini) / (time_steps[-1])
            if Calculated_Ave_Vd.numel() > 0:
                timers.start("final_append")
                k = Calculated_Ave_Vd.numel()
                avail = self.max_iter - velocities_index
                if k > 0 and avail > 0:
                    take = int(min(k, avail))
                    velocities[velocities_index:velocities_index + take] = Calculated_Ave_Vd[:take]
                    velocities_index += take
                timers.stop("final_append")

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
    input_parameters = 'input_parameters.xlsx'
    parameter_df = pd.read_excel(input_parameters)
    vd_chemotaxis = parameter_df.loc[:, 'drift_velocity']  # The theoretical drift velocity per deme.
    c_df_over_dc = parameter_df.loc[:, 'c_x_df_l_dc']  # Concentration*df/dc.
    Vo_max = parameter_df.loc[1, 'Vo_max']  # The run speed.
    # Timed rate of change of the amount of receptor protein bound.
    # This numpy vector is calculated from the above constant and pandas series.
    Rtroc = vd_chemotaxis * Grad * c_df_over_dc

    alpha = 500
    diff = 1.16
    dt = 0.1
    max_iter = 20000
    deme_start = 51

    # Initialize the data generator
    data_generator = Norm_Vd_Mean_Data_Generator(Rtroc, Angle, Vo_max, DL, nl, deme_start, diff, dt)

    # When calculating vel, the self.velocities attribute is set to the 'data_generator' object.
    # After self.velocities has been set to data_generator, use it to compute 'calculate_average_velocity()'.
    vel = data_generator.simulate_bacterial_movement_cuda(alpha, max_iter)

    print(vel)
    print(data_generator.calculate_average_velocity())
