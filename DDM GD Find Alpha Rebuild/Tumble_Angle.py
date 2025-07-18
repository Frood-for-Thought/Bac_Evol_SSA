import numpy as np
from scipy.stats import rv_continuous
import torch
import time


class Tumble_Angle_Distribution(rv_continuous):
    def _pdf(self, x):
        """
        Override the _pdf describe bacterial tumble angle from "rv_continuous" in "scipy.stats".
        :param x: Angle in Rad.
        :return: P(x) = 0.5*(1+cos(x))*sin(x)
        """
        return 0.5 * (1 + np.cos(x)) * np.sin(x)


class Angle_Generator(Tumble_Angle_Distribution):
    def __init__(self):
        # Initialize the base class with angles from 0 to pi radians.
        super().__init__(a=0, b=np.pi, name='custom_angle_distribution')

    def tumble_angle_function(self):
        """
        Randomly select a new tumble angle from the probability distribution, P(x) = 0.5*(1+cos(x))*sin(x).
        :return: The repositioned tumble angle in degrees.
        """
        random_angle_rad = self.rvs()
        return np.degrees(random_angle_rad)


# Recalculating for CUDA
class CustomTumbleAngleDistribution:
    def __init__(self, a=0, b=np.pi, num_points=1000000):
        self.a = a
        self.b = b
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_points = num_points
        self.x_vals = torch.linspace(self.a, self.b, self.num_points, device=self.device)
        self.pdf_vals = self.pdf(self.x_vals)
        self.cdf_vals = self.compute_cdf()

    def pdf(self, x):
        """
        Probability Density Function for the tumble angle.
        :param x: Angle in Rad.
        :return: P(x) = 0.5 * (1 + cos(x)) * sin(x)
        """
        return 0.5 * (1 + torch.cos(x)) * torch.sin(x)

    def compute_cdf(self):
        """
        Compute the Cumulative Distribution Function (CDF).
        :return: The normalized cdf with bins = num_points.
        """
        cdf_vals = torch.cumsum(self.pdf_vals, dim=0)
        cdf_vals /= cdf_vals[-1].clone()  # Normalize the CDF
        return cdf_vals

    def interpolate_inverse_cdf(self, u):
        """
        A linear interpolation for the inverse CDF.
        :param u: Uniform random samples in [0, 1].
        :return: Interpolated samples corresponding to the CDF values.
        """
        # Find the indices in self.cdf_vals where the values of u would be inserted to maintain order.
        indices = torch.searchsorted(self.cdf_vals, u)
        # Clamp the indices to ensure they are within the valid range (1 to len(self.x_vals) - 1)
        indices = torch.clamp(indices, 1, len(self.x_vals) - 1)

        # Get the x values corresponding to the indices and their preceding values.
        x1 = self.x_vals[indices - 1]
        x2 = self.x_vals[indices]
        # Get the corresponding CDF values for the indices and their preceding values.
        y1 = self.cdf_vals[indices - 1]
        y2 = self.cdf_vals[indices]
        # Calculate the slope for linear interpolation.
        slope = (y2 - y1) / (x2 - x1)
        # Calculate the intercept for linear interpolation.
        intercept = y1 - slope * x1
        # Perform the linear interpolation to find the corresponding x values for the given u.
        return (u - intercept) / slope

    def rvs(self, size=1):
        """
        Random variate generation for the custom distribution using inverse transform sampling.
        :param size: Number of samples to generate.
        :return: Tensor of samples in radians.
        """
        uniform_samples = torch.rand(size, device=self.device)
        samples = self.interpolate_inverse_cdf(uniform_samples)
        return samples


class AngleGenerator_cuda(CustomTumbleAngleDistribution):
    def __init__(self):
        super().__init__(a=0, b=np.pi)

    def tumble_angle_function_cuda(self, size=1):
        random_angle_rad = self.rvs(size)
        return torch.rad2deg(random_angle_rad).to(self.device)


# Function to pre-generate a large batch of angles
def pre_generate_angles(total_angles):
    angle_generator = AngleGenerator_cuda()
    return angle_generator.tumble_angle_function_cuda(size=total_angles)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  # Required for CUDA tensors

    # Example usage
    num_loops = 10000  # Number of parallel loops
    num_samples_per_loop = 10000  # Number of samples per loop
    total_angles_needed = num_loops * num_samples_per_loop

    # Pre-generate angles
    start_time = time.time()
    pre_generated_angles = pre_generate_angles(total_angles_needed)
    end_time = time.time()
    print(f"Pre-generated {total_angles_needed} angles in {end_time - start_time:.4f} seconds "
          f"with mean {torch.mean(pre_generated_angles).item():.2f}")
