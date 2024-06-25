In this experiment, the folder "Diffusion Model" has drift velocity set to zero and diffusion is just due to random diffusion instead of chemotaxis.

Bacteria are placed on the left hand side of the model, everything else is preserved.

The time is recorded to test how long it takes a mutant bacteria to fix into a deme.

The Diffusion Model Folder has the file Bac_Mig_Param_Class.m changed to remove alpha and Rtroc in the function obj:

classdef Bac_Mig_Param_Class
    properties
        i
        R_tum_up
        R_tum_down
    end
    methods
        function obj = Bac_Mig_Param_Class...
            (alpha, Rtroc, lnr, i)
            
            % Bacteria is moving down the gradient.
            R_tum_down = exp(-lnr);
            % Bacteria is moving up the gradient.
            R_tum_up = exp(-lnr);

            obj.i = i;
            obj.R_tum_up = R_tum_up;
            obj.R_tum_down = R_tum_down;
        end
    end
end

Now the model's diffusion is set to the same going up and down, (0.31349), which when plugged into the diffusion equation gives the theoretical value of
D = 112 micrometers^2/s.

D+ = D- = (Vo^2)/(4*dx*exp(-r))