% Class for Calculating Migration Parameters for Each Type of Bacteria
% at the Specific Deme Location

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