%% Function for Calculating Rates Bacterial Action During 
%  Initialization Loop with Each Type of Bacteria

function [Growth_Rate,R_mig_deme] = Initialize_Rates_Function(il,nl,CC,ntot_right,ntot_left,Food_Factor,...
	FF,MewMax,K,MIC,Fit,Max_Monod,Cipro,Pop_Factor,...
    DL, Vo_max,Rand_Part_Mtx_El, All_Prt_Ran_Order, R_a, ...
    R_Tum_Up_Mtx, R_Tum_Down_Mtx)

    % Fixing Pharma Error at end.
    % The pharma function starts to couteract the monod eqn.
    if (il == 99) | (il == 100) | (il == 101)
        Cipro = 8;
    end

    %% Growth_Rate_Function
    [Growth_Rate] = Growth_Rate_Function(Food_Factor,FF,...
        MewMax,K,MIC,Fit,Max_Monod,Cipro,Pop_Factor);
    
    %% Calculating Net Migration with Cipro repellent equation
    % If there are still some bacteria that are able to move then
    % calculate the rate of migration.
    R_mig_deme = 0;
    if Rand_Part_Mtx_El <= length(All_Prt_Ran_Order)
        % Checks the population values from Main_Program
        if ntot_right < CC | ntot_left < CC
            R_tum_up = NaN;
            R_tum_down = NaN;
            % Find the object label for position il for rates.
            if ntot_right < CC
                R_tum_up = R_Tum_Up_Mtx(il);
            end
            if ntot_left < CC
                R_tum_down = R_Tum_Down_Mtx(il);
            end
            % Multiply the speed by the repulsion equation.
            Vo_max = Vo_max*R_a;
            
            calc_mig_rate = 1; % Used as a boolean to calc. migration rate.
            
            [R_mig_deme] = Calculate_Position_Mig_Function...
                (R_tum_up, R_tum_down, Vo_max, DL, calc_mig_rate, il);
        end
    end % if Rand_Part_Mtx_El <= length(All_Prt_Ran_Order)
end