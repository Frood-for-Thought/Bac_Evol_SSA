%% Function for calculating migration rate

function [R_mig_deme, Mig_Occured, Mig_Right, Mig_Left]...
    = Calculate_Position_Mig_Function...
        (R_tum_up, R_tum_down, Vo_max, DL, calc_mig_rate, Mig_Occured, Mig_Right, Mig_Left)

    % This finds the migration rate of travelling from one deme to another
    if calc_mig_rate > 0
        % The rates calculated for migrating on either direction.
        R_mig_deme_up = 0;
        R_mig_deme_down = 0;
        if R_tum_up > 0
            R_mig_deme_up = (Vo_max)^2/(4*(R_tum_up)*(DL)^2); % [s^-1]
        end
        if R_tum_down > 0
            R_mig_deme_down = (Vo_max)^2/(4*(R_tum_down)*(DL)^2); % [s^-1]
        end
        % The rates are then summed up for the deme.
        R_mig_deme = R_mig_deme_up + R_mig_deme_down;
        % N bacteria
        % N/DL^2 bacterial density
        % (N/DL^2)*area_of_exit
        % Using assumption by Allen Supplemental Paper:
        %       Rate = R_mig_deme*(channel_width/deme_size)
        %       w/DL = 10/310 ~ 0.03
        R_mig_deme = R_mig_deme*0.03;
    else
        % The rates calculated for migrating on either direction.
        R_mig_deme_up = 0;
        R_mig_deme_down = 0;
        if R_tum_up > 0
            R_mig_deme_up = (Vo_max)^2/(2*(R_tum_up)*(DL^2)); % [s^-1]
        end
        if R_tum_down > 0
            R_mig_deme_down = (Vo_max)^2/(2*(R_tum_down)*(DL^2)); % [s^-1]
        end
        % The rates are then summed up for the deme.
        R_mig_deme = R_mig_deme_up + R_mig_deme_down;

        dir = rand();
        % The normailzed probabilities to determine direction.
        if (dir >= 0) && (dir < abs(R_mig_deme_up/R_mig_deme)) % Right.
            Mig_Occured = Mig_Occured + 1;
            Mig_Right = Mig_Right + 1;
        elseif (dir >= abs(R_mig_deme_up/R_mig_deme)) && (dir < abs((R_mig_deme_up + R_mig_deme_down)/R_mig_deme)) % Left
            Mig_Occured = Mig_Occured + 1;
            Mig_Left = Mig_Left + 1;
        end
    end % if calc_mig_rate > 0
end