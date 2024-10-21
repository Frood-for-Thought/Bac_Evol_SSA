function [i, Location_Mig_Probability, Find_Location,...
    WT_Selected,Mut1_Selected,Mut2_Selected,Mut3_Selected,Mut4_Selected...
    ,pos_selected]...
    = Select_Location_Bacteria_Mig_Function(...
    P_deme_Mig, R_Mig, Find_Location, nl, i, Location_Mig_Probability, ...
    P_deme_Mig_WT, P_deme_Mig_Mut1,P_deme_Mig_Mut2, P_deme_Mig_Mut3, P_deme_Mig_Mut4)
    if Find_Location > 0
        % FIND MIGRATION LOCATION 
        P_deme_Mig = P_deme_Mig/R_Mig; % Normalized mig probability for each deme
                                    % to find where in the system migration occurs
        P_il_new_location = randperm(length(P_deme_Mig));
        New_P_deme_Order = P_deme_Mig(:, P_il_new_location);
        P_deme_wthPosition_Random = [New_P_deme_Order; P_il_new_location];
        r3 = rand();
        Position_Found = 0;
        while Position_Found < 1
            for i = 1:nl % select position loop
                if r3 < P_deme_wthPosition_Random(1,i)
                    Location_Mig_Probability = P_deme_wthPosition_Random(1,i);
                    Mig_Location = P_deme_wthPosition_Random(2,i);
                    Position_Found = 1;
                end
            end
            r3 = rand(); % if no location is found
        end
        i = Mig_Location;
    else
        % Calculate the migration rate percentage for each bacteria at location i
        P_bac = [P_deme_Mig_WT(i), P_deme_Mig_Mut1(i), P_deme_Mig_Mut2(i), P_deme_Mig_Mut3(i), P_deme_Mig_Mut4(i)];
        mN_Normal = sum(P_bac);
        % After recalculating growth rate the rate may end up being zero
        % even though originally the P_deme here was non-zero
        if mN_Normal == 0
            mN_Normal = 1;
        end
        % If for some reason the migration rate percentage is negative.
        if (Location_Mig_Probability > 0) && (sum(P_bac) <= 0)
            Find_Location = 1;
            return % The while loop will cycle to a new probability.
        end
        P_bac = P_bac/mN_Normal;
        P_bac_new_location = randperm(length(P_bac));
        % Rearrange the Normalized mN values
        New_P_bac_Order = P_bac(:, P_bac_new_location);
        P_bac_wthPosition_Random = [New_P_bac_Order; P_bac_new_location];
        r4 = rand();
        Bacteria_Found = 0;
        while Bacteria_Found < 1
            for Bac_Type = 1:length(P_bac)
                      % P_bac_wthPosition_Random(Row 1, Column Bac_Type)
                if sum(P_bac_wthPosition_Random(1,:)) == 0
                    continue
                end
                if r4 < P_bac_wthPosition_Random(1,Bac_Type)
                    Bacteria_Type_Probability = P_bac_wthPosition_Random(1,Bac_Type);
                    Bacteria_Type_Selected = P_bac_wthPosition_Random(2,Bac_Type);
                    Bacteria_Found = 1;
                end
            end
            r4 = rand();
        end

        WT_Selected = 0; % Wild Type Bacteria Selected Condition
        Mut1_Selected = 0; % Mutant 1 Bacteria Selected Condition
        Mut2_Selected = 0; % Mutant 2 Bacteria Selected Condition
        Mut3_Selected = 0; % Mutant 3 Bacteria Selected Condition
        Mut4_Selected = 0; % Mutant 4 Bacteria Selected Condition
        if Bacteria_Type_Selected == 1
            WT_Selected = 1;
        elseif Bacteria_Type_Selected == 2
            Mut1_Selected = 1;
        elseif Bacteria_Type_Selected == 3
            Mut2_Selected = 1;
        elseif Bacteria_Type_Selected == 4
            Mut3_Selected = 1;
        elseif Bacteria_Type_Selected == 5
            Mut4_Selected = 1;
        end
        pos_selected = 1;
    end % if Find_Location > 0
end