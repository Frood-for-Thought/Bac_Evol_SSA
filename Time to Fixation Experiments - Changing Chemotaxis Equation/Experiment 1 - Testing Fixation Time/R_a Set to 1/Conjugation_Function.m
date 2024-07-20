
function [cjN,Combine_Mtx_F_Norm] = Conjugation_Function(Start_Conj,Bacteria_Num_Mtx, ...
                    Conj_Rate,WT_FF,Mut_1_FF,Mut_2_FF,Mut_3_FF,MIC,...
                        Fit_WT,Fit_m1,Fit_m2,Fit_m3,Cipro,Conjugation_Selected)
    
%     field = 'a';
    row = 0;
    Combine_Mtx = zeros();
    for i = 1:length(Bacteria_Num_Mtx)
        for j = 1:length(Bacteria_Num_Mtx)
            if (i ~= j)
                row = row + 1;
                Combine_Mtx(row,1) = i;
                Combine_Mtx(row,2) = j;
                Combine_Mtx(row,3) = Bacteria_Num_Mtx(i);
                Combine_Mtx(row,4) = Bacteria_Num_Mtx(j);
            end
        end
    end
%     s = struct(field,value);
%     s.a

    Pair_Mtx = zeros();
    Row = 1;
    for Every_Second_Row = 1:2:size(Combine_Mtx,1)
        Pair_Mtx(Row,1:size(Combine_Mtx,2)) = Combine_Mtx(Every_Second_Row,1:size(Combine_Mtx,2));
        Row = Row + 1;
    end
    for row = 1:size(Pair_Mtx,1)
        if Pair_Mtx(row,3) < Start_Conj
            Pair_Mtx(row,3) = 0;
        end
        if Pair_Mtx(row,4) < Start_Conj
            Pair_Mtx(row,4) = 0;
        end
    end

    % Calculate the conjugation rate at position "il"
    Conj_Rate_Mtx = zeros();
    for row = 1:size(Pair_Mtx,1)
        donor_cells = Pair_Mtx(row, 3);
        recipient_cells = Pair_Mtx(row, 4);
        if donor_cells == 0 || recipient_cells == 0
            % If either population is zero, the conjugation rate is zero
            Conj_Rate_Mtx(row, 1) = 0;
        else
            % Calculate the conjugation rate for the current row
            limiting_population = min(Pair_Mtx(row,3:4));
            Conj_Rate_Mtx(row, 1) = Conj_Rate * ((donor_cells * recipient_cells)/limiting_population);
        end
    end

    % Since demes are in units of 10, cjN growth rate is 10 times more likely
    % If there is only one type of bacteria, then conjugation is 0
    cjN = sum(Conj_Rate_Mtx)*10;

    %% Make variable Conjugation_Selected, when it gets selected the if
    % statement can activate here to avoid unnecessary calculations
    if Conjugation_Selected > 0
    
    % This calculates the fitness each bacteria have at position = il
    if (WT_FF > 0) && (Fit_WT > 0)
        Total_WT_Fit = WT_FF*heaviside(Fit_WT*MIC - Cipro)+ 0.1*(WT_FF);
        % heaviside(Cipro - Fit_WT*MIC)
    else
        Total_WT_Fit = 0.01;
    end
    if (Mut_1_FF > 0) && (Fit_m1 > 0)
        Total_Mut1_Fit = Mut_1_FF*heaviside(Fit_m1*MIC - Cipro) + 0.1*(Mut_1_FF);
    else
        Total_Mut1_Fit = 0.01;
    end
    if (Mut_2_FF > 0) && (Fit_m2 > 0)
        Total_Mut2_Fit = Mut_2_FF*heaviside(Fit_m2*MIC - Cipro) + 0.1*(Mut_2_FF);
    else
        Total_Mut2_Fit = 0.01;
    end
    if (Mut_3_FF > 0) && (Fit_m3 > 0)
        Total_Mut3_Fit = Mut_3_FF*heaviside(Fit_m3*MIC - Cipro) + 0.1*(Mut_3_FF);
    else
        Total_Mut3_Fit = 0.01;
    end
    
    % Record the total fitness of each bacteria in the matrix, Fit_Val_Mtx,
    % and then calculate the chance in fitness in the matrix, Fit_Diff
    Fit_Mtx = [Total_WT_Fit Total_Mut1_Fit Total_Mut2_Fit Total_Mut3_Fit];
    Fit_Val_Mtx = zeros();
    for row = 1:size(Combine_Mtx,1)
        if Combine_Mtx(row,1) == 1
            Fit_Val_Mtx(row,1) = Total_WT_Fit;
        elseif Combine_Mtx(row,1) == 2
            Fit_Val_Mtx(row,1) = Total_Mut1_Fit;
        elseif Combine_Mtx(row,1) == 3
            Fit_Val_Mtx(row,1) = Total_Mut2_Fit;
        elseif Combine_Mtx(row,1) == 4
            Fit_Val_Mtx(row,1) = Total_Mut3_Fit;
        end
        if Combine_Mtx(row,2) == 1
            Fit_Val_Mtx(row,2) = Total_WT_Fit;
        elseif Combine_Mtx(row,2) == 2
            Fit_Val_Mtx(row,2) = Total_Mut1_Fit;
        elseif Combine_Mtx(row,2) == 3
            Fit_Val_Mtx(row,2) = Total_Mut2_Fit;
        elseif Combine_Mtx(row,2) == 4
            Fit_Val_Mtx(row,2) = Total_Mut3_Fit;
        end
    end
    Fit_Diff = zeros();
    for row = 1:size(Combine_Mtx,1)
        Fit_Diff(row,1) = Fit_Val_Mtx(row,2) - Fit_Val_Mtx(row,1);
    end
    
    % Combine_Mtx_F records the types of bacteria in the first two rows,
    % then the total population of each bacteria at position il in the
    % second two rows, and also the change in fitness which would occur if
    % bacteria in row 1 were to become bacteria in row 2
    Combine_Mtx_F = [Combine_Mtx Fit_Val_Mtx Fit_Diff];
    
    % The raising and lowering ladder operators
    L_Plus = [0 0 0 0; 1 0 0 0; 1 1 0 0; 1 1 1 0];
    L_Minus = [0 1 1 1; 0 0 1 1; 0 0 0 1; 0 0 0 0];
    % The vectors for each type of bacteria
       X_Vec = [1; 0; 0; 0];
    Mut1_Vec = [0; 1; 0; 0];
    Mut2_Vec = [0; 0; 1; 0];
    Mut3_Vec = [0; 0; 0; 1];
    % Norm records if a <Bra|L|Ket> system is normalizable or orthogonal
    Norm = zeros();
    for row = 1:size(Combine_Mtx_F,1)
        % This assigns a Bra matrix to each type of bacteria
        if Combine_Mtx_F(row,1) == 1
            Bra = transpose(X_Vec);
        elseif Combine_Mtx_F(row,1) == 2
            Bra = transpose(Mut1_Vec);
        elseif Combine_Mtx_F(row,1) == 3
            Bra = transpose(Mut2_Vec);
        elseif Combine_Mtx_F(row,1) == 4
            Bra = transpose(Mut3_Vec);
        end
        % This assigns a Ket matrix to each type of bacteria
        if Combine_Mtx_F(row,2) == 1
            Ket = X_Vec;
        elseif Combine_Mtx_F(row,2) == 2
            Ket = Mut1_Vec;
        elseif Combine_Mtx_F(row,2) == 3
            Ket = Mut2_Vec;
        elseif Combine_Mtx_F(row,2) == 4
            Ket = Mut3_Vec;
        end
        % The fitness will decrease from Bra to Ket and the epigenetic
        % pathway of the Bra is one step LESS than the Ket, so genetic
        % information is transferred to the less fit bacteria
        if (Combine_Mtx_F(row,7) <= 0)&&(Combine_Mtx_F(row,1)<Combine_Mtx_F(row,2))
            L = L_Minus;
        % The fitness will decrease from Bra to Ket and the epigenetic
        % pathway of the Bra is one step GREATER than the Ket, so genetic
        % information is transferred to the less fit bacteria
        elseif (Combine_Mtx_F(row,7) <= 0)&&(Combine_Mtx_F(row,1)>Combine_Mtx_F(row,2))
            L = L_Plus;
        % The L+/- operator will still work out to zero, however this saves time on calculations.
        %(Combine_Mtx_F(row,7) > 0)&&(Combine_Mtx_F(row,1)<Combine_Mtx_F(row,2))
        % L = L_Plus
        % The fitness will increase if Bra turns to Ket so nothing happens
        % because the Bra will not transfer anything.  
        elseif Combine_Mtx_F(row,7) > 0
            L = 0;
        end
        Norm(row,1) = Bra*L*Ket;
    end
    Combine_Mtx_F_Norm_No_Conj = [Combine_Mtx_F Norm];
    
    % Recalculate the conjugation rate at position "il" as a larger matrix
    % to use for the matrix "Combine_Mtx_F_Norm"
    Conj_Rate_Mtx = zeros();
    for row = 1:size(Combine_Mtx_F_Norm_No_Conj,1)
        Conj_Rate_Mtx(row,1) = Conj_Rate*(min(Combine_Mtx_F_Norm_No_Conj(row,3:4)));
    end

    % This matrix combines the normalization constant, (<Bra|L|Ket>),
    % with the conjugation rates of each bacteria, (Conj_Rate_Mtx),
    % on the far right row
    Combine_Mtx_F_Norm = [Combine_Mtx_F_Norm_No_Conj ...
                           Conj_Rate_Mtx.*Combine_Mtx_F_Norm_No_Conj(:,8)];
                       
    % THIS ALGORITHM IS USED FOR CHANGING ALL THE BACTERIAL POPULATION
    % VALUES IN THE Combine_Mtx_F_Norm MATRIX ALLTOGETHER, BUT ONLY ONE
    % BACTERIAL PAIR CONJUGATES AT A TIME
%     for row = 1:size(Combine_Mtx_F_Norm,1)
%         if Combine_Mtx_F_Norm(row,3)*Combine_Mtx_F_Norm(row,4)*Combine_Mtx_F_Norm(row,6)>0
%             Combine_Mtx_F_Norm(row,3) = Combine_Mtx_F_Norm(row,3) + 1;
%             Combine_Mtx_F_Norm(row,4) = Combine_Mtx_F_Norm(row,4) - 1;
%             for next = 1:size(Combine_Mtx_F_Norm,1)
%                 % THIS IS THE RIGHT AND LEFT PAIR ON THE LEFT SIDE
%                 % If the left element on the next row = the left
%                 % element on the previous row
%                 if Combine_Mtx_F_Norm(next,1) == Combine_Mtx_F_Norm(row,1)
%                     Combine_Mtx_F_Norm(next,3) = Combine_Mtx_F_Norm(row,3);
%                 end
%                 % If the right element on the next row = the left
%                 % element on the previous row
%                 if Combine_Mtx_F_Norm(next,2) == Combine_Mtx_F_Norm(row,1)
%                     Combine_Mtx_F_Norm(next,4) = Combine_Mtx_F_Norm(row,3);
%                 end
%                 % If the left element on the next row = the right
%                 % element on the previous row
%                 if Combine_Mtx_F_Norm(next,1) == Combine_Mtx_F_Norm(row,2)
%                     Combine_Mtx_F_Norm(next,3) = Combine_Mtx_F_Norm(row,4);
%                 end
%                 % If the right element on the next row = the right
%                 % element on the previous row
%                 if Combine_Mtx_F_Norm(next,2) == Combine_Mtx_F_Norm(row,2)
%                     Combine_Mtx_F_Norm(next,4) = Combine_Mtx_F_Norm(row,4);
%                 end
%             end
%         end
%     end
                             
    end % End Conjugation_Selected > 0 if statement     
end