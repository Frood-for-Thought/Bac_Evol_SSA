%% Function for Calculating Bacterial Growth Rate

function [Growth_Rate] = Growth_Rate_Function(Food_Factor,FF,...
    MewMax,K,MIC,Fit,Max_Monod,Cipro,Pop_Factor)

    Monod = FF*MewMax*(Food_Factor)/(K + Food_Factor);% The growth rate, # bac/min; for Monod Eqn.
    % The monod eqn going to minus infinity caused infinite while loops
    % later because rand() would pick very small numbers at locations which
    % should have been zero
%     if Monod < 0.0001
%         Monod = 0;
%     end

    % Use MIC_Bac in case you need to dynamically calculate the MIC with
    % changing food concentration in the algorithm
%     MIC_Bac = Fit*MIC*(10 - 9*(Monod/Max_Monod)); % Fast-Growth Targeting Anitibiotic MIC
    Pharma_Function = 1 - (Cipro/MIC)^2; % Effect antibiotic has on bacteria
%     Pharma_Function = 1/(1 + (Cipro/MIC_Bac)); % FGTA
    if (Pharma_Function < 0) 
        Pharma_Function = 0;
    end
    Growth_Rate = Monod*Pharma_Function*Pop_Factor;
end