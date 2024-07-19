%% Function for calculating migration rate

function [vd,R_a,MIC_Bac] = Calculate_R_a_MIC_Function(nl,Food_Factor,FF,...
    MewMax,K,MIC,Fit,Max_Monod,Cipro,vd_chemotaxis,kson,ksoff,Max_Food_Conc)

    Monod = zeros();
    MIC_Bac = zeros();
    for i = 1:nl
        Monod(i) = FF*MewMax*(Food_Factor(i))/(K + Food_Factor(i));% The growth rate, # bac/min; for Monod Eqn.
        MIC_Bac(i) = Fit*MIC*(10 - 9*(Monod(i)/Max_Monod)); % Fast-Growth Targeting Anitibiotic MIC
    end
    [vd,R_a] = Repellant_Mig_Function(nl,MIC_Bac,Cipro,vd_chemotaxis,kson,ksoff,Max_Food_Conc);

end