
function [vd,R_a] = Repellant_Mig_Function(nl,MIC_Bac,Cipro,vd_chemotaxis,...
    kson,ksoff,Max_Food_Conc)

    Even = mod(nl,2);
    if Even == 0 % Even number
        disp('Cannot use this value. nl has to be odd.');
        return;
    end
    
    % This adds the response function from the antibiotic to reduce speed
    % to zero
    vd = zeros();
    R_a = zeros();
    for i = 1:nl
        % FOR EXPERIMENT 1 REPELLANT HAS NO EFFECT.
        R_a(i) = 1;
        vd(i) = R_a(i)*vd_chemotaxis(i);
    end

end