
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
        Antibiotic = Cipro(i);
        % As is the lowest relative run speed the bacteria can travel in an antibiotic
        As = 0.005;
        % if Antibiotic > MIC then R_a goes to the minimum value
        % The high number, (sigma), within the error function "erf(sigma(...))"
        % is there to create a sharp curve so that in regions without
        % antibiotic the error function has no effect, so that the mutant
        % bacteria will not have a chemotaxis advantage in regions without
        % antibiotic
%         R_a(i) = (0.5-c_adapt)*(erf(25*(MIC_Bac(i) - Antibiotic))+1)+(2*c_adapt);
%         R_a(i) = 0.5*(erf(32.8*(MIC_Bac(i) - Antibiotic))+ 1 + As);
        R_a(i) = 0.8 + 0.2*(erf(32.8*(MIC_Bac(i) - Antibiotic)));
        
%         if R_a(i) < 0.6 
%             % In this example
%             % R_a(i) = 0.8 - 0.2*(erf(32.8*(MIC_Bac(i) - Antibiotic)));
%             % Where 0.8 is the midpoint and the function goes from 0.6 to 1
%             R_a(i) = 0.6;
%         end
        
        % INSTEAD OF Vd, now can multiply R_a(i)*Vo_max
        
        vd(i) = R_a(i)*vd_chemotaxis(i);
    end

end