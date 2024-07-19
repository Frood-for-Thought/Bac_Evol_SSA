In this experiment, the folder "R_a Set to 1" has R_a = 1, and so there is no change to bacterial migration in antibiotic.
This folder contains Repellant_Mig_Function.m which has R_a set to 1.

The folder "R_a Set to Normal Equation" has the regular program, except the main program has been edited to record time as explained below:

Inside the Main_Program.m for each Folder the while loop ends when mutant bacteria start to outnumber the wild type.
This measurement occurs during the time progression section:
%% Given the Time Progression, Decide if the time is updated for the 
% Figure and Video, and Update the Positions of the Bacteria 
% Into the New Time Integer

...

        % CHECK TO SEE IF MUTANT BACTERIA OUTNUMBER WILD TYPE.
        % IF TRUE THEN END THE WHILE LOOP AND RECORD THE TIME.
        exit_while_loop = false;
        for i = 1:nl
            if (x(i,itim) < m1(i,itim)) | (x(i,itim) < m2(i,itim)) | (x(i,itim) < m3(i,itim))
                Time_Recorded = oldtim
                exit_while_loop = true;
            end
        end
        
        if exit_while_loop
            break;
        end

