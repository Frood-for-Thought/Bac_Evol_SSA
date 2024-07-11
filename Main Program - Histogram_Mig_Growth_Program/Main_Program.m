% ****GILLESPIE STOCHASTIC SIMULATION ALGORITHM (SSA)
%*********** SSA For Mutant Fixation at Goldilocks Point

%% initialize position space and units
close all
clf

% TOTAL TIME: (60sec)*(min)
ttim = 60*2880; % (1440min = 24h)

itertot = 1; % total iteration
nl = 101;   % Total number of demes
Even = mod(nl,2);
if Even == 0 % Even number
    disp('Cannot use this value. nl has to be odd.');
    return;
end
w = zeros(nl,ttim); % Initialize wild type baceria array
mut1 = zeros(nl,ttim); % Initialize mutant 1 array
mut2 = zeros(nl,ttim); % Initialize mutant 2 array
mut3 = zeros(nl,ttim); % Initialize mutant 3 array
% Deme population has thousands of bacteria but CC is lower to save computational time
CC = 600; % Carrying Capacity, CCx10 bacteria/deme
ntot = 900; % total particle number
death = 0.0045/60; % The Death Rate. (0.0045/60 = 0.000075)

% Fit
Fit_WT = 1;
Fit_m1 = 10; % The fitness of the bacteria is 20 for the first mutation
Fit_m2 = 50; % The fitness of the bacteria is 100 for the second mutation
Fit_m3 = 200; % The fitness of the bacteria is 200 for the second mutation
% FF
WT_FF = 1.0; % The Food Fitness of Wild Type bacteria
Mut_1_FF = 0.85; % The Food Fitness of Mutant bacteria 1
Mut_2_FF = 0.80;
Mut_3_FF = 0.9;
% FGTA: Beta = Fit*MIC*(10 - 9*(g(s)/g_max))
if (WT_FF > 1) || (Mut_1_FF > 1) || (Mut_2_FF > 1) || (Mut_3_FF > 1)
    disp('Since the FGTA depends on the ratio of g/g_max, then the numerator has to be below one.');
    return;
end
Conj_Rate = 0.01/3600; % Rate = 7.6x10^-3 conjugates/h, units [conj/s].
Start_Conj = round(3);
% Start_Conj = 0;
MIC = 0.05; % The MIC for WT E. coli (µg / mL)
K = 22; % The Monod Constant 22 µM ~ 110 molecules/µm^3 
% µ_max = 3 bac/hour = 0.00083 bac/sec
Max_Monod = (3/3600); % Max Growth Rate/sec, for MIC Eqn.
MewMax = (3/3600); % The growth rate, No. bac/sec, for Monod Eqn.
DL = 310; % The deme length (µm)

%% Insert Gradient Slope and Maximum Concentration Reached
Grad = 0.000405; % µm^-1
Max_Food_Conc = 60000; % µM

% Recently with the current Tar, Tsr receptor ranges, the maximum drift speed 
% and lowest critical gradient takes place around 315000 µM

%% INSEERT FOOD CONCENTRATION

% If Monod/Max_Monod = 99% at K = 1.83*10^-4 mM, then concentration of food is then
% equal to conc. = 99*K ~ 0.2 mM
% Minimum concentration: s_min = 12µg/L

Food_Function = zeros(nl,1);
for Food_Pos = 1:nl
    Food_Function(Food_Pos) = exp(Grad*Food_Pos*DL);
end

% Normalize with Food_Function(nl) to set Maximum to the numerator value
% Food_Function = Food_Function/Food_Function(nl); %Normalize Food Function.
Ini_Food_Const = Max_Food_Conc/Food_Function(nl); % µM
xbias = Ini_Food_Const*Food_Function;
% xbias = Food_Function;
Mark_Ini_Non_Zero_Deme = 0;
for i = 1:nl
    if (xbias(i) > 0.009) && (Mark_Ini_Non_Zero_Deme < 1)
        Non0_Deme = i;
        Mark_Ini_Non_Zero_Deme = 1;
    end
end

% yyaxis left
% plot(xbias)
% xlim([1 nl])
% ylabel('Concentration [µM]')
% xlabel('Position')
% title('Food Concentration')
Average_Diff = sum(xbias)/((nl-(Non0_Deme-1))*DL); % [µM/µm]
Difference_Label = 'The average difference in concentration between all demes with attractant is %.2f µM/µm';
Initial_Conc_Label = 'The initial concentration at deme %d is %.2f µM';
Final_Conc_Label = 'The largest concentration at deme %d is %.2f µM';
A = sprintf(Initial_Conc_Label, Non0_Deme, xbias(Non0_Deme))
B = sprintf(Final_Conc_Label, nl, xbias(nl))
C = sprintf(Difference_Label,Average_Diff)

%% The chemotaxis speed matrix program
[vd_chemotaxis,c_df_over_dc, Vo_max, kaon,kaoff,kson,ksoff] = ...
    Attractant_Mig_Function(nl,Grad,xbias,Max_Food_Conc);

%% Insert Gradient Slope and Maximum Concentration Reached for Cipro
Grad_Cip = 0.0003; % µm^-1
Max_Cip_Conc = 10; % µg/mL

%% INSEERT CIPROFLOXACIN CONCENTRATION
Cipro_Function = zeros(nl,1);
for Cip_Pos = 1:nl
    Cipro_Function(Cip_Pos) = exp(Grad_Cip*Cip_Pos*DL);
end
% Normalize with Cipro_Function(nl) to set Maximum to the numerator value
Ini_Cip_Const = Max_Cip_Conc/Cipro_Function(nl); % µM
LCipConc = (nl-1)/2;
RCiptConc = (nl-1)/2;
for Cipro_Pos = 1:nl
    if Cipro_Pos <= ((nl-1)/2) %From 1 to (Middle - 1)
        Cipro_Function(Cipro_Pos) = 0;
%     elseif Cipro_Pos >= ((nl-1)/2)+1 % From Middle to Max
%         Cipro_Function(Cipro_Pos) = (Cipro_Pos - RCiptConc);
    end
end
% 10 µg / mL is 200x the MIC
Cip_xbias = Ini_Cip_Const*Cipro_Function; 
% The Cipro gradient
Cipro_Gradient = Grad_Cip*Cip_xbias;

% Descriptions for Min Cip, Max Cip, and Average gradient concentration
Mark_Ini_Non_Zero_Deme = 0;
for i = 1:nl
    if (Cip_xbias(i) > 0.009) && (Mark_Ini_Non_Zero_Deme < 1)
        Non0_Deme = i;
        Mark_Ini_Non_Zero_Deme = 1;
    end
end
Average_Diff = sum(Cip_xbias)/((nl-(Non0_Deme-1))*DL); % [µg/mL*µm]
Difference_Label = 'The average difference in concentration between all demes with Cipro is %.2f µg/mL*µm';
Initial_Conc_Label = 'The initial concentration at deme %d is %.2f µg/mL';
Final_Conc_Label = 'The largest concentration at deme %d is %.2f µg/mL';
A = sprintf(Initial_Conc_Label, Non0_Deme, Cip_xbias(Non0_Deme))
B = sprintf(Final_Conc_Label, nl, Cip_xbias(nl))
C = sprintf(Difference_Label,Average_Diff)

%% Constant Cipro Function
% for Cipro_Pos = 1:nl
%     Cipro_Function(Cipro_Pos) = 0.02;
% end
% Cip_xbias = Cipro_Function;
% % Cip_xbias = zeros(nl,1);

%% PLOTTING vd VS Position Without Repellent Factor
close all
% Label the figures to be used
f1 = figure;
f2 = figure;


Current_Gradient_Max_Velocity = max(vd_chemotaxis);
Current_Max_Drift_Vel_Label = 'The max drift velocity with the antibiotic is %.2f µm/s';
Current_Max_Drift_Velocity = sprintf(Current_Max_Drift_Vel_Label,Current_Gradient_Max_Velocity)


figure(f1);
clf
title('Modification of the Drift Velocity From Ciprofloxacin', 'fontsize', 16)
yyaxis left
plot(vd_chemotaxis,'b', 'LineWidth',2);
ylabel('Drift Velocity, Vd [µm/s]')
xlim([0 nl])
ylim([0 5])
yyaxis right
% plot(Cip_xbias,'r')
ylabel('Bacteria WT -> Mut3 Vd from Repulsion Equation')
% ylim([0 max(max(Cip_xbias))])








%%
Food_Factor = xbias;
Cipro = Cip_xbias;
Bac_Pos_Type_vd = 0;
%% Calculating WT MIC with Cipro Repellent
FF = WT_FF; % Fitness factor for growth rate with Monod
Fit = Fit_WT; % Fitness factor for MIC
Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
[vd,R_a,MIC_Bac] = Calculate_R_a_MIC_Function(nl,Food_Factor,FF,...
    MewMax,K,MIC,Fit,Max_Monod,Cipro,vd_chemotaxis,kson,ksoff,Max_Food_Conc);
% The function used here is to find the repellant equation "R_a" for the WT bacteria
R_a_WT = R_a;
% The following plots are for showing the repellant equation in the drift velocity plot above
hold on
plot(vd)
ylim([0 5])
MIC_WT = MIC_Bac;

%% Calculating Mut1 MIC with Cipro Repellent
FF = Mut_1_FF; % Fitness factor for growth rate with Monod
Fit = Fit_m1; % Fitness factor for MIC
Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
[vd,R_a,MIC_Bac] = Calculate_R_a_MIC_Function(nl,Food_Factor,FF,...
    MewMax,K,MIC,Fit,Max_Monod,Cipro,vd_chemotaxis,kson,ksoff,Max_Food_Conc);
% The function used here is to find the repellant equation "R_a" for the Mut1 bacteria
R_a_Mut1 = R_a;
plot(vd)
MIC_Mut1 = MIC_Bac;

%% Calculating Mut2 MIC with Cipro Repellent
FF = Mut_2_FF; % Fitness factor for growth rate with Monod
Fit = Fit_m2; % Fitness factor for MIC
Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
[vd,R_a,MIC_Bac] = Calculate_R_a_MIC_Function(nl,Food_Factor,FF,...
    MewMax,K,MIC,Fit,Max_Monod,Cipro,vd_chemotaxis,kson,ksoff,Max_Food_Conc);
% The function used here is to find the repellant equation "R_a" for the Mut2 bacteria
R_a_Mut2 = R_a;
plot(vd)
MIC_Mut2 = MIC_Bac;

%% Calculating Mut3 MIC with Cipro Repellent
FF = Mut_3_FF; % Fitness factor for growth rate with Monod
Fit = Fit_m3; % Fitness factor for MIC
Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
[vd,R_a,MIC_Bac] = Calculate_R_a_MIC_Function(nl,Food_Factor,FF,...
    MewMax,K,MIC,Fit,Max_Monod,Cipro,vd_chemotaxis,kson,ksoff,Max_Food_Conc);
% The function used here is to find the repellant equation "R_a" for the Mut3 bacteria
R_a_Mut3 = R_a;
MIC_Mut3 = MIC_Bac;
plot(vd)
    % Legend
    Bacterial_Label_WT = 'Wild Type, Cip Fit = %d, Food Fit = %.2f';
    A = sprintf(Bacterial_Label_WT,Fit_WT,WT_FF);
    Bacterial_Label_M1 = 'Mutant 1, Cip Fit = %d, Food Fit = %.2f';
    Bacterial_Label_M2 = 'Mutant 2, Cip Fit = %d, Food Fit = %.2f';
    Bacterial_Label_M3 = 'Mutant 3, Cip Fit = %d, Food Fit = %.2f';
    legend({'Unchanged Drift Velocity',A,sprintf(Bacterial_Label_M1,Fit_m1,Mut_1_FF),...
        sprintf(Bacterial_Label_M2,Fit_m2,Mut_2_FF),...
        sprintf(Bacterial_Label_M3,Fit_m3,Mut_3_FF)},'Location','northwest');
f1.Position = [720 100 800 672];

% Repellant_Mig_Function has R_a alter the Vo_max for the type of 
% bacteria if it is in a region of ANTIBIOTIC > MIC

%% Calculate the Time Rate of Change of the Fractional Amount of 
 % Receptor (Protein) Bound

Rtroc = zeros();
for i = 1:nl
    % The receptor time rate of change, with respect to position
    Rtroc(i) = vd_chemotaxis(i)*Grad*c_df_over_dc(i);
end

%% Open the Alpha Values

file_title = 'alpha_MaxC_%d_Grad_%.6f.xlsx';
filename = sprintf(file_title,Max_Food_Conc,Grad);
A = xlsread(filename);
format short g
alpha = transpose(A(:,2));
lnr = 1.16; % Previously calculated.

%%  SET ITERATION LOOP

% Open video file to record
v = VideoWriter('BacteriaMutantFix.avi');
open(v)
video_Condition = 0;

for iter = 1:itertot; % set iterationloop
% first initialization

%% Set the Bacterial Type Object Migration Parameters per Deme
BacObj = 0;
% Obj_Ini is used to check to see if it is going through initalization to
% determine the rates of a reaction
Obj_Ini = 0;
% This should initalize the object array for bacterial migration
for i = 1:nl
    % Use variables to initialize the object with rates.
    BacObj = Bac_Mig_Param_Class...
            (alpha, Rtroc, lnr, i);
    name = strcat('BACobj_',num2str(i));
    % Assign an object to the field in the array.
    Bac_rate_obj.(name) = BacObj;
end

% Using the object in the program was requiring too many resources,
% instead the tumble rate values are put into arrays.
R_Tum_Up_Mtx = zeros();
R_Tum_Down_Mtx = zeros();
for j = 1:nl
    name = strcat('BACobj_',num2str(j));
    R_Tum_Up_Mtx(j) = Bac_rate_obj.(name).R_tum_up;
    R_Tum_Down_Mtx(j) = Bac_rate_obj.(name).R_tum_down;
end

%% Input More Program Parameters (POSITION AND TIME)

% x is a zeros matrix with dimensions "n1" by "ttim"
x = zeros(nl,ttim);
m1 = zeros(nl,ttim);
m2 = zeros(nl,ttim);
m3 = zeros(nl,ttim);

% x(1,1) = ntot; % Wild type bacteria placed at the left
% ini_i = 16;
ini_i = 18;
x(ini_i,1) = CC;
x(ini_i-1,1) = CC;
x(ini_i-2,1) = CC;

% initialize in the middle x = (nl-1)/2+1
% x(((nl-1)/2+1),1) = ntot; % Wild type bacteria placed in the center
% xx(((nl-1)/2+1),1) = ntot;

m1(((nl-1)/2+1),1) = 0; % The mutant bacteria
% m1(1,1) = ntot;

m2(((nl-1)/2+1),1) = 0;
% m2(20,1) = ntot;

m3(((nl-1)/2+1),1) = 0;
% m3(50,1) = ntot;


% Time Parameters
xttim = 1;
itim = 1;
oldtim = 0;
Time = 0; % Used for mutation calculation.
next_num = itim + 1;
tum_dt = 0;

% This counts the total number of migrations which can take place at each
% element from 1:nl
Count_Num_Mig = x(1:nl,itim) + m1(1:nl,itim) + m2(1:nl,itim) + m3(1:nl,itim);
All_Particles = sum(Count_Num_Mig); % Create a random order of all the particles to pick
Rand_Part_Mtx_El = 0;
New_Bacteria = 0; % Count the number of growths
Lose_Bacteria_70 = 0; % Count the number of deaths
Mutation_Occurs = 0; % Count the number of mutations
Conjugation_Occurs = 0; % Count the number of times conjugation occurs
Migration_Occurs = 0; % Count the number of times migration occurs

%% RUN THROUGH THE ALGORITHM
while (itim < ttim) % set time while loop

    % AFTER 24H THE FOOD GRADIENT BECOMES A CONSTANT
%     if itim > 720*60
% %         xbias(:,1) = 11000;
%         Cip_xbias(:,1) = 0;
%     end
    
    mutation_time = 0; % Used to check the first mutation
    % ttimold is the time before adding dt to itim, (xttim = itim + dt)
        
%% BEGIN INITIALIZING LOOP
    
    Tot_Num = 0; % Tot_Num counts each bacteria
    Pop_Factor = 0; % Population Factor
    R_growth = 0; % Initial Growth Rate
    R_Mig = 0; % Initial Migration Rate
    R_conj = 0; % Initial Conjugation Rate 
    Conjugation_Selected = 0; % Here to make sure unnecessary calculations 
                            % are not performed in the Conjugation_Function
    P_g_deme = zeros(1,nl); % Growth probability per position, il (deme)
    P_deme_conj = zeros(1,nl); % Conjugation probability per position
    P_deme_Mig = zeros(1,nl); % Migration probability per position
    P_deme_Mig_WT = zeros(1,nl);
    P_deme_Mig_Mut1 = zeros(1,nl);
    P_deme_Mig_Mut2 = zeros(1,nl);
    P_deme_Mig_Mut3 = zeros(1,nl);
    
    % select position loop
    for il = 1:nl
         nxtotal = x(il,itim)+ m1(il,itim) + m2(il,itim) + m3(il,itim);
         Pop_Factor = (1 - nxtotal/CC);
         if nxtotal > CC
             Pop_Factor = 0;
         end
         
         % nxloop is the total number of particles at position "il"
            % nxloop records the total number of particles at the  
                % original position and the old time
         if nxtotal > 0
            
             % Check if bacteria are on either side and find the population
            if il < nl
                i = il + 1;
                ntot_right = x(i,itim) + m1(i,itim) + m2(i,itim) + m3(i,itim);
            else % The boundary on the right
                ntot_right = NaN;
            end
            if il > 1
                i = il - 1;
                ntot_left = x(i,itim) + m1(i,itim) + m2(i,itim) + m3(i,itim);
            else % The boundary on the left
                ntot_left = NaN;
            end
             
             % Calculating the Initial Rates for Gillespie Algorithm
            Tot_Num = Tot_Num + nxtotal;
            Food_Factor = xbias(il); % Food concentration
            Cipro = Cip_xbias(il); % Cipro concentration
            % Selecting Bacteria Type
            if x(il,itim) > 0
    % MIC, MONOD, AND Pharma Function and Growth Rate for Wild Type bacteria
                FF = WT_FF; % Fitness factor for growth rate with Monod
                Fit = Fit_WT; % Fitness factor for MIC
                Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
                MIC = MIC_WT(il); % The MIC of this type of bacteria at position "il"
                R_a = R_a_WT(il); % The Repellant equation of this type of bacteria at position "il"
                [Growth_Rate,R_mig_deme] = Initialize_Rates_Function...
                    (il,nl,CC,ntot_right,ntot_left,Food_Factor,FF,MewMax,K,...
                    MIC,Fit,Max_Monod,Cipro,Pop_Factor,DL,Vo_max,...
                    Rand_Part_Mtx_El, All_Particles,R_a,...
                    R_Tum_Up_Mtx, R_Tum_Down_Mtx);
                WT_mN = R_mig_deme*x(il,itim); % Total rate of migration at deme for bacteria type
                WT_gN = Growth_Rate*x(il,itim); % Total growth rate at deme for bacteria type
            else
                WT_mN = 0;
                WT_gN = 0;
            end

            if  m1(il,itim) > 0
    % MIC, MONOD, AND Pharma Function and Growth Rate for Mutant 1 bacteria
                FF = Mut_1_FF; % Fitness factor for growth rate with Monod
                Fit = Fit_m1; % Fitness factor for MIC
                Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
                MIC = MIC_Mut1(il); % The MIC of this type of bacteria at position "il"
                R_a = R_a_Mut1(il); % The Repellant equation of this type of bacteria at position "il"
                [Growth_Rate,R_mig_deme] = Initialize_Rates_Function...
                    (il,nl,CC,ntot_right,ntot_left,Food_Factor,FF,MewMax,K,...
                    MIC,Fit,Max_Monod,Cipro,Pop_Factor,DL,Vo_max,...
                    Rand_Part_Mtx_El, All_Particles,R_a,...
                    R_Tum_Up_Mtx, R_Tum_Down_Mtx);
                Mut1_mN = R_mig_deme*m1(il,itim); % Total rate of migration at deme for bacteria type
                Mut1_gN = Growth_Rate*m1(il,itim); % Total growth rate at deme for bacteria type
            else
                Mut1_mN = 0;
                Mut1_gN = 0;
            end
            
            if m2(il,itim) > 0
    % MIC, MONOD, AND Pharma Function and Growth Rate for Mutant 1 bacteria
                FF = Mut_2_FF; % Fitness factor for growth rate with Monod
                Fit = Fit_m2; % Fitness factor for MIC
                Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
                MIC = MIC_Mut2(il); % The MIC of this type of bacteria at position "il"
                R_a = R_a_Mut2(il); % The Repellant equation of this type of bacteria at position "il"
                [Growth_Rate,R_mig_deme] = Initialize_Rates_Function...
                    (il,nl,CC,ntot_right,ntot_left,Food_Factor,FF,MewMax,K,...
                    MIC,Fit,Max_Monod,Cipro,Pop_Factor,DL,Vo_max,...
                    Rand_Part_Mtx_El, All_Particles,R_a,...
                    R_Tum_Up_Mtx, R_Tum_Down_Mtx);
                Mut2_mN = R_mig_deme*m2(il,itim); % Total rate of migration at deme for bacteria type
                Mut2_gN = Growth_Rate*m2(il,itim); % Total growth rate at deme for bacteria type
            else
                Mut2_mN = 0;
                Mut2_gN = 0;
            end
            
            if m3(il,itim) > 0
    % MIC, MONOD, AND Pharma Function and Growth Rate for Mutant 1 bacteria
                FF = Mut_3_FF; % Fitness factor for growth rate with Monod
                Fit = Fit_m3; % Fitness factor for MIC
                Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
                MIC = MIC_Mut3(il); % The MIC of this type of bacteria at position "il"
                R_a = R_a_Mut3(il); % The Repellant equation of this type of bacteria at position "il"
                [Growth_Rate,R_mig_deme] = Initialize_Rates_Function...
                    (il,nl,CC,ntot_right,ntot_left,Food_Factor,FF,MewMax,K,...
                    MIC,Fit,Max_Monod,Cipro,Pop_Factor,DL,Vo_max,...
                    Rand_Part_Mtx_El, All_Particles,R_a,...
                    R_Tum_Up_Mtx, R_Tum_Down_Mtx);
                Mut3_mN = R_mig_deme*m3(il,itim); % Total rate of migration at deme for bacteria type
                Mut3_gN = Growth_Rate*m3(il,itim); % Total growth rate at deme for bacteria type
            else
                Mut3_mN = 0;
                Mut3_gN = 0;
            end
            
            % This calculates the conjugation rate for position "il"
            cjN = 0;
            Conjugation_Selected = 0;

            Bacteria_Num_Mtx = [x(il,itim) m1(il,itim) m2(il,itim) m3(il,itim)];
            Bac_Interact_in_Deme = sum(any(Bacteria_Num_Mtx,1));
            Bacteria_Over_Conj = zeros(1,4);
            for col = 1:length(Bacteria_Num_Mtx)
                Bacteria_Over_Conj(1,col) = ...
                    0.5*(Conj_Rate*(erf(10000*(Bacteria_Num_Mtx(1,col) - Start_Conj))) + Conj_Rate);
            end
            Bac_Over_Start_Conj = sum(any(Bacteria_Over_Conj,1));
            if (Bac_Interact_in_Deme >= 2) && (Bac_Over_Start_Conj >= 2)
                % This function calculates the conjugation rate at il
                [cjN] = Conjugation_Function(Start_Conj,Bacteria_Num_Mtx, ...
                    Conj_Rate,WT_FF,Mut_1_FF,Mut_2_FF,Mut_3_FF,MIC,...
                        Fit_WT,Fit_m1,Fit_m2,Fit_m3,Cipro,Conjugation_Selected);
            end
            
            %% Calculate Total Rates and Prob. Rate per deme
            gN = (WT_gN + Mut1_gN + Mut2_gN + Mut3_gN)*10;
            % Since demes are in units of 10, gN growth rate is 
            % 10 times more likely
            R_growth = R_growth + gN;
            % Make a  matrix to describe growth 
            % probability per position, P_il
            P_g_deme(il) = P_g_deme(il) + gN;
            
            mN = (WT_mN + Mut1_mN + Mut2_mN + Mut3_mN)*10;
            R_Mig = R_Mig + mN;
            % Make a matrix to describe Migration probability
            % per position, P_il
            P_deme_Mig(il) = P_deme_Mig(il) + mN;
            P_deme_Mig_WT(il) = P_deme_Mig_WT(il) + WT_mN;
            P_deme_Mig_Mut1(il) = P_deme_Mig_Mut1(il) + Mut1_mN;
            P_deme_Mig_Mut2(il) = P_deme_Mig_Mut2(il) + Mut2_mN;
            P_deme_Mig_Mut3(il) = P_deme_Mig_Mut3(il) + Mut3_mN;

            % Calculate Rate of conjugation for this time, "itim"
            R_conj = R_conj + cjN*10;
            % Make a  matrix to describe Conjugation probability 
            % per position, P_il
            P_deme_conj(il) = P_deme_conj(il) + cjN;
         end % if nxtotal > 0
    end % for il = 1:nl
    
    % In case total rates are needed to be shown.
    R_growth;
    R_Mig;
    R_death = death*Tot_Num*10; % the death rate includes the x10 pop factor
    R_conj;
    
    % The total rate of reaction
    R_tot = R_growth + R_Mig + R_death + R_conj;
    
%% TIME PROGRESSION
    r1 = rand();
    dt = (1/R_tot)*log(1/r1);
    xttim = xttim + dt; % new time
    Time = Time + dt;
    
%% RANDOMLY SELECT RATE
    % r1 determines time progression while r2 selects which action to take
    r2 = rand();
    
    %% MIGRATION IS SELECTED
    if (0 <= r2) && (r2 < R_Mig/R_tot)
        
        % Check to see if a migration occured at a non-boundary zone
        Mig_Occured = 0;
        Mig_Left = 0;
        Mig_Right = 0;
        Cant_Select_Bacteria = 0; % This variable is security to prevent the while loop from getting stuck.
        while Mig_Occured <= 0
            % Look for a deme location using normalized migration
            % probabilites for each deme.
            Find_Location = 1; % Boolean to signify function to look for location.
            [i, Location_Mig_Probability, Find_Location]...
                = Select_Location_Bacteria_Mig_Function(...
                P_deme_Mig, R_Mig, Find_Location, nl);
            
            % Look for location and bacterial type to migrate
            pos_selected = 0;
            while pos_selected < 1
                Find_Location = 0; % Find Bacterial Type.
                [~, ~, Find_Location,WT_Selected,Mut1_Selected,Mut2_Selected,Mut3_Selected,pos_selected]...
                    = Select_Location_Bacteria_Mig_Function(...
                    P_deme_Mig, R_Mig, Find_Location, nl, i, Location_Mig_Probability,...
                    P_deme_Mig_WT, P_deme_Mig_Mut1,P_deme_Mig_Mut2, P_deme_Mig_Mut3);
                % Find_Location = 1, no position found
                if Find_Location > 0
                    % CHOOSE A NEW LOCATION
                    [i, Location_Mig_Probability, Find_Location]...
                        = Select_Location_Bacteria_Mig_Function(...
                        P_deme_Mig, R_Mig, Find_Location, nl);
                end
            end
            % Go through loop to migrate bacteria selected.
            if Cant_Select_Bacteria <= 10
                % Bacteria at location i
                NIniTot = x(i,itim) + m1(i,itim) + m2(i,itim) + m3(i,itim);
                i_R = i + 1;
                i_L = i - 1;
                ntot_right = 0;
                ntot_left = 0;
                % Count the number of bacteria on the sides
                if (i > 1) && (i < nl)
                    ntot_right = x(i_R,itim)+m1(i_R,itim)+m2(i_R,itim)+m3(i_R,itim);
                    ntot_left = x(i_L,itim)+m1(i_L,itim)+m2(i_L,itim)+m3(i_L,itim);
                elseif i == 1
                    ntot_right = x(i_R,itim)+m1(i_R,itim)+m2(i_R,itim)+m3(i_R,itim);
                    ntot_left = NaN;
                elseif i == nl
                    ntot_right = NaN;
                    ntot_left = x(i_L,itim)+m1(i_L,itim)+m2(i_L,itim)+m3(i_L,itim);
                end
                if ntot_right < CC | ntot_left < CC
                    R_tum_up = NaN;
                    R_tum_down = NaN;
                    % Find the object label for position il for rates.
                    if ntot_right < CC
                        R_tum_up = R_Tum_Up_Mtx(i);
                    end
                    if ntot_left < CC
                        R_tum_down = R_Tum_Down_Mtx(i);
                    end
                    
                    % Determine if the speed is reduced by antibiotics
                    if WT_Selected > 0
                        Vo_WT = Vo_max*R_a_WT(i); % The speed of this type of bacteria at position "il"
                        Vo_bac = Vo_WT;
                    elseif Mut1_Selected > 0
                        Vo_m1 = Vo_max*R_a_Mut1(i); % The speed of this type of bacteria at position "il"
                        Vo_bac = Vo_m1;
                    elseif Mut2_Selected > 0
                        Vo_m2 = Vo_max*R_a_Mut2(i); % The speed of this type of bacteria at position "il"
                        Vo_bac = Vo_m2;
                    elseif Mut3_Selected > 0
                        Vo_m3 = Vo_max*R_a_Mut3(i); % The speed of this type of bacteria at position "il"
                        Vo_bac = Vo_m3;
                    end
                    
                    % The boolean is set to 0 to instruct the function not to 
                    % calculate the initialized rates, but to calculate which 
                    % direction migration occurs.
                    calc_mig_rate = 0;
                    
                    [~, Mig_Occured, Mig_Right, Mig_Left]...
                        = Calculate_Position_Mig_Function...
                            (R_tum_up, R_tum_down, Vo_max, DL, calc_mig_rate,...
                             Mig_Occured, Mig_Right, Mig_Left);
                else
                    % If nothing happens then it just means that the bacteria moved
                    % within its own Deme and did not move to a neighbouring one
                    Cant_Select_Bacteria = Cant_Select_Bacteria + 1;
                    continue
                end % if ntot_right < CC | ntot_left < CC
            else % No bacteria was selected
                break
            end % if Cant_Select_Bacteria <= 10
        end % while Mig_Occured <= 0
        
        % Move the bacteria that was selected
        if Mig_Occured > 0
            if Mig_Right > 0
                i_fin = i + 1;
            elseif Mig_Left > 0
                i_fin = i - 1;
            end
            if WT_Selected > 0
                x(i,itim) = x(i,itim) - 1;
                x(i_fin,itim) = x(i_fin,itim) + 1;
            elseif Mut1_Selected > 0
                m1(i,itim) = m1(i,itim) - 1;
                m1(i_fin,itim) = m1(i_fin,itim) + 1;
            elseif Mut2_Selected > 0
                m2(i,itim) = m2(i,itim) - 1;
                m2(i_fin,itim) = m2(i_fin,itim) + 1;
            elseif Mut3_Selected > 0
                m3(i,itim) = m3(i,itim) - 1;
                m3(i_fin,itim) = m3(i_fin,itim) + 1;
            end
        end % if Mig_Occured > 0
        
    elseif (R_Mig/R_tot <= r2) && (r2 < (R_Mig+R_growth)/R_tot)
        %% GROWTH IS SELECTED
        
        % Previously the growth rate of all the particles was calculated,
          % but now that growth is selected, the location of where a
          % particle replicates is now calculated using P_deme(il)
        
        % FIND GROWTH LOCATION
        R_growth; % Total rate of growth
        P_g_deme = P_g_deme/sum(P_g_deme); % Normalized growth probability for each deme
                                    % to find where in the system growth occurs
        P_il_location = 1:size(P_g_deme, 2);
        P_deme_Order = P_g_deme(:, P_il_location);
        P_deme_wth_Position = [P_deme_Order; P_il_location];
        r3 = rand();
        sum_prob = 0;
        for il = 1:length(P_deme_wth_Position) % select position loop
            sum_prob = sum_prob + P_deme_wth_Position(1,il);
            if r3 <= sum_prob
                Location_Growth_Probability = P_deme_wth_Position(1,il);
                Growth_Location = P_deme_wth_Position(2,il);
                break
            end
        end
        il = Growth_Location;
        New_Bacteria = New_Bacteria + 1;
        
        % FIND INDIVIDUAL BACTERIA
            % Now that location is found need to find the location where
            % growth occurs
        nxtotal = x(il,itim)+ m1(il,itim) + m2(il,itim) + m3(il,itim);
        Wild_Type_Bac = x(il,itim);
        Mut1_Bac = m1(il,itim);
        Mut2_Bac = m2(il,itim);
        Mut3_Bac = m3(il,itim);
        Pop_Factor = (1 - nxtotal/CC);
        if nxtotal > CC
            Pop_Factor = 0;
        end
        Food_Factor = xbias(il); % Food concentration at deme
        Cipro = Cip_xbias(il); % Cipro concentration at deme
        
    % MIC, MONOD, AND Pharma Function and Growth Rate for Wild Type bacteria
        FF = WT_FF; % Fitness factor for growth rate with Monod
        Fit = Fit_WT; % Fitness factor for MIC
        Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
        MIC = MIC_WT(il); % The MIC of this type of bacteria at position "il"
        [Growth_Rate] = Growth_Rate_Function(Food_Factor,FF,...
            MewMax,K,MIC,Fit,Max_Monod,Cipro,Pop_Factor);
        WT_gN = Growth_Rate*x(il,itim);
        
    % MIC, MONOD, AND Pharma Function and Growth Rate for Mutant 1 bacteria
        FF = Mut_1_FF; % Fitness factor for growth rate with Monod
        Fit = Fit_m1; % Fitness factor for MIC
        Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
        MIC = MIC_Mut1(il); % The MIC of this type of bacteria at position "il"
        [Growth_Rate] = Growth_Rate_Function(Food_Factor,FF,...
            MewMax,K,MIC,Fit,Max_Monod,Cipro,Pop_Factor);
        Mut1_gN = Growth_Rate*m1(il,itim);
        
    % MIC, MONOD, AND Pharma Function and Growth Rate for Mutant 2 bacteria
        FF = Mut_2_FF; % Fitness factor for growth rate with Monod
        Fit = Fit_m2; % Fitness factor for MIC
        Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
        MIC = MIC_Mut2(il); % The MIC of this type of bacteria at position "il"
        [Growth_Rate] = Growth_Rate_Function(Food_Factor,FF,...
            MewMax,K,MIC,Fit,Max_Monod,Cipro,Pop_Factor);
        Mut2_gN = Growth_Rate*m2(il,itim);
        
    % MIC, MONOD, AND Pharma Function and Growth Rate for Mutant 3 bacteria
        FF = Mut_3_FF; % Fitness factor for growth rate with Monod
        Fit = Fit_m3; % Fitness factor for MIC
        Max_Monod = FF*MewMax; % Max Growth Rate/min, for MIC Eqn
        MIC = MIC_Mut3(il); % The MIC of this type of bacteria at position "il"
        [Growth_Rate] = Growth_Rate_Function(Food_Factor,FF,...
            MewMax,K,MIC,Fit,Max_Monod,Cipro,Pop_Factor);
        Mut3_gN = Growth_Rate*m3(il,itim);
        
        % Calculate the growth rate percentage for each bacteria at
        % location il
        P_bac = [WT_gN, Mut1_gN, Mut2_gN, Mut3_gN];
        gN_Normal = WT_gN + Mut1_gN + Mut2_gN + Mut3_gN;
        % After recalculating growth rate the rate may end up being zero
        % even though originally the P_deme here was non-zero
        if gN_Normal == 0
            gN_Normal = 1;
        end
        % If for some reason the growth rate percentage is negative.
        if (Location_Growth_Probability > 0) && (sum(P_bac) <= 0)
            continue
        end
        P_bac = P_bac/gN_Normal;
        P_bac_new_location = randperm(length(P_bac));
        % Rearrange the Normalized gN values
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
        
        % CHECK IF MUTATION OCCURS
        % The probability of mutation should be independent of the other rates
        % because it occurs during growth.  A mutation occuring should be 
        % calculated and then randomly selected, which is then implimented the 
        % next time a growth occurs.  The SOS response increases the
        % mutation rate.
        % µ = 50*10^-6 mut/(cell*day)
        %   ~ 1*10^-9 mut/(cell*sec)
        R_mutation = (1.15*10^-10)*Tot_Num*10; % s^-1
        P_mut = 1 - exp(-R_mutation*Time);
        R_m = rand();
        if R_m <= P_mut
            % If mutation occurs shift the bacteria type to the next mutant
              % until it is the most fit mutant
            Bacteria_Type_Selected = Bacteria_Type_Selected + 1;
            if Bacteria_Type_Selected > length(P_bac)
                Bacteria_Type_Selected = 3; % length(P_bac);
            end
            Mutation_Occurs = Mutation_Occurs + 1
            Time = 0;
            New_Bacteria = 0;
        end
        
        % Bacteria Type
            % 1 = WT
            % 2 = Mut1
            % 3 = Mut2
            % 4 = Mut3
        if Bacteria_Type_Selected == 1
            x(Growth_Location,itim) = x(Growth_Location,itim) + 1;   
        elseif Bacteria_Type_Selected == 2
            m1(Growth_Location,itim) = m1(Growth_Location,itim) + 1;
        elseif Bacteria_Type_Selected == 3
            m2(Growth_Location,itim) = m2(Growth_Location,itim) + 1;
        elseif Bacteria_Type_Selected == 4
            m3(Growth_Location,itim) = m3(Growth_Location,itim) + 1;
        end

    elseif  ((R_Mig+R_growth)/R_tot <= r2) && (r2 < (R_Mig+R_growth+R_death)/R_tot)
        %% CELL DEATH IS SELECTED
        % This function works like Pick_Particle_Lim_Mig_Function except
        % the number of migrations allowed is not updated because no
        % bacteria has moved.
        [WT_Selected,Mut1_Selected,Mut2_Selected,Mut3_Selected,il] ...
                    = Pick_Particle_Function(Tot_Num,nl,itim,x,m1,m2,m3);
        
        if WT_Selected > 0
            x(il,itim) = x(il,itim) - 1;
        elseif Mut1_Selected > 0
            m1(il,itim) = m1(il,itim) - 1;
        elseif Mut2_Selected > 0
            m2(il,itim) = m2(il,itim) - 1;
        elseif Mut3_Selected > 0
            m3(il,itim) = m3(il,itim) - 1;
        end
        
    elseif ((R_Mig+R_growth+R_death)/R_tot <= r2) && (r2 < (R_Mig+R_growth+R_death+R_conj)/R_tot)
        %% CONJUGATION IS SELECTED
        
        % Find conjugation location
        R_conj; % Total rate of Conjugation
        P_deme_conj = P_deme_conj/sum(P_deme_conj); % Normalized conj probability for each location
        P_il_location = 1:size(P_deme_conj, 2);
        P_deme_Order = P_deme_conj(:, P_il_location);
        P_deme_wth_Position = [P_deme_Order; P_il_location];
        r3 = rand();
        sum_prob = 0;
        for il = 1:length(P_deme_wth_Position) % select position loop
            sum_prob = sum_prob + P_deme_wth_Position(1,il);
            if r3 <= sum_prob
                Location_Conj_Probability = P_deme_wth_Position(1,il);
                Conjugation_Location = P_deme_wth_Position(2,il);
                break
            end
        end
        il = Conjugation_Location;

        Conjugation_Selected = 1;
        Cipro = Cip_xbias(il); % Cipro concentration
        Bacteria_Num_Mtx = [x(il,itim) m1(il,itim) m2(il,itim) m3(il,itim)];
        [cjN,Combine_Mtx_F_Norm] = Conjugation_Function(Start_Conj,Bacteria_Num_Mtx, ...
                            Conj_Rate,WT_FF,Mut_1_FF,Mut_2_FF,Mut_3_FF,MIC,...
                                Fit_WT,Fit_m1,Fit_m2,Fit_m3,Cipro,Conjugation_Selected);
                        
                            
        for row = 1:size(Combine_Mtx_F_Norm,1)
            if Combine_Mtx_F_Norm(row,3) < Start_Conj
                Combine_Mtx_F_Norm(row,3) = 0;
            end
            if Combine_Mtx_F_Norm(row,4) < Start_Conj
                Combine_Mtx_F_Norm(row,4) = 0;
            end
        end
        
        % Take the probabililties on the right column of Combine_Mtx_F_Norm
        P_bac_pair_conj = transpose(Combine_Mtx_F_Norm(:,9));
        % Normalize the probabilities
        P_bac_pair_conj = P_bac_pair_conj/(sum(P_bac_pair_conj));
        % Generate random numbers according to the length of the right column
        P_bac_pair_num = randperm(size(Combine_Mtx_F_Norm,1));
        % Rearrange the conjugation probability values on the right of
        % Combine_Mtx_F_Norm in accordance with the random numbers
        % generated
        New_Conj_Order = P_bac_pair_conj(:, P_bac_pair_num);
        % Combine the new conjugation probabilities on row 1 and the
        % corresponding random numbers on row 2
        P_conj_wthPosition_Random = [New_Conj_Order; P_bac_pair_num];
        
        % This cycles through the conjugation probabilities of Combine_Mtx_F_Norm
        r4 = rand();
        Bacteria_Found = 0;
        while (Bacteria_Found < 1)
            for col = 1:size(P_conj_wthPosition_Random,2)
                Conjugation_Select_Prob = P_conj_wthPosition_Random(1,col);
                if r4 < Conjugation_Select_Prob
                    Bacteria_Found = 1;
                    pair_selected = P_conj_wthPosition_Random(2,col);
                    break
                end
            end
            r4 = rand();
        end
        
        if Combine_Mtx_F_Norm(pair_selected,4) > 1
            Combine_Mtx_F_Norm(pair_selected,3) = Combine_Mtx_F_Norm(pair_selected,3) + 1;
            Combine_Mtx_F_Norm(pair_selected,4) = Combine_Mtx_F_Norm(pair_selected,4) - 1;
        end
        
        if Combine_Mtx_F_Norm(pair_selected,1) == 1
            x(il,itim) = Combine_Mtx_F_Norm(pair_selected,3);
        elseif Combine_Mtx_F_Norm(pair_selected,1) == 2
            m1(il,itim) = Combine_Mtx_F_Norm(pair_selected,3);
        elseif Combine_Mtx_F_Norm(pair_selected,1) == 3
            m2(il,itim) = Combine_Mtx_F_Norm(pair_selected,3);
        elseif Combine_Mtx_F_Norm(pair_selected,1) == 4
            m3(il,itim) = Combine_Mtx_F_Norm(pair_selected,3);
        end
        if Combine_Mtx_F_Norm(pair_selected,2) == 1
            x(il,itim) = Combine_Mtx_F_Norm(pair_selected,4);
        elseif Combine_Mtx_F_Norm(pair_selected,2) == 2
            m1(il,itim) = Combine_Mtx_F_Norm(pair_selected,4);
        elseif Combine_Mtx_F_Norm(pair_selected,2) == 3
            m2(il,itim) = Combine_Mtx_F_Norm(pair_selected,4);
        elseif Combine_Mtx_F_Norm(pair_selected,2) == 4
            m3(il,itim) = Combine_Mtx_F_Norm(pair_selected,4);
        end
        Conjugation_Selected = 0;
        
    else
        % Nothing Happened
        x(il,itim) = x(il,itim);
        m1(il,itim) = m1(il,itim);
        m2(il,itim) = m2(il,itim);
        m3(il,itim) = m3(il,itim);
    end

%% Given the Time Progression, Decide if the time is updated for the 
% Figure and Video, and Update the Positions of the Bacteria 
% Into the New Time Integer

    % next_num = itim + 1
    % xttim progress with decimal dt while itim and next_num are whole num
    if xttim < next_num
        itim = next_num - 1;
    elseif xttim >= next_num
        video_Condition = 1;
        oldtim = itim;
        itim = next_num;
        next_num = next_num + 1;
        x(:,itim) = x(:,oldtim);
        m1(:,itim) = m1(:,oldtim);
        m2(:,itim) = m2(:,oldtim);
        m3(:,itim) = m3(:,oldtim);
        % Also Need to Reset Migration Parameters
            Count_Num_Mig = x(1:nl,itim) + m1(1:nl,itim) + m2(1:nl,itim) + m3(1:nl,itim);
            All_Particles = sum(Count_Num_Mig); % Create a random order of all the particles to pick
            Rand_Part_Mtx_El = 0;
    end
    
%% RECORD THE POSITIONS OF ALL THE PARTICLES FOR THE NEXT LOOP
        for ill = 1:nl; % select position loop
           % before this, w and mut1 were originally set to zero
           % records number of particle at postion w and others
           % All of this is recorded for the plot in the next section:
                                                             % RECORD VIDEO
           w(ill,itim) = x(ill,itim);
           mut1(ill,itim) = m1(ill,itim);
           mut2(ill,itim) = m2(ill,itim);
           mut3(ill,itim) = m3(ill,itim);
           % For this "for" loop only, re-record the values of the old time
               % itim into the new time ixxx
               % This cycle continues before it reaches itim  + 1, 
               % continuously updating the number of particles into the 
               % new time the values at a new position get moved from time 
               % itim to the new time ixxx when it's at the boundary of itim
           ixxx = next_num; % next_num = itim + 1
           if xttim >= itim % itim = next_num - 1
                x(ill,ixxx) = x(ill,itim);
                m1(ill,ixxx) = m1(ill,itim);
                m2(ill,ixxx) = m2(ill,itim);
                m3(ill,ixxx) = m3(ill,itim);
           end
        end % end system position loop

%     % Count the number of bacteria in each Deme
%     Num_WT_Bac = 0;
%     Num_Mut1_Bac = 0;
%     Num_Mut2_Bac = 0;
%     Num_Mut3_Bac = 0;
%     if itim > 1    
%         Tot_Num = 0;
%         for il = 1:nl
%             nxtotal = x(il,itim) + m1(il,itim) + m2(il,itim) + m3(il,itim);
%             Tot_Num = Tot_Num + nxtotal;
%             Num_WT_Bac = Num_WT_Bac + x(il,itim);
%             Num_Mut1_Bac = Num_Mut1_Bac + m1(il,itim);
%             Num_Mut2_Bac = Num_Mut2_Bac + m2(il,itim);
%             Num_Mut3_Bac = Num_Mut3_Bac + m3(il,itim);
%         end
%     end
        
        
        %% RECORD VIDEO
        % The value added within the "mod" function of the "if statement" 
        % below determines how many seconds, (within the model), pass
        % for the program to take a screenshot of current system.
        time = itim;
        if (mod(itim,30) == 0) && video_Condition > 0;
            figure(f2);
            f2.Position = [300 100 800 840];
            clf
            % Plot the Theoretical Drift Velocity
            subplot(2, 1, 1);
            title('Theoretical Vd (Top) and Stochastic Simulation Algorithm (SSA) (Bottom)', 'fontsize', 13)
            yyaxis left
            plot(vd_chemotaxis,'b', 'LineWidth',2);
            ylabel('Theoretical Drift Velocity [µm/s]', 'fontsize',14)
            xlim([0 nl])
            ylim([0 5])
            yyaxis right
            plot(xbias,'r')
            ylabel('LB Concentration [µM]', 'fontsize',14)
            ylim([0 max(xbias)])
            xlabel ('Note: The LB Concentration is also in the SSA', 'fontsize', 13);
        
            % Plot the bacteria below
            subplot(2, 1, 2);
            xpos = zeros(nl,1);
            xpos(1) = 1;
            for il = 2:nl
                xpos(il) = xpos(il-1) + 1;
            end
            xdata1=xpos(1:nl);
            ydata1=w(1:nl,(itim));
            ydata1(ydata1==0)=nan;
            xdata2=xpos(1:nl);
            mutdata2=mut1(1:nl,(itim));
            mutdata2(mutdata2==0)=nan;
            xdata3=xpos(1:nl);
            mutdata3=mut2(1:nl,(itim));
            mutdata3(mutdata3==0)=nan;
            xdata4=xpos(1:nl);
            mutdata4=mut3(1:nl,(itim));
            mutdata4(mutdata4==0)=nan;
            yyaxis left
            plot(xdata1,ydata1,'b',xdata2,mutdata2,'r',xdata3,mutdata3,'g',xdata4,mutdata4,'m','LineWidth',2);
            xlim([1 nl])
            ylim([0 (CC+100)])
            yyaxis right
            plot(Cip_xbias)
            xlim([1 nl])
                % Legend
                Bacterial_Label_WT = 'Wild Type, Cip Fit = %d, Food Fit = %.2f';
                A = sprintf(Bacterial_Label_WT,Fit_WT,WT_FF);
                Bacterial_Label_M1 = 'Mutant 1, Cip Fit = %d, Food Fit = %.2f';
                Bacterial_Label_M2 = 'Mutant 2, Cip Fit = %d, Food Fit = %.2f';
                Bacterial_Label_M3 = 'Mutant 3, Cip Fit = %d, Food Fit = %.2f';
                legend({A,sprintf(Bacterial_Label_M1,Fit_m1,Mut_1_FF),...
                    sprintf(Bacterial_Label_M2,Fit_m2,Mut_2_FF),...
                    sprintf(Bacterial_Label_M3,Fit_m3,Mut_3_FF)},'Location','northwest');
            xlabel ('Deme Position (Length = 310µm)', 'fontsize', 16);
            Time_In_Min = itim/60;
            Time_In_Hours = itim/3600;
            title(['Time is ',num2str(Time_In_Min,'%4.2f'),' min = ',num2str(Time_In_Hours,'%4.2f'),' hours'], 'fontsize', 16)
            yyaxis left
            ylabel ('Number of Bacteria x 10', 'fontsize',14);
            yyaxis right
            ylabel ('Cipro Concentration [µg/mL]', 'fontsize',14);
            
            frame = getframe(gcf);
            writeVideo(v,frame)
    
            drawnow;
            video_Condition = 0;
        end
        
end %  end of time while loop
end % end iteration loop

% Num_WT_Bac
% Num_Mut1_Bac
% Num_Mut2_Bac
% Num_Mut3_Bac
% Lose_Bacteria
% New_Bacteria
% Remaining_Bacteria = Tot_Num
% Mutation_Occurs
% WildType
% Mutant1_Grows
% Mutant2
% Conjugation_Occurs
Migration_Occurs

% Close the video file recorded
close(v);

% ttim is the total time
% Normalize y for total amount of iterations
for itim = 1:ttim
    for il = 1:nl
        % For each iteration the average bacteria is recorded and averaged
        w(il,itim) = w(il,itim)/(itertot);
        mut1(ill,itim) = mut1(ill,itim)/(itertot);
        mut2(ill,itim) = mut2(ill,itim)/(itertot);
        mut3(ill,itim) = mut3(ill,itim)/(itertot);
    end
end

% Image of the model after it has finished
figure(f2);
f2.Position = [300 100 800 840];
clf
% Plot the Theoretical Drift Velocity
subplot(2, 1, 1);
            title('Theoretical Vd (Top) and Stochastic Simulation Algorithm (SSA) (Bottom)', 'fontsize', 13)
yyaxis left
plot(vd_chemotaxis,'b', 'LineWidth',2);
ylabel('Theoretical Drift Velocity [µm/s]', 'fontsize',14)
xlim([0 nl])
ylim([0 5])
yyaxis right
plot(xbias,'r')
ylabel('LB Concentration [µM]', 'fontsize',14)
ylim([0 max(xbias)])
xlabel ('Note: The LB Concentration is also in the SSA', 'fontsize', 13);

% Plot the bacteria below
subplot(2, 1, 2);
xpos = zeros(nl,1);
xpos(1) = 1;
for il = 2:nl
    xpos(il) = xpos(il-1) + 1;
end
xdata1=xpos(1:nl);
ydata1=w(1:nl,(itim));
ydata1(ydata1==0)=nan;
xdata2=xpos(1:nl);
mutdata2=mut1(1:nl,(itim));
mutdata2(mutdata2==0)=nan;
xdata3=xpos(1:nl);
mutdata3=mut2(1:nl,(itim));
mutdata3(mutdata3==0)=nan;
xdata4=xpos(1:nl);
mutdata4=mut3(1:nl,(itim));
mutdata4(mutdata4==0)=nan;
yyaxis left
plot(xdata1,ydata1,'b',xdata2,mutdata2,'r',xdata3,mutdata3,'g',xdata4,mutdata4,'m','LineWidth',2);
xlim([1 nl])
ylim([0 (CC+100)])
yyaxis right
plot(Cip_xbias)
xlim([1 nl])
    Bacterial_Label_WT = 'Wild Type, Cip Fit = %d, Food Fit = %.2f';
    A = sprintf(Bacterial_Label_WT,Fit_WT,WT_FF);
    Bacterial_Label_M1 = 'Mutant 1, Cip Fit = %d, Food Fit = %.2f';
    Bacterial_Label_M2 = 'Mutant 2, Cip Fit = %d, Food Fit = %.2f';
    Bacterial_Label_M3 = 'Mutant 3, Cip Fit = %d, Food Fit = %.2f';
    legend({A,sprintf(Bacterial_Label_M1,Fit_m1,Mut_1_FF),...
        sprintf(Bacterial_Label_M2,Fit_m2,Mut_2_FF),...
        sprintf(Bacterial_Label_M3,Fit_m3,Mut_3_FF)},'Location','northwest');
xlabel ('Deme Position (Length = 310µm)', 'fontsize', 16);
Time_In_Min = itim/60;
Time_In_Hours = itim/3600;
title(['Time is ',num2str(Time_In_Min,'%4.2f'),' min = ',num2str(Time_In_Hours,'%4.2f'),' hours'], 'fontsize', 16)
yyaxis left
ylabel ('Number of Bacteria x 10', 'fontsize',14);
yyaxis right
ylabel ('Cipro Concentration [µg/mL]', 'fontsize',14);
