
function [vd_chemotaxis,c_df_over_dc, Vo_max, kaon,kaoff,kson,ksoff] = ...
    Attractant_Mig_Function(nl,Grad,xbias,Max_Concentration)

% This model now has the Grad variable controlling the max speed up to when 
% a saturation gradient has a max speed when the run time is comparable 
% with tumble time.  As well, the receptor sensitivity from the receptor
% free-energy difference controls the peaks at concentrations with optimum
% free-energy and troughs at saturated concentrations.  The maximum drift 
% velocity increases up until a maximum running speed while the critical
% gradient decreases down until the lowest critical gradient.

%% The Receptor Free Energy Equation
kaon = 500; % in µM
kaoff = 18; % in µM
kson = 1000000; % in µM
ksoff = 100000; % in µM
va = 1/3; % Tar receptor fraction
vs = 2/3; % Tsr receptor fraction
N = 6; % cooperrative receptor factor

% Define function
c1 = zeros();
vd = zeros();
df_over_dc_va = zeros(); % 
df_over_dc_vs = zeros();
c_df_over_dc = zeros();
% c1 = linspace(1, 3000, nl);
for i = 1:nl;
    c1(i) = xbias(i);
%     c1(i) = 100*exp(Grad*300*i);
    df_over_dc_va(i) = va*((kaon - kaoff)/((c1(i)+kaoff)*(c1(i)+kaon)));
    df_over_dc_vs(i) = vs*((kson - ksoff)/((c1(i)+ksoff)*(c1(i)+kson)));
%     df_over_dc_vs(i) = 0;
    c_df_over_dc(i) = c1(i)*N*(df_over_dc_va(i) + df_over_dc_vs(i));
end

%% Find the Max Receptor Free Energy Sensitivity
% c_df_over_dc = F(c)
% at dF/dc = 0, c(max) is the concentration where the response is highest
% for the receptor regions between Ki < c < Ka
if Max_Concentration < kaon
    % The Mobility Constant is approximated in the regions where 
    % kaoff < c < kaon
    c_max = sqrt(kaon*kaoff);
    Max_c_df_over_dc_input_c_max = c_max*N*va*((kaon - kaoff)/((c_max+kaoff)*(c_max+kaon)));
elseif (Max_Concentration >= kaon) && (Max_Concentration < ksoff)
    % The Mobility Constant is approximated in the regions where 
    % kaoff < c < ksoff
    c_max = sqrt(kson*ksoff);
    Max_c_df_over_dc_input_c_max = max(c_df_over_dc);
elseif (Max_Concentration >= ksoff)
    c_max = sqrt(kson*ksoff);
    Max_c_df_over_dc_input_c_max = max(c_df_over_dc);
end
Max_c_df_over_dc = max(c_df_over_dc);

%% Calculating Max Drift Velocity
% These calculations were obtained from the paper 
%   "A Pathway-based Mean-field Model for Escherichia coli Chemotaxis"
% the maximum vd is reached when tr is comparable with t_theta
% the approximation is what is physically obtainable by the bacteria
z_theta = 0.14; % s^-1, the rotational diffusion constant, (Dr)
Ts = 0.8; % s^-1, the average switching time from CCW to CW
kR = 0.005; % s^-1, the methylation rate, or the adaptation rate
% Methylation Free Energy, alpha = 1.7
A = (1.7*kR*(1-(z_theta*Ts)^(1/10)));
Vo = 16/sqrt(2); % µm/s
B = (10*6*(Vo^2)*(Ts^-1)*(z_theta*Ts)*(1-0.5*(z_theta*Ts)^(1/10)));
C = (2*z_theta)^-2;
% Gc = (alpha*F(ac)/X(ac))
Gc = (A/(B*C))^0.5; % µm^-1
% The mobility constant is modulated by the tumble frequency which can be
% approximated by the value X(a) -> pX(a)
p = 1.15;
Gc = Gc/sqrt(p);
% Vd_max = (alpha*F(ac)*X(ac)^1/2
Vd_max = (A*B*C)^0.5; % µm/s
Vd_max = Vd_max*sqrt(p);
Critical_Grad_Label = 'The critical gradient at which the drift velocity is max in the linear Tar region is %.2d µm-1';
Max_Drift_Vel_Label = 'The max drift velocity in the linear Tar region is around %.2f µm/s';
Critical_Gradient_Speed = sprintf(Critical_Grad_Label,Gc)
Max_Drift_Velocity = sprintf(Max_Drift_Vel_Label, Vd_max)
% In higher concentrations the run time can increase.  This is the drift
% velocity from the bacteria's fastest run speed
Vo = 25/sqrt(2); % µm/s
B = (10*6*(Vo^2)*(Ts)*(z_theta*Ts)*(1-0.5*(z_theta*Ts)^(1/10)));
Max_Drift_Fastest_Run = (A*B*C)^0.5; % µm/s
Max_Drift_Fastest_Run = Max_Drift_Fastest_Run*sqrt(p);
Max_Drift_FR_Label = 'The bacterias physical max drift velocity at the maximum run speed is around %.2f µm/s';
Fastest_Drift_Vel = sprintf(Max_Drift_FR_Label,Max_Drift_Fastest_Run)
Lowest_Grad_Fastest_Run = (A/(B*C))^0.5; % µm^-1
Lowest_Grad_Fastest_Run = Lowest_Grad_Fastest_Run/sqrt(p);
Low_Grad_FR_Label = 'The lowest gradient at the bacterias physical maximum run speed, (with conc. >= max receptor conc.), is around %.2d µm^-1';
Lowest_Grad_Speed = sprintf(Low_Grad_FR_Label,Lowest_Grad_Fastest_Run)

%%  This is calculating the maximum speed in various concentrations
% As the concentration increases the max drift speed increases until it
% reaches the value Max_Drift_Fastest_Run
Run_Speed_Diff = Max_Drift_Fastest_Run - Vd_max;
                    % Adding "R" here makes the max speed plateau 
                    % around Vd_max at regions between Ki < c < Ka
Conc_Ratio = Max_Concentration/sqrt(kson*ksoff);% - 0.035;
if Conc_Ratio >= 1                            % R = _.____
    Conc_Ratio = 1;
end
Vd_max = Vd_max + Run_Speed_Diff*Conc_Ratio;
Vo_max = 16/sqrt(2) + ((25-16)/sqrt(2))*Conc_Ratio;
Gc = Gc - (Gc - Lowest_Grad_Fastest_Run)*Conc_Ratio;

%% Calculating Mobility Constant
% Calculating the susceptibility when c*df/dc is max and vd_max is reached
% Initially calculating the Vd_max and inserting it into the suceptibility
% equation makes sure that the critical velocity is reached and any
% gradient over Gc will still only increase the speed to Vd_max

% The model will use a Mobility Constant approximation which calculates the
% Vd in the linear region between Ki < c < Ka as a limitation of the
% average run speed
Xo = Vd_max/(Max_c_df_over_dc*Grad);
% In the repellant paper, as the concentration increases, then the run
% speed, Vo, increases thereby increasing the Vd_max, which could explain
% higher speeds at greater concentrations.

% The Xo was calculated previously for the linear region.  What the "if 
% statement" below is saying is that when the gradient is lower than the 
% critical gradient then Xo is calculated from vd_max/F(c_max)*Grad 
% within the maximum of the linear region
if (Grad < Gc)
    Xo = Vd_max/(Max_c_df_over_dc_input_c_max*Gc);
end
for i = 1:nl
    % Xo being constant is a limitation due to the Grad's ability to
    % continuously increase the velocity, vd.  A more advanced model has
    % X(a) adapt to the gradient, which is why there is a saturation
    % gradient Gc
    vd(i) = Xo*Grad*c_df_over_dc(i);
end
Current_Gradient = Gc;
Current_Gradient_Label = 'The current critical gradient at this max concentrations run speed is %.2d µm^-1';
Current_Critical_Gradient = sprintf(Current_Gradient_Label, Current_Gradient)
Current_Gradient_Max_Velocity = max(vd);
Current_Max_Drift_Vel_Label = 'The max drift velocity at gradient %.2d µm^-1, and Max Conc %d µM, is %.2f µm/s';
Current_Max_Drift_Velocity = sprintf(Current_Max_Drift_Vel_Label, Grad, Max_Concentration, Current_Gradient_Max_Velocity)
% This model now has the Grad variable controlling the max speed up to when 
% a saturation gradient has a max speed when the run time is comparable 
% with tumble time.  As well, the receptor sensitivity from the receptor
% free-energy difference controls the peaks at concentrations with optimum
% free-energy and trophs at saturated concentrations
vd_chemotaxis = vd;

end