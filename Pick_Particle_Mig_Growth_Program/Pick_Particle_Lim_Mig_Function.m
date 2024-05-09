    %% Function to Pick one random particle on the Bacterial arrays 
     % and to not Reselect it
    
    function [WT_Selected,Mut1_Selected,Mut2_Selected,Mut3_Selected,i,...
            Count_Num_Mig,All_Prt_Ran_Order,Rand_Part_Mtx_El,R_Mig,if_sel] ...
            = Pick_Particle_Lim_Mig_Function(Rand_Part_Mtx_El,Count_Num_Mig,...
            All_Prt_Ran_Order,nl,itim,x,m1,m2,m3,R_Mig)
        if_sel = 0; % Check if nothing was selected for the pickparticle if statement
        WT_Selected = 0; % Wild Type Bacteria Selected Condition
        Mut1_Selected = 0; % Mutant 1 Bacteria Selected Condition
        Mut2_Selected = 0; % Mutant 2 Bacteria Selected Condition
        Mut3_Selected = 0; % Mutant 3 Bacteria Selected Condition
        % Out of the total particles present pick one particle
        Rand_Part_Mtx_El = Rand_Part_Mtx_El + 1;
        % All_Prt_Ran_Order creates a random order of 
        % total number of particles to pick. 
        % Tot_Num = sum(Count_Num_Mig); gives the total number of particles
        % to pick given the number of allowed migrations.
        % Rand_Part_Mtx_El goes through the list All_Prt_Ran_Order to
        % determine which particle to pick.  If it gets to the end of the
        % list then there are no more particles to pick and the rate of
        % migration is over til the next time unit, (xttim >= next_num)
        if Rand_Part_Mtx_El > length(All_Prt_Ran_Order)
            R_Mig = 0;
            i = NaN;
            return
        end
        % PickParticle is the particle number chosen out of the total
        % amount to choose from.  It counts the number of particles
        % starting at Deme one then moves towards Deme nl.
        PickParticle = All_Prt_Ran_Order(Rand_Part_Mtx_El);
        % This is here to restart the following for loop if a new 
        % All_Prt_Ran_Order matrix is constructed to go through all the 
        % allowed bacteria to move again.
        pos_selected = 0;
        while pos_selected < 1
            for i = 1:nl % select position loop
                nxtotal = x(i,itim)+ m1(i,itim) + m2(i,itim) + m3(i,itim);
                % Removes the total number of particles at this Deme from the
                % picked particle number
                PickParticle = PickParticle - nxtotal;
                % If the number of migrations are allowed in the 
                % matrix which counts the number of migrations per Deme
                if Count_Num_Mig(i) >= 1
                    if_sel = 1; % This if statement was selected
                    % When PickParticle goes negative, then the position has been selected
                    if (PickParticle <= 0) && (nxtotal > 0)
                        Count_Num_Mig(i) = Count_Num_Mig(i) - 1;
                        ParticlesAtPosition = PickParticle + nxtotal;
                        % The particle is a WT
                        if (ParticlesAtPosition <= x(i,itim)) && (x(i,itim) > 0)
                            WT_Selected = 1;
                        % The particle is Mutant 1
                        elseif (ParticlesAtPosition > x(i,itim)) && (ParticlesAtPosition <= (x(i,itim) + m1(i,itim))) && (m1(i,itim) > 0)
                            Mut1_Selected = 1;
                        elseif (ParticlesAtPosition > (x(i,itim) + m1(i,itim))) && (ParticlesAtPosition <= (x(i,itim) + m1(i,itim)+ m2(i,itim))) && (m2(i,itim) > 0)
                            Mut2_Selected = 1;
                        elseif (ParticlesAtPosition > (x(i,itim) + m1(i,itim)+ m2(i,itim))) && (ParticlesAtPosition <= nxtotal)&& (m3(i,itim) > 0)
                            Mut3_Selected = 1;
                        end
                        % the "break" makes it so that location "il" is selected
                        pos_selected = 1;
                        break
                    end
                % If the number of migrations can take place but the
                % population of the section has no bacteria, function moves to
                % the next Rand_Part_Mtx_El
                elseif (Count_Num_Mig(i) >= 1) && (nxtotal <= 0)
                    if_sel = 2;
                    Rand_Part_Mtx_El = Rand_Part_Mtx_El + 1;
                    if Rand_Part_Mtx_El > length(All_Prt_Ran_Order)
                        R_Mig = 0;
                        Count_Num_Mig = x(1:nl,itim) + m1(1:nl,itim) + m2(1:nl,itim) + m3(1:nl,itim);
                        All_Prt_Ran_Order = randperm(sum(Count_Num_Mig)); % Create a random order of all the particles to pick
                        Rand_Part_Mtx_El = 1;
                    end
                    PickParticle = All_Prt_Ran_Order(Rand_Part_Mtx_El);
                    continue
                end
                % If the number of migrations are not allowed at this location
                % function moves to the next position i
            end
        % Go back to beginning and restart the if statement because
        % no position was selected
            if pos_selected < 1
                Rand_Part_Mtx_El = Rand_Part_Mtx_El + 1;
                if Rand_Part_Mtx_El > length(All_Prt_Ran_Order)
                    R_Mig = 0;
                    break
                end
                PickParticle = All_Prt_Ran_Order(Rand_Part_Mtx_El);
            end
        end
    end