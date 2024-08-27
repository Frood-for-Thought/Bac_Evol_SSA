    %% Function to Pick one random particle on the Bacterial arrays
    
    function [WT_Selected,Mut1_Selected,Mut2_Selected,Mut3_Selected,Mut4_Selected,il] ...
            = Pick_Particle_Function(Tot_Num,nl,itim,x,m1,m2,m3,m4)
    
        WT_Selected = 0; % Wild Type Bacteria Selected Condition
        Mut1_Selected = 0; % Mutant 1 Bacteria Selected Condition
        Mut2_Selected = 0; % Mutant 2 Bacteria Selected Condition
        Mut3_Selected = 0; % Mutant 3 Bacteria Selected Condition
        Mut4_Selected = 0; % Mutant 4 Bacteria Selected Condition
        % Out of the total particles present pick one particle
        PickParticle = round(rand()*Tot_Num);
        for il = 1:nl % select position loop
            nxtotal = x(il,itim)+ m1(il,itim) + m2(il,itim) + m3(il,itim) + m4(il,itim);
            PickParticle = PickParticle - nxtotal;
            % When PickParticle goes negative, then the position has been selected
            if (PickParticle <= 0) && (nxtotal > 0)
                ParticlesAtPosition = PickParticle + nxtotal;
                % The particle is a WT
                if (ParticlesAtPosition <= x(il,itim)) && (x(il,itim) > 0)
                    WT_Selected = 1;
                % The particle is Mutant 1
                elseif (ParticlesAtPosition > x(il,itim)) && (ParticlesAtPosition <= (x(il,itim) + m1(il,itim))) && (m1(il,itim) > 0)
                    Mut1_Selected = 1;
                elseif (ParticlesAtPosition > (x(il,itim) + m1(il,itim))) && (ParticlesAtPosition <= (x(il,itim) + m1(il,itim)+ m2(il,itim))) && (m2(il,itim) > 0)
                    Mut2_Selected = 1;
                elseif (ParticlesAtPosition > (x(il,itim) + m1(il,itim)+ m2(il,itim))) && (ParticlesAtPosition <= nxtotal)&& (m3(il,itim) > 0)
                    Mut3_Selected = 1;
                elseif (ParticlesAtPosition > (x(il,itim) + m1(il,itim)+ m2(il,itim) + m3(il,itim))) && (ParticlesAtPosition <= nxtotal) && (m4(il,itim) > 0)
                    Mut4_Selected = 1;
                end
                % the "break" makes it so that location "il" is selected
                break;
            end
        end
        
    end