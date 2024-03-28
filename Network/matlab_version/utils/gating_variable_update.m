function updated = gating_variable_update(V, variable, type, Vt, tau_max)
% input V is membrane potential; variable is the gating variable to update;
% type is string n, m, h, p, q, or s; 
    V = V .* 1e3;   % convert to mV unit
    Vt = Vt .* 1e3;
    if type == 'p'
        p_inf = 1 ./(exp((-V-35)/10)+1);
        tau_p = tau_max ./ (3.3*exp((V+35)/20) + exp((-V-35)/20));
        updated = (p_inf - variable) ./ tau_p;

    elseif type == 'r'
        tau_r = .5e-3;  % ms
        tau_d = 8e-3;   % ms
        V_0 = -20e-3;   % mV
        updated = (1/tau_r - 1/tau_d) * (1 - variable)./(1+exp(-V+V_0)) - variable/tau_d;
    else
        if type == 'n'
            alpha = (V-Vt-15) .* (-0.032 ./ (exp(-(V-Vt-15)/5) -1));
            beta = 0.5 * exp(-(V-Vt-10) ./ 40);

        elseif type == 'm'
            alpha = (V-Vt-13) .* (-0.32 ./ (exp((-V+Vt+15)/5 -1)));
            beta = .28 * (exp((V-Vt-40) ./ 5)) -1;

        elseif type == 'h'
            alpha = .128 * exp(-(V-Vt-17)/18);
            beta = 4 ./ (exp(-(V-Vt-40)/5) + 1);

        elseif type == 'q'
            alpha = .0055 * (-27-V) ./ (exp((-27-V)/3.8)-1);
            beta = .94 * exp((-75-V)/17);

        elseif type == 's'
            alpha = .000457 * exp((-13-V)/50);
            beta = .0065 ./ (exp((-15-V)/28) + 1);

        end
        
        updated = alpha.*(1-variable) - beta.*variable;

    end