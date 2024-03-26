function updated = gating_variable_update(V, variable, type, Vt)
    V = V .* 1e3;
    Vt = Vt .* 1e3;
    if type == 'p'
        
    else
        if type == 'n'
            alpha = ((V - Vt) -15) .* (-0.032 ./ (exp(-(V-Vt-15)/5) -1));
            beta = 0.5 * exp(-(V-Vt-10) ./ 40);
        elseif type == 'm'

        else

        end
        
        updated = alpha.*(1-variable) - beta.*variable;

    end