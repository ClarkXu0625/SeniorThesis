addpath('utils')
import gating_variable_update.*


m = zeros(1,150);
V_t = -56.6e-3 * ones(1,150);
V = -70e-3 * ones(1,150);
updated = gating_variable_update(V, m, 'n', V_t);

