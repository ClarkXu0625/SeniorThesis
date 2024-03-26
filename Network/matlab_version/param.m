import param_to_array.*
% def param
C_m = [0.5, 1, 1] .* 1e-6;
V_k = [-90, -90, -90] .* 1e-3;
V_ca = [0, 0, 120] .* 1e-3;
V_na = [50, 56, 50] .* 1e-3;
V_l = [-70, -70.3, -70] .* 1e-3;
V_t = [-52.6, -52.6, -52.6] .* 1e-3;
g_k = [10, 6, 5] .* 1e-3;
g_m = [0, .075, .03] .* 1e-3;
g_ca = [0, 0, .2] .* 1e-3;
g_na = [56, 56, 50] .* 1e-3;
g_l = [1.5e-2, 2.05e-2, .01] .* 1e-3;
tau_max = [1, 608, 608] .* 1e-3;

% creating arrays
N_FS = 30;
N_RSA = 90;
N_IB = 30;
N = N_FS + N_RSA + N_IB;

V = zeros(1, N);

C_m = param_to_array(C_m);

% gating variables
n = zeros(1, N);
m = zeros(size(n));
h = zeros(size(n));
p = zeros(size(n));
q = zeros(size(n));
s = zeros(size(n));
