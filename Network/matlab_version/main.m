addpath('utils')
import param_to_array.*
import gating_variable_update.*

% def param
% three space in array matches param for FS, RSA, and IB neuron
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

% convert param array to length N
C_m = param_to_array(C_m);
V_k = param_to_array(V_k);
V_ca = param_to_array(V_ca);
V_na = param_to_array(V_na);
V_l = param_to_array(V_l);
V_t = param_to_array(V_t);
g_k = param_to_array(g_k);
g_m = param_to_array(g_m);
g_ca = param_to_array(g_ca);
g_na = param_to_array(g_na);
g_l = param_to_array(g_l);
tau_max = param_to_array(tau_max);


% initiate gating variables
n = zeros(1, N);
m = zeros(size(n));
h = zeros(size(n));
p = zeros(size(n));
q = zeros(size(n));
s = zeros(size(n));

tmax = 10;          % total simulation time
dt = 1e-4;          % 0.1ms time step
t = 0: dt: tmax;    % time vector
Nt = length(t);
V = zeros(Nt, N);
V(1,:) = V_l;

% define connectivity matrix
E_el = rand(N)*.06e-3;    % connectivity of electrical synapses, 0-0.06mS
E_ch = rand(N)*.1e-3;     % connectivity of chemical synapses, 0-0.1mS
I_max = 1e-7;           % 0.1 uA maximum amplitude as input current


for i = 2:Nt
    I_inj = zeros(1, N);
    
    % update gating variable before each run
    dmdt = gating_variable_update(V(i-1,:), m, 'm', V_t, tau_max);
    dndt = gating_variable_update(V(i-1,:), n, 'n', V_t, tau_max);
    dhdt = gating_variable_update(V(i-1,:), h, 'h', V_t, tau_max);
    dpdt = gating_variable_update(V(i-1,:), p, 'p', V_t, tau_max);
    dqdt = gating_variable_update(V(i-1,:), q, 'q', V_t, tau_max);
    dsdt = gating_variable_update(V(i-1,:), s, 's', V_t, tau_max);

    m = m + dt*dmdt;
    n = n + dt*dndt;
    h = h + dt*dhdt;
    p = p + dt*dpdt;
    q = q + dt*dqdt;
    s = s + dt*dsdt;

    
    dvdt = 1./ C_m .* (I_inj - ...
        g_k .* (n.^4).*(V(i-1,:)-V_k) -...
        g_m .* p.*(V(i-1,:)-V_k) -...
        g_ca .* (q.^2).*s.*(V(i-1,:)-V_ca) -...
        g_na .* (m.^3).*h.*(V(i-1,:)-V_na) -...
        g_l .* (V(i-1,:)-V_l));
    
    V(i,:) = V(i-1,:) + dt*dvdt;

end


