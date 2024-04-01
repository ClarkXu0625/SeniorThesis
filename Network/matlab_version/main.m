addpath('utils')
import param_to_array.*
import gating_variable_update.*

% def param
% three space in array matches param for FS, RSA, and IB neuron
C_m = [0.5, 1, 1] .* 1e-2;          % uF/cm^2
V_k = [-90, -90, -90] .* 1e-3;      % mV
V_ca = [0, 0, 120] .* 1e-3;         % mV
V_na = [50, 56, 50] .* 1e-3;        % mV
V_l = [-70, -70.3, -70] .* 1e-3;        % mV
V_t = [-52.6, -52.6, -52.6] .* 1e-3;    % mV
g_k = [10, 6, 5] .* 10;           % mS/cm^2
g_m = [0, .075, .03] .* 10;       % mS/cm^2
g_ca = [0, 0, .2] .* 10;          % mS/cm^2
g_na = [56, 56, 50] .* 10;        % mS/cm^2
g_l = [1.5e-2, 2.05e-2, .01] .* 10;   % mS/cm^2
tau_max = [1, 608, 608] .* 1e-3;    % ms

% Sypase params
tau_r = [.5, .5, .5] .* 1e-3;   % ms
tau_d = [8, 8, 8] .* 1e-3;      % ms
V_syn = [-80, 20, 20] .* 1e-3;  % mV
V_0 = [-20, -20, -20] .* 1e-3;  % mV

% creating arrays
N_FS = 30;
N_RSA = 90;
N_IB = 30;
N = N_FS + N_RSA + N_IB;

% convert param array to row vector with length N
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
tau_r = param_to_array(tau_r);
tau_d = param_to_array(tau_d);
V_syn = param_to_array(V_syn);
V_0 = param_to_array(V_0);

% initiate gating variables
n = zeros(N, 1);
m = zeros(size(n));
h = zeros(size(n));
p = zeros(size(n));
q = zeros(size(n));
s = zeros(size(n));
r = zeros(size(n));

tmax = 10;          % total simulation time
dt = 1e-4;          % 0.1ms time step
t = 0: dt: tmax;    % time vector
Nt = length(t);
V = zeros(N, Nt);
V(:,1) = V_l;

% define connectivity matrix
E_el = rand(N)*.06e-3;    % connectivity of electrical synapses, 0-0.06mS
E_ch = rand(N)*.1e-3;     % connectivity of chemical synapses, 0-0.1mS, also defines the adacency matrix
I_max = 0.1e-2;           % 0.1 uA/cm^2 maximum amplitude as input current

% set random piecewise input current amplitude..
% maximum aplitude set to 0.1 uA
piece_duration = 0.1;           % every piece is 0.5 second duration
piece = 0:piece_duration:tmax;     
j = 1;
amplitude = rand(N,1) * I_max;


for i = 2:Nt
    Vi = V(:,i-1);

    % function for piecewise random amplitude
    % update amplitude to another random value in range of 0 to maximum
    % amplitude for every time piece.
    if piece(j) <= t(i)
        amplitude = rand(N,1) * I_max;
        j = j+1;
    end
    I_inj = amplitude;
    
    % update gating variable before each run
    dmdt = gating_variable_update(Vi, m, 'm', V_t, tau_max);
    dndt = gating_variable_update(Vi, n, 'n', V_t, tau_max);
    dhdt = gating_variable_update(Vi, h, 'h', V_t, tau_max);
    dpdt = gating_variable_update(Vi, p, 'p', V_t, tau_max);
    dqdt = gating_variable_update(Vi, q, 'q', V_t, tau_max);
    dsdt = gating_variable_update(Vi, s, 's', V_t, tau_max);
    drdt = gating_variable_update(Vi, r, 'r', V_t, tau_max);


    m = m + dt*dmdt;
    n = n + dt*dndt;
    h = h + dt*dhdt;
    p = p + dt*dpdt;
    q = q + dt*dqdt;
    s = s + dt*dsdt;
    r = r + dt*drdt;

    % synapsic factor
    % electric synapse
    S = ones(N,1);      % S is the selection column vector determine which neuron receives it
    A_el = E_el;
    D_sum_el = sum(E_el, 2);
    D_el = zeros(N);
    for k = 1: N
        D_el(k, k) = D_sum_el(k);
        A_el(k, k) = 0;
    end
    U_el = -(D_el - A_el) * Vi;

    % chemical synapse
    % U_ch = E_ch * (I_inj .* kron(ones(1, N), r) .* (V_syn - Vi).');
    U_ch = zeros(N, 1);
    for k = 1:N
        temp = 0;
        for l = 1:N
            temp = temp + E_ch(k,l) * r(l) * (V_syn(k) - Vi(l));
        end
        U_ch(k) = temp;
    end
    
    % update V
    dvdt = 1./ C_m .* (I_inj .* S - ...
        g_k .* (n.^4).*(Vi-V_k) -...
        g_m .* p.*(Vi-V_k) -...
        g_ca .* (q.^2).*s.*(Vi-V_ca) -...
        g_na .* (m.^3).*h.*(Vi-V_na) -...
        g_l .* (Vi-V_l) + U_ch + U_el);
    
    V(:, i) = Vi+dt*dvdt;

end

plot(t, V(1,:))


