clear all



%% Parameters
trialN = 1000;                      % number of trials
d = 1;                              % scene depth(m)


% Lighting conditions
r_a = 1.0;                          % relative ambient light strength: r_a = e_a/e_s
r_i = 1.0;                          % relative interfering signal strength: r_i = e_i/e_s
N = 5;                              % number of interfering cameras


% ToF parameters
f_mod = 20e6;                       % modulation frequency(Hz)
T = 10e-3;                          % total integration time(s)


% Number of generated electrons
e_s = 1e7;                          % average number of signal photons
e_a = r_a*e_s;                      % average number of ambient photons
e_i = r_i*e_s;                      % average number of interfering photons


% Misc
c = 3e8;                            % Light speed(m/s)
d_max = c/(2*f_mod);                % Maximum measurable distance(m)


% Parameters for our approaches
A = 7;                              % Peak power amplification
M = 500;                            % Number of slots


% Parameters for PN
stageN = 7;                         % Number of LFSR stages
bitN = 2^stageN - 1;                % Number of bits
sampleNperBit = 1000;               % Number of samples per bit


% Photon energy
h = 6.62607015e-34;                 % Planck constant
lambda = 900e-9;                    % source wavelength
E_photon = h*c/lambda;              % unit photon energy



%% Repeat depth estimation and get depth standard deviations


% dummy variables for various parameters
dummyVariables = 1 : 1 : 10;            % N   [1 2 3 4 5 6 7 8 9 10]
% dummyVariables = [10 : 10 : 90]*1e-3;   % T
% dummyVariables = [10 : 10 : 90]*1e6;    % f_mod



% buffers for depth standard deviations
stdSet_PN_sim = zeros(size(dummyVariables, 2), 1);
stdSet_ACO_sim = zeros(size(dummyVariables, 2), 1);
stdSet_SEC_sim = zeros(size(dummyVariables, 2), 1);
stdSet_SECsync_sim = zeros(size(dummyVariables, 2), 1);
stdSet_CMB_sim = zeros(size(dummyVariables, 2), 1);

stdSet_ACO_eq = zeros(size(dummyVariables, 2), 1);
stdSet_SEC_eq = zeros(size(dummyVariables, 2), 1);
stdSet_CMB_eq = zeros(size(dummyVariables, 2), 1);


% loop over dummy variables
idx = 1;

for N = dummyVariables
% for T = dummyVariables
% for f_mod = dummyVariables
    
    idx
    
    
    % optimal slot ON probability
    p_SEC = min(1/(2*N+1), 1/A);            % for SEC
    p_CMB = 1/A;                            % for CMB
    
    
    % adjust parameters for fair comparisons
    f_mod_PN = f_mod/bitN;
    T_SEC = T/(A*p_SEC);
    
    
    % buffers for estimated depths
    dSet_PN = zeros(trialN, 1);
    dSet_ACO = zeros(trialN, 1);
    dSet_SEC = zeros(trialN, 1);
    dSet_SECsync = zeros(trialN, 1);
    dSet_CMB = zeros(trialN, 1);
    
    for trial = 1 : trialN
        
        
        % PN sequence approach
        d_hat = estimateDepth_PN(d, c, N, e_s, e_a, e_i, f_mod_PN, T, stageN, sampleNperBit);
        dSet_PN(trial, 1) = d_hat;

    
        % ACO
        d_hat = estimateDepth_ACO(d, c, N, e_s, e_a, e_i, f_mod, T);
        dSet_ACO(trial, 1) = d_hat;
        

        % SEC
        d_hat = estimateDepth_SEC(d, c, p_SEC, N, M, A, e_s, e_a, e_i, f_mod, T_SEC);
        dSet_SEC(trial, 1) = d_hat;
        
        
        % SEC_sync
        d_hat = estimateDepth_SECsync(d, c, p_SEC, N, M, A, e_s, e_a, e_i, f_mod, T_SEC);
        dSet_SECsync(trial, 1) = d_hat;
        
        
        % CMB
        d_hat = estimateDepth_CMB(d, c, p_CMB, N, M, A, e_s, e_a, e_i, f_mod, T);
        dSet_CMB(trial, 1) = d_hat;


    end
    
    
    % save depth standard deviations by simulations
    stdSet_PN_sim(idx, 1) = std(dSet_PN);
    stdSet_ACO_sim(idx, 1) = std(dSet_ACO);
    stdSet_SEC_sim(idx, 1) = std(dSet_SEC);
    stdSet_SECsync_sim(idx, 1) = std(dSet_SECsync);
    stdSet_CMB_sim(idx, 1) = std(dSet_CMB);
    
    
    % save depth standard deviations by derived equations
    stdSet_ACO_eq(idx, 1) = c/(2*sqrt(2)*pi*f_mod*sqrt(T))*sqrt(e_s + e_a + N*e_i)/e_s;
    p_noclsh = p_SEC*(1-p_SEC)^(2*N);
    stdSet_SEC_eq(idx, 1) = c/(2*sqrt(2)*pi*f_mod*sqrt(T_SEC*p_noclsh))*sqrt(A*e_s + e_a)/(A*e_s);
    stdSet_CMB_eq(idx, 1) = c/(2*sqrt(2)*pi*f_mod*sqrt(T*p_CMB))*sqrt(A*e_s + e_a + p_CMB*N*A*e_i)/(A*e_s);
       
    idx = idx + 1;
end



%% Comparisons (lower is better)
colors = [
  1 0 0;
  1 0.7 0
  0.2 0.5 0.2;
  0.2 1 0.2
  0 0 1
  0.8 0 0.8
];


figure; hold on; grid on;
% plot(dummyVariables', stdSet_PN_sim, 'color', [0, 0, 0], 'lineWidth', 4);
plot(dummyVariables', stdSet_ACO_sim, 'color', colors(1, :), 'lineWidth', 4);
plot(dummyVariables', stdSet_ACO_eq, ':', 'color', colors(2, :), 'lineWidth', 4);
plot(dummyVariables', stdSet_SEC_sim, 'color', colors(3, :), 'lineWidth', 4);
plot(dummyVariables', stdSet_SEC_eq, ':', 'color', colors(4, :), 'lineWidth', 4);
plot(dummyVariables', stdSet_SECsync_sim, 'color', 'c', 'lineWidth', 4);
plot(dummyVariables', stdSet_CMB_sim, 'color', colors(5, :), 'lineWidth', 4);
plot(dummyVariables', stdSet_CMB_eq, ':', 'color', colors(6, :), 'lineWidth', 4);
xlim([dummyVariables(1), dummyVariables(end)])

legend('ACO(sim)', 'ACO(eq)', 'SEC(sim)', 'SEC(eq)', 'SEC\_sync(sim)', 'CMB(sim)', 'CMB(eq)')
set(gca,'FontName','Times New Roman');
set(gca,'FontSize',16); 



