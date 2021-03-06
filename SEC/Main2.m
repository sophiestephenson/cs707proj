function main2 = Main2(groundfile, fire_file, out_file, sec_out_file)

clc
close all
if exist('sec_out_file','var') == 1
    %then we run regular SEC
    %todo: channel output to sec_out_file
    %main
end
%% Data input
ground_truth = csvread (groundfile);
num_camera = size(ground_truth,1);
num_frame = length (ground_truth);

p_sec = csvread (fire_file);
%% Parameters
trialN = 5;                         % number of trials
d = ground_truth;                   % 1st Row - Camera 1 ; 2nd Row - Camera 2


% Lighting conditions
r_a = 1.0;                          % relative ambient light strength: r_a = e_a/e_s
r_i = 1.0;                          % relative interfering signal strength: r_i = e_i/e_s
N = num_camera;                              % number of interfering cameras


% ToF parameters
%f_mod = 20e6;                       % modulation frequency(Hz)
f_mod = 1e6;                        % allows for longer d_max
T = 1/30;                          % total integration time(s)


% Number of generated electrons
e_s = 1e7;                          % average number of signal photons
e_a = r_a*e_s;                      % average number of ambient photons
e_i = r_i*e_s;                      % average number of interfering photons


% Misc
c = 3e8;                            % Light speed(m/s)
d_max = c/(2*f_mod);                % Maximum measurable distance(m)


% Parameters for our approaches
A = 7;                              % Peak power amplification
M = 1;                            % Number of slots


% Parameters for PN
stageN = 7;                         % Number of LFSR stages
bitN = 2^stageN - 1;                % Number of bits
sampleNperBit = 1000;               % Number of samples per bit


% Photon energy
h = 6.62607015e-34;                 % Planck constant
lambda = 900e-9;                    % source wavelength
E_photon = h*c/lambda;              % unit photon energy



%% Repeat depth estimation and get depth standard deviations

% loop over dummy variables
dSet_SEC_cam = zeros(N,num_frame);
for N = 1:num_camera
    previous_d = 0;
    for j = 1:num_frame
        
        dSet_SEC = 0;

        for k = 1:trialN
            % adjust parameters for fair comparisons
            %T_SEC = T/(A*p_SEC);                    %p_SEC cannot be 0
            % SEC
            if p_sec(N, j) == 1
                if d(N, j) == 0
                    d_hat = 0;
                else
                    d_hat = estimateDepth_SEC_AI(d(N,j), c, p_sec(N,j),N, M, A, e_s, e_a, e_i, f_mod, T,p_sec);
                end
                previous_d = d_hat;
            else
                d_hat = previous_d;
            end
            dSet_SEC = d_hat + dSet_SEC;   %Estimated Depth for Camera 1
        end
        dSet_SEC_cam(N,j) = dSet_SEC/trialN;

    end
    
end

%don't need this anymore. handled by the if statement above
%for j=1:N
%    for i=2:length(dSet_SEC_cam(j,:))
%        if dSet_SEC_cam(j,i)==0
%            dSet_SEC_cam(j,i)=dSet_SEC_cam(j,i-1);
%        end
%    end
%end

%deltaD = dSet_SEC_cam - ground_truth;   %please check if this gives you correct dimension and values
csvwrite(out_file,dSet_SEC_cam);
disp("sim wrote to " + out_file)







