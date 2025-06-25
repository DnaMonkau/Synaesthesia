function [params] = model_parameters(need_set)
persistent params_p
if nargin < 1 || ~need_set
    params = params_p;
    return;
end
params = struct;

%% Timeline
params.t_end = 5.5;
params.step = 0.0001;
params.n = fix(params.t_end / params.step);

%% Experiment
params.learn_start_time = 0.5;
params.learn_impulse_duration = 0.2;
params.learn_impulse_shift = 0.1; %0.3
params.learn_order =  [22, 2, 31,0 ] + 1;

params.test_start_time = 2.3;
params.test_impulse_duration = 0.15;
params.test_impulse_shift = 0.25; %0.4
params.test_order = [22, 3, 2, 16 31, 4, 0, 25]+ 1;


%% Applied pattern current 
params.variance_learn = 0.05;
params.variance_test = 0.2;
params.Iapp_learn = 80; 
params.Iapp_test = 8; 

%% Movie
params.after_sample_frames = 200;
params.before_sample_frames = 1;

%% Poisson noise
params.poisson_nu = 1.5;
params.poisson_n_impulses = 15;
params.poisson_impulse_duration = fix(0.03 / params.step);
params.poisson_impulse_initphase = fix(1.5 / params.step);
params.poisson_amplitude = 20;

%% Runge-Kutta steps
params.u2 = params.step / 2;
params.u6 = params.step / 6;

%% Network size
params.mneuro = 79; % Neural network axis size
params.nneuro = 79;
params.quantity_neurons = params.mneuro * params.nneuro;
params.mastro = 26; % Astrocyte network axis size
params.nastro = 26;
az = 4; % Astrosyte zone size
params.az = az - 1;

%% Initial conditions
params.v_0 = -70; % Initial membrane potential
params.ca_0 = 0.072495; % Initial calcium rate
params.h_0 = 0.886314; % Initial fraction of actovated IP3 receptors
params.ip3_0 = 0.820204; % Initial rate of IP3 production through glutamate divided by duration

%% Neuron mode
params.aa = 0.1; %FS timescale of recovery variable
params.b = 0.2; % Recovery variable sensitivity
params.c = -65; % After spike reset value membrane potential
params.d = 2; % After spike reset value recovery variable
params.alf = 10; % Glutmate clearance constant
params.k = 600; % Efficacy of glutamate relese
params.neuron_fired_thr = 30;
params.I_input_thr = 25; % Current input threshold

%% Synaptic connections
params.N_connections = 40; % maximum number of connections between neurons
params.quantity_connections = params.quantity_neurons * params.N_connections;
params.lambda = 5; % average exponential distribution
params.beta = 5;  
params.gsyn = 0.025; % synaptic weight eta+ v_ca
params.aep = 0.5; % astrocyte effect parameter
params.Esyn = 0; % Synaptic reversal potential for excitatory synapses
params.ksyn = 0.2; % Slope of synaptic activation function

%% Astrocyte model
params.dCa = 0.05;
params.dIP3 = 0.1; % 0.05
params.enter_astro = 6; % Astrocyte threshold for spiking
params.min_neurons_activity = 8;
params.t_neuro = 0.06; % Duration of neuro-induced modulation of synapse t(sec)
params.amplitude_neuro = 5; % Amplitude of neuro-induced modulation of synapse
params.threshold_Ca = 0.15; % Threshold concentration of Ca2+ astro-modulation synaptic transmission
window_astro_watch = 0.01; % t(sec)
shift_window_astro_watch = 0.001; % t(sec)
impact_astro = 0.25; % Duration of neuro-induced modulation of synapse t(sec)
params.impact_astro = fix(impact_astro / params.step);
params.window_astro_watch = fix(window_astro_watch / params.step);
params.shift_window_astro_watch = fix(shift_window_astro_watch / params.step);

%% Memory performance
params.max_spikes_thr = 30;

%% 
params_p = params;
end
