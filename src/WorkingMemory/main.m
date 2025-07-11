tic;
%% Initialization
close all; clearvars;
%%
num = 200;
params = model_parameters(true);
% disp('Parameters defined');

%% multiple runs
for i = 1:num;
    model = init_model(i-1);
    disp('Model defined');
    % define amount of neurons dynamically
    params.quantity_neurons = prod(model.dimensions);
    params.quantity_connections = params.quantity_neurons * params.N_connections;
    %% Simulation 
    [model] = simulate_model(model, params); 
    
    %% Visualization of learning and testing processes
    %  Video consist of 3 frames (left to right):
    %  1. input pattern
    %  2. neuron layer
    % %  3. astrocyte layer
    % [model.video] = make_video(model.Ca_size_neuros, ...
    %     model.V_line, ...
    %     model.Iapp_v_full, ...
    %     model.T_record_met, model.dimensions);
    % 
    % show_video(model.video, struct('limits', [0, 255], 'fps', 30));
    % 
    %% Compute memory performance
    [memory_performance] = ...
        compute_memory_performance(model.images, model.V_line, model.T_Iapp, model.dimensions);
    fprintf('Mean memory performance: %0.4f\n', memory_performance.mean_performance);
    fmt = repmat(' %0.4f',1,numel(memory_performance.learned_pattern_similarities));
    fprintf(['Memory performance per image: ', fmt, '\n'], ...
        memory_performance.learned_pattern_similarities);

    txt = sprintf('results/check_performance_%.1f.mat', i);

    save(txt, model.Ca_size_neuros, ...
        model.V_line, ...
        model.Iapp_v_full, ...
        model.T_record_met, model.dimensions, memory_performance)

    %% Predicted learned images
    % show_video(memory_performance.freq_images); % by frequency
    % 
    % show_video(memory_performance.spike_images_best_thr); % with threshold
    %% Clear variables
    clear model memory_performance;

 end
 toc;
% catch ME
%     if (strcmp(ME.identifier,'MATLAB:nomem'))
%         error('Out of memory. Please, increase the amount of available memory. \nThe minimum required amount of RAM is 16 GB.', 0);
%     else
%         rethrow(ME);
%     end
%   end
