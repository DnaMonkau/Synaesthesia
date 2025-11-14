tic;
%% Initialization
close all; clearvars;
%%
num = 1;
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

    %txt = sprintf('results/extended_trivial_performance_%.1f.mat', i);

    %save(txt);
     %   "model.V_line", ...
      %  "model.Iapp_v_full", ...
       % "model.T_record_met", "model.dimensions"," memory_performance")
    %% Raster plot
    %%% Inspired by Felix Schneider
    %%% Auditory Cognition Group: https://www.auditorycognition.org/
    %%% Biosciences Institute, Newcastle University Medical School
    %%% 02/2020
    figure();
    ax = subplot(1, 1,1); hold on
    % For all trials...
    
    T = length(model.V_line);
    for iTrial = 1:T
    display(iTrial);                  
        spks            = find(model.V_line(iTrial,:)>-70);         % Get all spikes of respective trial    
        plot( spks(1,:),iTrial, '.k')
    end
    ax.YLim             = [0 T+1];
    ax.XLim             = [0 length(spks)];
    ax.YTick            = [0 :T+1];
    
    ax.XLabel.String  	= 'Time [s]';
    ax.YLabel.String  	= 'Neurons';
    savefig('images/emergent_Rasterplot.fig')
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
