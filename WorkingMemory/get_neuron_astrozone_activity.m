function [neuron_astrozone_activity, neuron_astrozone_spikes] = ...
        get_neuron_astrozone_activity(G, Mask_line, dimensions)
    params = model_parameters();
    
    mask1 = reshape(Mask_line, dimensions);
    mask1 = single(mask1);
    glutamate_above_thr = reshape(G, dimensions); % removed the threshold 0.7

    neuron_astrozone_activity = zeros(params.mastro, params.nastro, 'int8');
    neuron_astrozone_spikes = zeros(params.mastro, params.nastro, 'int8');

    if length(dimensions) == 2
        % neuron_astrozone_activity = conv2(glutamate_above_thr, kernel, 'same');
        % neuron_astrozone_spikes  = conv2(mask1, kernel, 'same');
        sj = 0;
        for j = 1 : params.az : (dimensions(1) - params.az)
            sk = 0;
            for k = 1 : params.az : (dimensions(2) - params.az)
                neuron_astrozone_activity(j - sj, k - sk) = ...
                    sum(glutamate_above_thr(j : j + params.az, k : k + params.az), 'all');
                neuron_astrozone_spikes(j - sj, k - sk) = ...
                    sum(mask1(j : j + params.az, k : k + params.az), 'all');
                sk = sk + 2;
            end
            sj = sj + 2;
        end
    else
        sj = 0;
        for j = 1 : params.az : (dimensions(1) - params.az)
            sk = 0;
            for k = 1 : params.az : (dimensions(2) - params.az)
                % Sum over both spatial dimensions and color channels
                for l = 1: dimensions(3)
                    neuron_astrozone_activity(j - sj, k - sk, l) = ...
                        sum(glutamate_above_thr(j : j + params.az, k : k + params.az, l), 'all');
                    neuron_astrozone_spikes(j - sj, k - sk, l) = ...
                        sum(mask1(j : j + params.az, k : k + params.az, l), 'all');
                end
                sk = sk + 2;
            end
            sj = sj + 2;
        end
        % neuron_astrozone_activity = convn(glutamate_above_thr, kernel, 'same');
        % neuron_astrozone_spikes  = convn(mask1, kernel, 'same');
        % neuron_astrozone_activity = neuron_astrozone_activity(1:params.az:end, 1:params.az:end);
        % neuron_astrozone_activity = neuron_astrozone_activity(1:params.mastro, 1:params.nastro);
        % neuron_astrozone_spikes = neuron_astrozone_spikes(1:params.az:end, 1:params.az:end);
        % neuron_astrozone_spikes = neuron_astrozone_spikes(1:params.mastro, 1:params.nastro);

    end
    
end