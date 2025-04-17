function [performance] = compute_memory_performance(images, V_line, T_Iapp, dimensions)
    params = model_parameters();
    params.quantity_neurons = prod(dimensions);
    num_learn_patterns = length(params.learn_order);
    pattern_shift = num_learn_patterns;
    
    learned_patterns = unique(params.learn_order);
    num_learned_patterns = length(learned_patterns);
    
    test_patterns = params.test_order;
    spike_count = zeros(params.quantity_neurons, num_learned_patterns);
    for j = 1:num_learned_patterns
        pattern_id = find(test_patterns == learned_patterns(j));
        for i = 1:params.quantity_neurons
            be = T_Iapp(pattern_shift + pattern_id,1);
            en = be + params.impact_astro;
            spike_count(i,j) = sum(V_line(i, be:en) > params.neuron_fired_thr - 1);
             
        end
    end
    

    dimensions_learned_patterns = [dimensions, num_learned_patterns];
    spike_images = reshape(spike_count, dimensions_learned_patterns);
    mean_similarities = zeros(params.max_spikes_thr,1);
    spikes_thrs = (1:params.max_spikes_thr);
    for i = 1:params.max_spikes_thr
        spike_images_thr = spike_images > spikes_thrs(i);
        pattern_similarity = ...
            compute_images_similarity(images, spike_images_thr, learned_patterns, dimensions);
        mean_similarities(i) = mean(pattern_similarity);
    end
    
    performance.spike_images = spike_images;
    [performance.mean_performance, id_best_thr] = max(mean_similarities);
    performance.best_thr = spikes_thrs(id_best_thr);
    performance.spike_images_best_thr = spike_images > performance.best_thr;
    performance.best_thr_freq = performance.best_thr / (params.impact_astro * params.step);
    performance.freq_images = spike_images / (params.impact_astro * params.step);
    
    performance.learned_pattern_similarities = ...
        compute_images_similarity(images, ...
            performance.spike_images_best_thr, ...
            learned_patterns, dimensions);
end

function similarity = compute_image_similarity(true_image, estimated_image, dimensions)
    otherdims = repmat({':'},1, length(dimensions));
    pattern_mask = true_image(otherdims{:});
    % background_mask = true_image(otherdims{:}) >=127;
    
    % n_pattern = sum(pattern_mask, 'all');
    % n_background = sum(background_mask, 'all');
    n_true_pattern = sum(pattern_mask == estimated_image, 'all');
    % n_true_background = sum(sum(background_mask == estimated_image, 'all');
    % similarity = (n_true_background / n_background + n_true_pattern / n_pattern) / 2;
    similarity = (n_true_pattern) / prod(dimensions);
end

function similarity = compute_images_similarity(images, spike_images, test_patterns, dimensions)
    otherdims = repmat({':'},1, length(dimensions));
    similarity = zeros(length(test_patterns), 1);
    for k = 1:length(test_patterns)
        estimated_image = spike_images(otherdims{:}, k);
        % disp(sum(images{test_patterns(k)}>0));
        if (length(dimensions) == 2)
            true_image = images{test_patterns(k)}; % threshold for images  color needs lower threshold and black background
        else
            true_image = images{test_patterns(k)};
        end
        similarity(k) = compute_image_similarity(true_image, estimated_image, dimensions);
    end
end
