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
    num_test_patterns = length(test_patterns);
    spike_count_test = zeros(params.quantity_neurons, num_test_patterns);
    % for j = 1:test_patterns
    %     for i = 1:params.quantity_neurons
    %         be = T_Iapp(pattern_shift + j,1);
    %         en = be + params.impact_astro;
    %         spike_count_test(i,j) = sum(V_line(i, be:en) > params.neuron_fired_thr - 1);
    % 
    %     end
    % end
    % disp(size(spike_count_test))
    % 
    dimensions_learned_patterns = [dimensions, num_learned_patterns];
    spike_images = reshape(spike_count, dimensions_learned_patterns);
    dimensions_test_patterns = [dimensions, num_test_patterns];
    spike_images_test = reshape(spike_count_test, dimensions_test_patterns);

    mean_similarities = zeros(params.max_spikes_thr,1);
    spikes_thrs = (1:params.max_spikes_thr);
    for i = 1:params.max_spikes_thr
        spike_images_thr = spike_images > spikes_thrs(i);
        spike_images_test = spike_images_test > spikes_thrs(i);
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
    %ACROSS TEST
    performance.spike_images_test_best_thr = spike_images_test > performance.best_thr;

    pattern_similarity = ...
        compute_images_similarity_test(images, ...
            performance.spike_images_best_thr, ...
            learned_patterns, test_patterns, dimensions);
    pattern_similarity = [test_patterns;pattern_similarity];
    performance.similarities = transpose([[0,learned_patterns];transpose(pattern_similarity)]);
    disp(performance.similarities)
end

function similarity = compute_image_similarity(true_image, estimated_image, dimensions)
    otherdims = repmat({':'},1, length(dimensions));

    % Split back and foreground images by central number
    pattern_mask = true_image(otherdims{:}) ;
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
    % retrieve similarity scores per test
    for k = 1:length(test_patterns)
        estimated_image = spike_images(otherdims{:}, k);
        true_image = images{test_patterns(k)};   
        if length(dimensions) == 2
            true_image = true_image <127;
        end
        similarity(k) = compute_image_similarity(true_image, estimated_image, dimensions);
    end
end

function similarity = compute_images_similarity_test(images, spike_images,learned_patterns, test_patterns, dimensions)
    otherdims = repmat({':'},1, length(dimensions));
    similarity = zeros(length(learned_patterns),length(test_patterns));
    % retrieve similarity scores per test
    for k = 1:length(learned_patterns)
        estimated_image = spike_images(otherdims{:}, k);
        for l = 1:length(test_patterns)
            if length(dimensions) == 2
               true_image = images{test_patterns(l)}<127;    
            else
               true_image = images{test_patterns(l)};    
            end
            image_similarity = compute_image_similarity(true_image, estimated_image, dimensions);
            similarity(k, l) = image_similarity(1);
        end
    end
end