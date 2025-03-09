function [I_signals, full_timeline, timeline_signal_id, ...
          timeline_signal_id_movie] = make_experiment(images, dimensions)
    params = model_parameters();

    num_images = length(images);
    if isfield(params, 'learn_order')
        learn_order = params.learn_order;
    else
        learn_order = make_image_order(num_images, 10, true); 
    end
    
    learn_signals = make_noise_signals(images, learn_order, ...
            dimensions, ...
            params.variance_learn, params.Iapp_learn);
    
    if isfield(params, 'test_order')
        test_order = params.test_order;
    else
        test_order = make_image_order(num_images, 1, true);
    end
   
    test_signals = make_noise_signals(images, test_order, ...
            dimensions, ...
            params.variance_test, params.Iapp_test);
   
    I_signals = cat(length(dimensions)+1, learn_signals, test_signals);
    I_signals = uint8(I_signals);

    learn_timeline = make_timeline(params.learn_start_time, ...
        params.learn_impulse_duration, params.learn_impulse_shift, ...
        length(learn_order));
    test_timeline = make_timeline(params.test_start_time, ...
        params.test_impulse_duration, params.test_impulse_shift, ...
        length(test_order));

    full_timeline = [learn_timeline; test_timeline];
    full_timeline = fix(full_timeline ./ params.step);
    %full_timeline = uint16(full_timeline);
	%full_timeline = typecast(full_timeline, 'uint16');
disp(full_timeline)
    timeline_signal_id = zeros(1, params.n, 'uint8');
    timeline_signal_id_movie = zeros(1, params.n, 'uint8');
    %disp(full_timeline);
    % Iterate over the full number of samples (learn+test)
    for i = 1 : size(I_signals, length(dimensions)+1)
        be = full_timeline(i, 1);
        en = full_timeline(i, 2);
        timeline_signal_id(be : en) = i;
        
        be = be - params.before_sample_frames;
        en = en + params.after_sample_frames;
        
        timeline_signal_id_movie(be : en) = i;
    end
    
end

function [image] = make_noise_signal(image, dimensions, variance, Iapp0, thr)
    if nargin < 6
       
        thr = 127; % important  for signal transfer og =127 for < 127
    end
    % test shuffle only image dimensions
    if length(dimensions) == 3
        % image = image < thr; % check max fow bw vs for color
        img = image;
        p = randperm(prod(dimensions(1:2)));
        b = p(1 : uint16(prod(dimensions(1:2)) * variance));
        img(b) = ~img(b);
        image = img;
        
    elseif length(dimensions) == 2
        image = image < thr;
        p = randperm(prod(dimensions));
        b = p(1 : uint16(prod(dimensions) * variance));
        image(b) = ~image(b);
    end
    % shifts pixels around and binary flips images
    
    image = double(image) .* Iapp0; 
end

function [image_order] = make_image_order(num_images, num_repetitions, need_shuffle)
    image_order = [];
    for id_image = 1:num_images
        for j = 1:num_repetitions
            image_order(end + 1) = id_image;
        end
    end
    if need_shuffle
        image_order = image_order(randperm(length(image_order)));
    end
end

function [signals] = make_noise_signals(images, order, dimensions, variance, Iapp0)
    signal_dimensions = [dimensions, length(order)];
    signals = zeros(signal_dimensions, 'double');
    otherdims = repmat({':'},1, length(dimensions));

    for i = 1:length(order)
        image_id = order(i);
        
        [signal] = make_noise_signal(images{image_id}, ...
            dimensions, variance, Iapp0);
        signals(otherdims{:}, i) = signal;
    end
end

function [timeline] = make_timeline(start, duration, step, num_samples)
    timeline = zeros(num_samples, 2, 'double');
    for i = 1:num_samples
        be = start + step * (i - 1);
        en = be + duration;
        timeline(i, :) = [be, en];
    end
end
