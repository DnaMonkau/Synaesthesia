function [Post_line,Pre] = create_connections(dimensions)
    params = model_parameters();
    dimensions_post = [dimensions, params.N_connections];
    Post = zeros(dimensions_post, 'int16');
    Post_for_one = zeros(dimensions,'int8');
    ties_stock = 1000 * params.N_connections;
    if length(dimensions) == 2
        for i = 1 : dimensions(1)
            for j = 1 : dimensions(2)
                %[samples] = fast_weighted_sampling(weights, m);
                XY = zeros(2, ties_stock, 'int8');
                R = random('exp', params.lambda, 1, ties_stock);
                fi = 2 * pi * rand(1, ties_stock);
                XY(1,:) = fix(R .* cos(fi));
                XY(2,:) = fix(R .* sin(fi));
                XY1 = unique(XY', 'row','stable');
                XY = XY1';
                n = 1;
                for k = 1 : ties_stock
                    x = i + XY(1, k);
                    y = j + XY(2, k);
                    if (i == x && j == y)
                        pp = 1;
                    else pp = 0;
                    end
                    if (x > 0 && y > 0 && x <= dimensions(1) && y <= dimensions(2) && pp == 0)
                        Post(i,j,n) = sub2ind(size(Post_for_one), x, y);
                        n = n + 1;
                    end
                    if n > params.N_connections
                        break
                    end
                end
            end
        end
    else
      for i = 1 : dimensions(1)
        for j = 1 : dimensions(2)
          for p = 1 : dimensions(3)
            %[samples] = fast_weighted_sampling(weights, m);
            
            XY = zeros(length(dimensions), ties_stock, 'int8');
            R = random('exp', params.lambda, 1, ties_stock);
            fi = 2 * pi * rand(1, ties_stock);
            XY(1,:) = fix(R .* cos(fi));
            XY(2,:) = fix(R .* sin(fi));
            XY(3,:) = fix(R .*cos(fi) .*sin(fi)); % check with this off
            XY1 = unique(XY', 'row','stable');
            XY = XY1';
            n = 1;
            for k = 1 : ties_stock
                x = i + XY(1, k);
                y = j + XY(2, k);
                z = p + XY(3, k);
                if z <= 1
                    z=1;
                end
                if (i == x && j == y && p == z)
                    pp = 1;
                else pp = 0;
                end
                % if (x > 0 && y > 0  && z > 0 && x <= dimensions(1) && y <= dimensions(2) && z <= dimensions(3) && pp == 0) 
                if (x > 0 && y > 0  && z > 0 && x <= dimensions(1) && y <= dimensions(2) && z <= dimensions(3) && pp == 0) 
                    Post(i,j,p,n) = sub2ind(size(Post_for_one), x, y, z);
                    n = n + 1;
                end
                if n > params.N_connections
                    break
                end
            end
          end
        end
      end
    end
        if length(dimensions) == 2
            Post2 = permute(Post, [3 1 2]);
        else
            Post2 = permute(Post, [4 1 2 3]);
        end
        Post_line = Post2(:)';
        Pre = zeros(1,size(Post_line, 2), 'int16');
        k = 1;
        for i = 1 : params.N_connections : size(Post_line, 2)
            Pre(i : i + params.N_connections - 1) = k;
            k = k + 1;
        end
 end

 

function [samples] = fast_weighted_sampling(weights, m)
    % std::pow(dist(gen), 1. / iter)
    q = pow(rand(length(weights), 1), 1 ./ weights);
    [~, samples] = sort(q);
    samples = samples(1:m);
end

function get_sub_weights(weights, i, j, w, h)
    weights()
end

function [weights] = make_weights(w, h)
    weights = zeros(2 * w + 1, 2 * h + 1, 'double');
    for i = -w:w
        for j = -h:h
            weight = exp(-norm([i, j]) * params.lambda) * params.lambda;
            weights(i + w + 1, j + h + 1) = weight;
        end
    end
end