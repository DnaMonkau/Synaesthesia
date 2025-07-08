function [Iastro_neuron_expanded, Ca_expanded] = expand_astrocytes(Ca, Iastro_neuron, dimensions)
    % Expands astrocyte calcium concentration and electrical currents 
    % to connected neurons
    params = model_parameters();
    km = 0;
    Iastro_neuron_expanded = zeros(dimensions);
    Ca_expanded = zeros(dimensions);
    if length(dimensions) == 2
        for j = 1 : params.az : (dimensions(1) - params.az)
            kmm = 0;
            for k = 1 : params.az : (dimensions(2) - params.az)
                Iastro_neuron_expanded(j : j + params.az, k : k + params.az) = ...
                    Iastro_neuron(j - km, k - kmm);
                Ca_expanded(j : j + params.az, k : k + params.az) = ...
                    Ca(j - km, k - kmm);
                kmm = kmm + 2;
            end
            km = km + 2;
        end
    elseif length(dimensions)==3
        for j = 1 : params.az : (dimensions(1) - params.az)
             kmm = 0;
            for k = 1 : params.az : (dimensions(2) - params.az)
                % Get single values
                Iastro_value = Iastro_neuron(j - km, k - kmm);
                Ca_value = Ca(j - km, k - kmm);
                for l = 1 : dimensions(3)
                    % Fill 3D block
                    Iastro_neuron_expanded(j : j + params.az, k : k + params.az, l) = Iastro_value;
                    Ca_expanded(j : j + params.az, k : k + params.az, l) = Ca_value;
                end
                kmm = kmm + 2;
            end
            km = km + 2;
        end
    end
    Iastro_neuron_expanded = Iastro_neuron_expanded(:)';
end