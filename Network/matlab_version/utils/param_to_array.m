function [param_array] = param_to_array(params)
    param_array = zeros(1, 150);
    param_array(1:30) = params(1);
    param_array(31:120) = params(2);
    param_array(121:150) = params(3);
end