function [param_array] = param_to_array(params)
% convert param to 150 length array
    param_array = zeros(1, 150);
    param_array(1:30) = params(1);      % FS
    param_array(31:120) = params(2);    % RSA
    param_array(121:150) = params(3);   % IB
end