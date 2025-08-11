function mat2csv(data, filename, column_names)
% Save N-D matrix as a CSV
%	Convert to table first (with each dimension corresponding to 1 column)
%	Then save table to csv
%
% Inputs:
%	data = ND matrix
%	filename = string; output filename
%	column_names = cell array; containing N+1 field names, last is the
%		field name for the values themselves
%
    % Validate input
    num_dims = ndims(data);
    if length(column_names) ~= num_dims + 1
        error('column_names must have %d elements: one for each dimension plus one for the value.', num_dims + 1);
    end

    % Get matrix size and number of elements
    dims = size(data);
    num_elements = numel(data);
    
    % Get subscripts
    subs = cell(1, num_dims);
    [subs{:}] = ind2sub(dims, 1:num_elements);
    
    % Create table
    T = table;
    for d = 1:num_dims
        T.(column_names{d}) = subs{d}';
    end
    
    % Add the values
    T.(column_names{end}) = data(:);
    
    % Write to CSV
    writetable(T, filename);
end
