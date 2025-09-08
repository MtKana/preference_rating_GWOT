function [distances] = rating2dist(ratings, rating_type, process_type)
%rating2dist
%
% Inputs:
%	ratings = matrix of participant ratings (0-7)
%		colour x colour x participant
%	rating_type = string; 'similarity' or 'preference'
%	process_type = string; for rating_type 'preference'
%		'remap': remap values to new scale (with equal intervals)
%		'antisym': 'remap' then antisymmetrise
%		'distance': 'remap' then 'antisym' then absolute
%		'distOld': 'remap' and then convert to distance (old method)
%
% Outputs:
%	distances = matrix, same dimensions as ratings

% Assumes raw ratings go from 0 to 7
max_rating_raw = 7;
max_pref_dist = max_rating_raw/2;

switch rating_type
	
	case 'similarity'
		
		distances = max_rating_raw - ratings;
		
		% Set diagonal to 0 (for distance)
		tmp = logical(repmat(eye(size(ratings, 1)), [1 1 size(ratings, 3)]));
		distances(tmp) = 0;
		
		% Symmetrise by averaging top right and bot left triangles together
		distances = (distances + permute(distances, [2 1 3])) / 2;
		
	case 'preference'
		
		% remap 0 to 7 -> -3.5 to 3.5
		distances = ratings - max_pref_dist;
		
		switch process_type
			
			case 'antisym'
				
				distances = antisymmetrise(distances);
				
			case 'distance'
				
				distances = abs(antisymmetrise(distances));
				
			case 'distOld'
				
				% Flip the sign of the upper (or lower) triangle
				upper = find(triu(ones(size(ratings, 1)), 1));
				for p = 1 : size(distances, 3)
					tmp = distances(:, :, p);
					tmp(upper) = tmp(upper) * -1;
					distances(:, :, p) = tmp;
				end
				
				% Average the upper and lower triangles
				for p = 1 : size(distances, 3)
					tmp = distances(:, :, p);
					tmp = (tmp + tmp') ./2;
					distances(:, :, p) = tmp;
				end
				
				% Take the absolute value
				distances = abs(distances);
			
		end
end

end

