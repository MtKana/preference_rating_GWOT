function [violation_mat] = triangleInequalityViolations(rating_mats)
%triangleInequalityViolations
% triangle inequality: d(x,z) <= d(x,y) + d(y,z)
%
% Inputs:
%	rating_mats = stim x stim x participants matrix
% Outputs:
%	violation_mat = stim x stim x stim x participants x thresholds matrix
%		Holds 1 where inequality was violated

% Set of colour indexes
colours = (1:size(rating_mats, 1));

% x * y * z * participants
violation_mat = zeros(numel(colours), numel(colours), numel(colours), size(rating_mats, 3));

for p = 1 : size(rating_mats, 3)
	for x = colours
		for y = colours(colours~=x)
			for z = colours(colours~=x & colours~=y)
				
				xyz = rating_mats(x, y, p) + rating_mats(y, z, p);
				
				xz = rating_mats(x, z, p);
				
				if xz > xyz % if xz is not less than or equal than xyz
					violation_mat(x, y, z, p) = 1;
				end
				
			end
		end
	end
end

end