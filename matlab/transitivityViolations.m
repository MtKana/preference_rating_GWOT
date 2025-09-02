function [violation_mat, b_thresholds, valid_mat] = transitivityViolations(rating_mats)
%transitivityViolations
% transitivity: if x <= y <= z, then x <= z
%
% Inputs:
%	rating_mats = stim x stim x participants matrix
% Outputs:
%	violation_mat = stim x stim x stim x participants x thresholds matrix
%		Holds 1 where transitivity was violated
%	b_thresholds = vector of thresholds used for determining preference
%	valid_mat = stim x stim x stim x participants x thresholds matrix
%		Holds 1 where the first two transitivity conditions are satisfied

% transitivity: if x <= y <= z, then x <= z
%
% Negative ratings correspond to preferring the left colour
%
% If participants prefer a over b, and b over c,
%	then they should prefer a over c
%
% First, binarise ratings with some threshold
%	Because negative ratings correspond to preferring the left colour:
%		Make values below the threshold 1 (left colour preferred)
%		Make values above the threshold 0 (right colour preferred)
%	So, pref(a, b) = 1 means:
%		a (on the left) has a lower rating value than b (on the right)
%
% For each x and z, check x <= z
%	If false, then check for each y:
%		x <= y
%		and
%		y <= z
%	If both are true, then record a violation
%	If one or both are false, then there is no violation
%
% What to consider as "percentage violations"?
%	nViolations / A - out of every combination of x,y,z
%		A = 12*11*10 combinations
%	nViolations / B - out of every case where the first two conditions are true
%		B = count of cases where x <= y and y <= z

%{
b_thresh = 0;

% Binarise ratings matrices
bmat = rating_mats <= b_thresh;

violation_mat = zeros(size(rating_mats));

for p = 1 : size(rating_mats, 3) % for each participant
	
	for x = 1 : size(rating_mats, 1)
		for z = 1 : size(rating_mats, 2)
			
			% skip if x and y are the same (no transitivity to test for)
			if x ~= z
				
				% Check if last condition (x <= z) is violated (false)
				if bmat(x, z, p) ~= 1
					
					% Check preceding conditions for each other colour are
					% true
					for y = 1 : size(rating_mats, 1)
						if (y ~= x) && (y ~= z)
							
							% Assumes boolean matrix (not raw ratings)
							violation_mat(x, z, p) =...
								violation_mat(x, z, p) +...
								(bmat(x, y, p) & bmat(y, z, p));
							
						end
					end
					
				end
				
			end
			
		end
	end
	
end
%}

% Do it the "long" way
%	(start from the first condition instead of the last)

b_thresholds = unique(rating_mats(:));

% Set of colour indexes
colours = (1:size(rating_mats, 1));

% x*y*z*participants
violation_mat = zeros(numel(colours), numel(colours), numel(colours), size(rating_mats, 3), numel(b_thresholds));
valid_mat = zeros(size(violation_mat)); % for counting where the first two conditions are true

bmats = nan([size(rating_mats, 1) size(rating_mats, 2) size(rating_mats, 3) numel(b_thresholds)]);
for b = 1 : numel(b_thresholds)
	b_thresh = b_thresholds(b);
	
	% Binarise ratings matrices
	bmats(:, :, :, b) = rating_mats <= b_thresh;
	
	for p = 1 : size(rating_mats, 3)
		for x = colours
			for y = colours(colours~=x)
				for z = colours(colours~=x & colours~=y)
					
					if bmats(x, y, p, b) == 1 && bmats(y, z, p, b) == 1
						% Then the first two conditions are satisfied
						
						valid_mat(x, y, z, p, b) = 1;
						
						if bmats(x, z, p, b) ~= 1
							% Then there is a violation
							
							violation_mat(x, y, z, p, b) = 1;
							
						end
						
					end
					
				end
			end
		end
	end
	
end


end

