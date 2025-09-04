function [violation_mat, b_thresholds, bmats, relevant_mat] = transitivityViolations(rating_mats)
%transitivityViolations
% transitivity: if x <= y  and y <= z, then x <= z
% Supposed to be used with ratings that are antisymmetric
%	(i.e. (a,b) and (b,a) have equal magnitude)
%
% Inputs:
%	rating_mats = stim x stim x participants matrix
% Outputs:
%	violation_mat = stim x stim x stim x participants x thresholds matrix
%		Holds 1 where transitivity was violated
%	b_thresholds = vector of thresholds used for determining preference
%	bmats = stim x stim x participants x thresholds matrix
%		Logical mask - zeros ratings which are within the thresholded range
%	relevant_mat = stim x stim x stim x participants x thresholds matrix
%		Holds 1 where the first two transitivity conditions are satisfied
%		Note that because/if ratings are asymmetric, (a,b) = -(b,a)
%			If (a,b) is negative, a is preferred (and so (b,a) should be
%				positive)
%		So unless (a,b)==(b,a)==0, only one of them will be relevant

% Check first two conditions, then the third condition

% Signs of the ratings (ignoring magnitudes)
rating_signs = sign(rating_mats);

% Set of possible thresholds
b_thresholds = unique(abs(rating_mats(:)));

% Set of colour indexes
colours = (1:size(rating_mats, 1));

% x * y * z * participants * thresholds
violation_mat = zeros(numel(colours), numel(colours), numel(colours), size(rating_mats, 3), numel(b_thresholds));
relevant_mat = zeros(size(violation_mat)); % for counting where the first two conditions are true

bmats = nan([size(rating_mats, 1) size(rating_mats, 2) size(rating_mats, 3) numel(b_thresholds)]);
for b = 1 : numel(b_thresholds)
	b_thresh = b_thresholds(b);
	
	% "Clear preference" mask
	% If rating is within range (-thresh, thresh), then consider it as no
	% preference
	bmats(:, :, :, b) = (rating_mats <= -b_thresh) | (rating_mats >= b_thresh);
	
	% zero-out ratings which were within the threshold range
	rating_thresholded = bmats(:, :, :, b) .* rating_signs;
	
	for p = 1 : size(rating_mats, 3)
		
		% For each colour pair, select a third colour
		% Note that because/if ratings are asymmetric, (a,b) = -(b,a)
		%	So at most only one of (a,b) and (b,a) will be relevant
		for x = colours
			for y = colours(colours~=x)
				
				for z = colours(colours~=x & colours~=y)
					
					% If (x,y) is negative, x is preferred
					%	and x <= y
					% If (x,y) is positive, y is preferred (y<x)
					%	and x !<= y
					% If (x,y) is 0, neither were preferred (x=y)
					%	and x <= y
					
					if rating_thresholded(x, y, p) <= 0 && rating_thresholded(y, z, p) <= 0
						% Then the first two conditions are satisfied
						
						relevant_mat(x, y, z, p, b) = 1;
						
						if rating_thresholded(x, z, p) > 0
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

%{
% Check first two conditions, then the third condition
% Consider preference to the left colour if rating is below some threshold

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
%}