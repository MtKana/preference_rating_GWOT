function [asym_mats] = antisymmetrise(rating_mats)
%antisymmetrise
% Antisymmetrise (skew-symmetrise) a matrix by modifying values
%
% Inputs:
%	rating_mats = stim x stim x participants matrix
% Outputs:
%	asym_mats = antisymmetrised matrix, same size as rating_mats

asym_mats = nan(size(rating_mats));

for p = 1 : size(rating_mats, 3)
	
	% Go through each a,b pair and compare to b,a
	for a = 1 : size(rating_mats, 1)
		for b = a : size(rating_mats, 1)
			
			ab = rating_mats(a, b, p);
			ba = rating_mats(b, a, p);
			
			if sign(ab) ~= sign(ba)
				% Opposite signs - take average of absolute values
				
				ab_mean = mean(abs([ab ba]));
				
				ab_new = ab_mean;
				ba_new = ab_mean;
				
				% If either ab or ba were 0, assign the opposite sign
				%	based on the other one
				if ab == 0
					ab_new = ab_new * (-1*sign(ba));
					ba_new = ba_new * sign(ba);
				elseif ba == 0
					ab_new = ab_new * sign(ab);
					ba_new = ba_new * (-1*sign(ab));
				else % neither were 0
					% Give their original signs
					ab_new = ab_new * sign(ab);
					ba_new = ba_new * sign(ba);
				end
				
			else % sign(ab) == sign(ba)
				% Same sign
				% i) take the mean
				% ii) subtract the mean from the original
				%	So, if ab==ba, it becomes 0
				%		This applies to the diagonal of the matrix
				%	If ab ~= ba
				%		e.g. pref(a,b) = 3, pref(b,a) = 1
				%			apref(a,b) = 3-2 = 1, apref(b,a) = 1-2 = -1
				%		e.g. pref(a,b)=-3, pref(b,a)=-1
				%			apref(a,b) = -3--2 = -1, apref(b,a) = -1--2 = 1
				
				ab_mean = mean([ab ba]);
				
				ab_new = ab - ab_mean;
				ba_new = ba - ab_mean;
			
			end
			
			asym_mats(a,b,p) = ab_new;
			asym_mats(b,a,p) = ba_new;
	end
	
end

end

