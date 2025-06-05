function [pRatings,cOrders] = colour2pref(cRatings)
%COLOUR2PREF Summary of this function goes here
%
% Inputs:
%	cRatings = matrix of participant preference ratings (0-7 or -3.5 to 3.5)
%		colour x colour x participant
%		Stimuli will be sorted based on preference to the left colour
%			i.e. low preference values, which correspond to preferring the
%			the left stimulus, will be ranked high 
%
% Outputs:
%	pRatings = matrix as cRatings, but
%		rows/columns correspond to overall preference - not colour label
%		This means - row/col 1 will not necessarily mean the same colour
%			for all participants - it just means the most preferred
%			stimulus for the participant
%	cOrders = matrix (colours x participants)
%		first dimension gives the corresponding colour index for each
%		overall preference

overall_prefs = mean(cRatings, 2);

[sorted_prefs, sorted_order] = sort(overall_prefs, 1, 'ascend');

cOrders = permute(sorted_order, [1 3 2]);

pRatings = nan(size(cRatings));

for participant = 1 : size(cRatings, 3)
	
	pRatings(:, :, participant) = cRatings(cOrders(:, participant), cOrders(:, participant), participant);
	
end

end

