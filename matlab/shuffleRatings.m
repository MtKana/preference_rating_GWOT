function [shuffled] = shuffleRatings(ratings)
%shuffleRatings
%
% Shuffle rows and columns for each participant
%
% Inputs:
%	ratings = C x C x N matrix; C is number of colours, N is number of
%		participants
% Outputs:
%	shuffled = C x C x N matrix; rows and columns shuffled per participant

shuffled = nan(size(ratings));

for p = 1 : size(ratings, 3)
	
	row_order = randperm(size(ratings, 1));
	col_order = randperm(size(ratings, 2));
	
	shuffled(:, :, p) = ratings(row_order, col_order, p);
	
end

end

