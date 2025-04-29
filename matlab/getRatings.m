function [rating_mats] = getRatings(data, rating_type, colour_positions)
%getRatings
%	Extract rating matrices for each participant
%
%	Inputs:
%		data = cell array, each cell holds data table for one participant
%		rating_type = string; 'similarity' or 'preference'
%		colour_positions = dictionary; from getColours()
%	Outputs:
%		rating_mats = CxCxP matrix; C = number of colours; P = number of
%			participants

% Identify trials corresponding to the desired response type
relevant_trials = cell(size(data));
for p = 1 : length(data)
	relevant_trials{p} = data{p}.response_type;
	relevant_trials{p} = cellfun(@(x) strcmp(x, rating_type), relevant_trials{p});
	relevant_trials{p} = relevant_trials{p} & ~data{p}.practice_trial; % ignore practice trials
end

% Get ratings
% colours x colours x participants
rating_mats = nan(numEntries(colour_positions), numEntries(colour_positions), length(data));
rating_mats_count = zeros(size(rating_mats)); % count number of ratings for each colour pair
for p = 1 : length(data)
	
	for trial = 1 : size(data{p}, 1)
		rating = data{p}.response(trial);
		stimA = string(data{p}.colour1{trial});
		stimB = string(data{p}.colour2{trial});
		
		if relevant_trials{p}(trial) == 1
			if isnan(rating_mats(colour_positions(stimA), colour_positions(stimB), p))
				rating_mats(colour_positions(stimA), colour_positions(stimB), p) = rating;
			else
				rating_mats(colour_positions(stimA), colour_positions(stimB), p) =...
					rating_mats(colour_positions(stimA), colour_positions(stimB), p) + rating;
			end
			rating_mats_count(colour_positions(stimA), colour_positions(stimB), p) =...
				rating_mats_count(colour_positions(stimA), colour_positions(stimB), p) + 1;
		end
		
	end
	
end

% Divide rating sums by number of ratings
rating_mats = rating_mats ./ rating_mats_count;

end
