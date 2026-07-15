%%

% Number of time to randomly split participants into two groups
nSplits = 100;

source_file = 'preference_distance.mat';
out_file = 'preference_groups.mat';

%% Load

source = load(source_file);

%% Generate participant groupings
% TODO: make it so the groupings are the same between sim and pref

nParticipants = size(source.rating_mats, 3);

groupings = cell(nSplits, 2);

% Randomly split participants into two groups
for split = 1 : nSplits
	participant_order = randperm(nParticipants);
	groupings{split, 1} = participant_order(1:floor(nParticipants/2)+1);
	groupings{split, 2} = participant_order(floor(nParticipants/2)+1:end);
end

%% Average ratings within group

% stim x stim x group x split
dims = size(source.rating_mats);
rating_mats = nan(dims(1), dims(1), 2, nSplits);

for split = 1 : nSplits
	
	rating_mats(:, :, 1, split) = mean(source.rating_mats(:, :, groupings{split, 1}), 3);
	rating_mats(:, :, 2, split) = mean(source.rating_mats(:, :, groupings{split, 2}), 3);
	
end

%% Save

% -v7 allows us to load into python using scipy.io.loadmat(?)

save(out_file, 'rating_mats', 'groupings', '-v7');
disp(['saved ' out_file]);