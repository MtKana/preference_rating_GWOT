%%

%%

source_file = 'OT_preference_distance.mat';

% Note order of stimuli
% distance matrix is formed from (freq x int x cond)
%	So grouping is frequencies within intensities (within condition)

ot = load(source_file);

%%

if ~isfield(ot, 'p_pairs')
	p_pairs = nchoosek((1:19), 2);
	ot.p_pairs = p_pairs;
else
	
	if min(ot.p_pairs(:)) == 0
		ot.p_pairs = ot.p_pairs + 1; % convert 0-index to 1-index
	end
end

%% Plot GWD as function of epsilon (image)

figure;
set(gcf, 'Color', 'w');

imagesc(ot.OT.dist);

%% Plot GWD as function of epsilon (lines)

figure;
set(gcf, 'Color', 'w');

plot(ot.epsilons, ot.OT.dist);
xscale('log');

%% Get results for epsilon with smallest distance

[dist, e_min] = min(ot.OT.dist, [], 2);

dims = size(ot.OT.plan);
plan = nan(dims(1:3));
for pair = 1 : dims(3)
	plan(:, :, pair) = ot.OT.plan(:, :, pair, e_min(pair));
end

%% Binarise plans

plan_binary = nan(size(plan));
for pair = 1 : size(plan, 3)
	
	if ~all(isnan(plan(:, :, pair)), 'all')
		plan_binary(:, :, pair) = binarise_plan(plan(:, :, pair), 0);
	else
		plan_binary(:, :, pair) = zeros(size(plan(:, :, pair)));
	end
	
end

%% Show example plans

nExamples = 80;
selection = randi(size(plan_binary, 3), nExamples, 1);

figure;
set(gcf, 'Color', 'w');

for s = 1 : nExamples
	subplot(8, 10, s);
	
	imagesc(plan_binary(:, :, selection(s)));
	
	title(['P' num2str(ot.p_pairs(selection(s), 1)) ':P' num2str(ot.p_pairs(selection(s), 2))]);
	axis square;
end

%% Show specific plan

pair_selection = [7 8];
pair = find(ot.p_pairs(:, 1) == pair_selection(1) & ot.p_pairs(:, 2) == pair_selection(2));

figure;
set(gcf, 'Color', 'w');

imagesc(plan_binary(:, :, pair));

%% Generate "correct" transport plans
% Diagonal, plus 11 circular shifts
% Tranposed diagonal, plus 11 circular shifts

correct_plans = nan(size(plan, 1), size(plan, 1), 24);
plan_flip = nan(24, 1);
plan_shift = nan(24, 1);

% Diagonal plan
ref_plan = diag(ones(size(plan, 1), 1));

plan_counter = 1;

% Diagonal + shifted plans
for shift = 0 : size(plan, 1) - 1
	correct_plans(:, :, plan_counter) = circshift(ref_plan, shift);
	
	plan_flip(plan_counter) = 0;
	plan_shift(plan_counter) = shift;
	
	plan_counter = plan_counter + 1;
end

% Flipped + shifted diagonal plans
for shift = 0 : size(plan, 1) - 1
	correct_plans(:, :, plan_counter) = rot90(circshift(ref_plan, shift));
	
	plan_flip(plan_counter) = 1;
	plan_shift(plan_counter) = shift;
	
	plan_counter = plan_counter + 1;
end

%% Evaluate transport plan "accuracies"

pair_accuracies = zeros(size(plan, 3), size(correct_plans, 3));
for pair = 1 : size(plan_binary, 3)
	
	for scheme = 1 : size(correct_plans, 3)
		
		% Check row by row
		for stim = 1 : size(plan, 1)
			if find(plan_binary(stim, :, pair)) == find(correct_plans(stim, :, scheme))
				pair_accuracies(pair, scheme) = pair_accuracies(pair, scheme) + 1;
			end
		end
		
	end
	
end

pair_accuracies = pair_accuracies ./ size(plan, 1);

% Take maximum across correct schemes
% Keep track of which flip+rotations gives the highest matching rate
[pair_accuracies, pair_transforms] = max(pair_accuracies, [], 2);

%% Transform each plan (flip+shift to give highest matching rate)

plan_transformed = nan(size(plan));
plan_binary_transformed = nan(size(plan_binary));
for pair = 1 : size(plan_binary, 3)
	
	flip = plan_flip(pair_transforms(pair));
	shift = -plan_shift(pair_transforms(pair)); % We want to do the inverse
	
	if flip == 1
		plan_transformed(:, :, pair) = rot90(circshift(plan(:, :, pair), shift), -1);
		plan_binary_transformed(:, :, pair) = rot90(circshift(plan_binary(:, :, pair), shift), -1);
	else
		plan_transformed(:, :, pair) = circshift(plan(:, :, pair), shift);
		plan_binary_transformed(:, :, pair) = circshift(plan_binary(:, :, pair), shift);
	end
	
end

%% Show plans (flipped+shifted to give highest matching rate)

figure;
set(gcf, 'Color', 'w');

for s = 1 : nExamples
	subplot(8, 10, s);
	
	imagesc(plan_binary_transformed(:, :, selection(s)));
	
	title(['P' num2str(ot.p_pairs(selection(s), 1)) ':P' num2str(ot.p_pairs(selection(s), 2))]);
	axis square;
end

%% Show specific plan

figure;
set(gcf, 'Color', 'w');

imagesc(plan_binary_transformed(:, :, pair));

%% Cluster participants based on alignment distances

% Convert distances to square matrix
align_dists = nan(max(ot.p_pairs));

for pair = 1 : size(ot.p_pairs, 1)
	align_dists(ot.p_pairs(pair, 1), ot.p_pairs(pair, 2)) = dist(pair);
	align_dists(ot.p_pairs(pair, 2), ot.p_pairs(pair, 1)) = dist(pair);
end

% Set diagonal to 0
for p = 1 : size(align_dists, 1)
	align_dists(p, p) = 0;
end

% Show distance matrix
figure;
set(gcf, 'Color', 'w');
imagesc(align_dists);
colorbar;
xlabel('participant');
ylabel('participant');
axis square

% Hierarchical clustering
% Create dendrogram from distances

clusterDistance_method = 'average';

distances_p = squareform(align_dists); % convert to pdist vector form

tree = linkage(align_dists, clusterDistance_method);

%  Plot dendrogram

figure;
set(gcf, 'color', 'w');
[h, T, outperm] = dendrogram(tree, size(tree, 1)+1);
xlabel('participant');
axis tight

% Plot distance matrices

figure;
set(gcf, 'color', 'w');

subplot(1, 2, 1);
imagesc(align_dists); c = colorbar;
title(c, '1-r');
set(gca, 'XTick', (1:size(align_dists, 1)));
set(gca, 'YTick', (1:size(align_dists, 1)));
xlabel('participant');
ylabel('participant');
axis square;

subplot(1, 2, 2);
imagesc(align_dists(outperm, outperm)); c = colorbar;
title(c, '1-r');
set(gca, 'XTick', (1:size(align_dists, 1)), 'XTickLabel', outperm);
set(gca, 'YTick', (1:size(align_dists, 1)), 'YTickLabel', outperm);
xlabel('participant');
ylabel('participant');
axis square;

%% Relabel colours with overall preference rank, recheck accuracy

