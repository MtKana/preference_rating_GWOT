
%%

source_sim = 'OT_similarity_groups.mat';
source_pref = 'OT_preference_groups.mat';

ot = struct();
ot.sim = struct();
ot.pref = struct();
rtypes = fieldnames(ot);

% Note order of stimuli
% distance matrix is formed from (freq x int x cond)
%	So grouping is frequencies within intensities (within condition)

ot.sim = load(source_sim);
ot.pref = load(source_pref);

%% Plot GWD as function of epsilon (image)

figure;
set(gcf, 'Color', 'w');

subplot(1, 2, 1);
imagesc(ot.sim.OT.dist);
xlabel('eps');
ylabel('grouping');
title('sim');

subplot(1, 2, 2);
imagesc(ot.pref.OT.dist);
xlabel('eps');
ylabel('grouping');
title('pref');

%% Plot GWD as function of epsilon (lines)

figure;
set(gcf, 'Color', 'w');

subplot(1, 2, 1);
plot(ot.sim.epsilons, ot.sim.OT.dist);
xscale('log');
xlabel('eps');
ylabel('grouping');
title('sim');

subplot(1, 2, 2);
plot(ot.pref.epsilons, ot.pref.OT.dist);
xscale('log');
xlabel('eps');
ylabel('grouping');
title('pref');

%% Get results for epsilon with smallest distance

for r = 1 : numel(rtypes)
	rtype = rtypes{r};
	
	[dist, e_min] = min(ot.(rtype).OT.dist, [], 2);
	
	dims = size(ot.(rtype).OT.plan);
	plan = nan(dims(1:3));
	for pair = 1 : dims(3)
		plan(:, :, pair) = ot.(rtype).OT.plan(:, :, pair, e_min(pair));
	end
	
	ot.(rtype).dist = dist;
	ot.(rtype).e_min = e_min;
	ot.(rtype).plan = plan;
	
end

%% Binarise plans

for r = 1 : numel(rtypes)
	rtype = rtypes{r};
	
	plan = ot.(rtype).plan;
	plan_binary = nan(size(plan));
	
	for pair = 1 : size(plan, 3)
		
		if ~all(isnan(plan(:, :, pair)), 'all')
			plan_binary(:, :, pair) = binarise_plan(plan(:, :, pair), 0);
		else
			plan_binary(:, :, pair) = zeros(size(plan(:, :, pair)));
		end
		
	end
	
	ot.(rtype).plan_binary = plan_binary;
	
end

%% Show example plans

rtype = 'sim';

nExamples = 80;
selection = randi(size(ot.(rtype).plan_binary, 3), nExamples, 1);

figure;
set(gcf, 'Color', 'w');

for s = 1 : nExamples
	subplot(8, 10, s);
	
	imagesc(ot.(rtype).plan_binary(:, :, selection(s)));
	
	%title(['P' num2str(ot.p_pairs(selection(s), 1)) ':P' num2str(ot.p_pairs(selection(s), 2))]);
	axis square;
end

%% Show average transport plans

addpath('../');
[colour_hexes, colour_positions, colours_rgb] = getColours();

figure;
set(gcf, 'Color', 'w');
colormap inferno;

for r = 1 : numel(rtypes)
	rtype = rtypes{r};
	
	ax = subplot(1, numel(rtypes), r);
	imagesc(sum(ot.(rtype).plan, 3));
	
	cbar = colorbar;
	ylabel(cbar, 'weight');
	
	title(rtype);
	
	axis square;
	
	colourTickLabels(ax, cbar, colours_rgb, 0);
	
end

%% Print figure

output_fig = 1;

figure_name = 'figures/ot_comparison_plans';

if output_fig == 1
	set(gcf, 'PaperOrientation', 'Landscape');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG
end

%% Generate "correct" transport plans
% Diagonal, plus 11 circular shifts
% Tranposed diagonal, plus 11 circular shifts

plan = ot.sim.plan;

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

for r = 1 : numel(rtypes)
	rtype = rtypes{r};
	
	plan_binary = ot.(rtype).plan_binary;
	
	pair_accuracies = zeros(size(plan, 3), size(correct_plans, 3));
	for pair = 1 : size(plan_binary, 3)

		for scheme = 1 : size(correct_plans, 3)

			% Check row by row
			for stim = 1 : size(plan_binary, 1)
				if find(plan_binary(stim, :, pair)) == find(correct_plans(stim, :, scheme))
					pair_accuracies(pair, scheme) = pair_accuracies(pair, scheme) + 1;
				end
			end

		end

	end

	pair_accuracies = pair_accuracies ./ size(plan_binary, 1);

	% Take maximum across correct schemes
	% Keep track of which flip+rotations gives the highest matching rate
	[pair_accuracies, pair_transforms] = max(pair_accuracies, [], 2);
	
	ot.(rtype).pair_accuracies = pair_accuracies;
	ot.(rtype).pair_transforms = pair_transforms;
	
end

%% Transform each plan (flip+shift to give highest matching rate)

for r = 1 : numel(rtypes)
	rtype = rtypes{r};
	
	plan = ot.(rtype).plan;
	plan_binary = ot.(rtype).plan_binary;
	
	plan_transformed = nan(size(plan));
	plan_binary_transformed = nan(size(plan_binary));
	for pair = 1 : size(plan_binary, 3)

		flip = plan_flip(ot.(rtype).pair_transforms(pair));
		shift = -plan_shift(ot.(rtype).pair_transforms(pair)); % We want to do the inverse

		if flip == 1
			plan_transformed(:, :, pair) = rot90(circshift(plan(:, :, pair), shift), -1);
			plan_binary_transformed(:, :, pair) = rot90(circshift(plan_binary(:, :, pair), shift), -1);
		else
			plan_transformed(:, :, pair) = circshift(plan(:, :, pair), shift);
			plan_binary_transformed(:, :, pair) = circshift(plan_binary(:, :, pair), shift);
		end

	end
	
	ot.(rtype).plan_transformed = plan_transformed;
	ot.(rtype).plan_binary_transformed = plan_binary_transformed;
	
end

%% Show plans (flipped+shifted to give highest matching rate)

rtype = 'pref';

figure;
set(gcf, 'Color', 'w');

for s = 1 : nExamples
	subplot(8, 10, s);
	
	imagesc(ot.(rtype).plan_binary_transformed(:, :, selection(s)));
	
	axis square;
end

%%

figure;
set(gcf, 'Color', 'w');
colormap inferno;

for r = 1 : numel(rtypes)
	rtype = rtypes{r};
	
	ax = subplot(1, numel(rtypes), r);
	imagesc(sum(ot.(rtype).plan_transformed, 3));
	
	cbar = colorbar;
	ylabel(cbar, 'weight');
	
	title(rtype);
	
	axis square;
	
	% Note colour tick labels don't make sense because of the flips+shifts
	%	Each flip/shift will change the order the colours (reverse/shift)
	%colourTickLabels(ax, cbar, colours_rgb, 0);
	
	xlabel('right colour');
	ylabel('left colour');
	
end

%% Compare alignment distances, matching rates

figure;

subplot(1, 2, 1);
%violinplot([ot.sim.dist ot.pref.dist]);
boxplot([ot.sim.dist ot.pref.dist], 'Notch', 'on', 'Whisker', 10);
hold on;
daviolinplot([ot.sim.dist ot.pref.dist],...
	'smoothing', 0.05,...
	'box', 0,...
	'scatter', 0, 'scatteralpha', 0.2, 'jitter', 1, 'jitterspacing', 0.1,...
	'outliers', 0);
set(gca, 'XTickLabel', {'sim', 'pref'});
ylabel('alignment distance');

subplot(1, 2, 2);
%violinplot([ot.sim.pair_accuracies ot.pref.pair_accuracies]);
boxplot([ot.sim.pair_accuracies ot.pref.pair_accuracies], 'Whisker', 10);
hold on;
daviolinplot([ot.sim.pair_accuracies ot.pref.pair_accuracies],...
	'violinmax', 1, 'violinmin', 0, 'smoothing', 0.05,...
	'box', 0,...
	'scatter', 0, 'scatteralpha', 0.2, 'jitter', 1, 'jitterspacing', 0.1,...
	'outliers', 0);
line([0 3], [1/12 1/12]);
set(gca, 'XTickLabel', {'sim', 'pref'});
ylabel('matching rate');

%%

figure;
set(gcf, 'Color', 'w');

subplot(1, 2, 1);
daviolinplot([ot.sim.dist ot.pref.dist],...
	'smoothing', [], 'violinmin', 0,...
	'box', 1,...
	'scatter', 2, 'scatteralpha', 0.2, 'jitter', 1, 'jitterspacing', 0.1,...
	'outliers', 0);
set(gca, 'XTickLabel', {'sim', 'pref'});
ylabel('alignment distance');

subplot(1, 2, 2)
h = daviolinplot([ot.sim.pair_accuracies ot.pref.pair_accuracies],...
	'violinmax', 1, 'violinmin', 0, 'smoothing', 0.05,...
	'box', 1,...
	'scatter', 2, 'scatteralpha', 0.2, 'jitter', 1, 'jitterspacing', 0.1,...
	'outliers', 0);
% Add jitter to otherwise discrete looking matching rates (steps of 1/12)
yjitter = 0.02 * 2*(rand(size(ot.sim.pair_accuracies)) - 0.5)';
for s = 1 : numel(h.sc)
	h.sc(s).YData = h.sc(s).YData + yjitter;
end
line([0 3], [1/12 1/12], 'LineStyle', '--', 'Color', 'k');
set(gca, 'XTickLabel', {'sim', 'pref'});
ylabel('matching rate');
ylim([0 1]);

%% Print figure

output_fig = 1;

figure_name = 'figures/ot_comparison';

if output_fig == 1
	set(gcf, 'PaperOrientation', 'Landscape');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG
end

