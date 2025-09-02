%% Description
%{

Check number of violations for pre-order

preorder requires:
	reflexivity
	transitivity
	
partial order requires:
	reflexivity
	transitivity
	antisymmetry

reflexivity
	x <= x
transitivity
	if x <= y <= z, then x <= z
antisymmetry
	if x <= y <= x, then x = y

https://ncatlab.org/nlab/show/preorder
https://ncatlab.org/nlab/show/partial+order

%}

%% Load data

source_dir = 'data_mat/';
source_file = 'data.mat';

loaded = load([source_dir source_file]);

%%

% 'similarity' or 'preference'
rating_type = 'preference';

% 'raw' or 'remap' or 'distance' - 'remap' only for rating_type 'preference'
process_type = 'remap';

% '' or 'preference_ordered'
object_order = '';

% 0 = no average; 1 = average ratings across participants; 2 = average within participant clusters
% Note - check hardcoded clusters in the code in the following sections
participant_mean = 0;

data = loaded.data;

[colour_hexes, colour_positions, colours_rgb] = getColours();

%% Extract rating matrices

rating_mats = getRatings(data, rating_type, colour_positions);

%% Convert to distances

switch rating_type
	case 'similarity'
		
		switch process_type
			case 'raw'
				rating_mats = rating_mats;
				relation_string = 'sim';
			case 'distance'
				rating_mats = rating2dist(rating_mats, rating_type, []);
				relation_string = 'dissim';
		end
		
		clim = [0 7];
		
	case 'preference'
		
		switch process_type
			case 'raw'
				rating_mats = rating_mats;
				clim = [0 7];
				relation_string = 'pref';
			case 'remap'
				rating_mats = rating2dist(rating_mats, rating_type, 'remap');
				clim = [-3.5 3.5];
				relation_string = 'pref';
			case 'distance'
				rating_mats = rating2dist(rating_mats, rating_type, 'distance');
				clim = [0 3.5];
				relation_string = 'prefdis';
		end
end

% Reorder rating matrices based on overall preference for each stimulus
switch object_order
	case ''
		% Do nothing
		% Todo - create equivalent variable for colour order as for
		%	preferenced_ordered, with the default stimulus order
	case 'preference_ordered'
		[rating_mats, pColor_orders] = colour2pref(rating_mats);
end

% Average across participants
switch participant_mean
	case 0 % Do nothing
	case 1 % Average across all participants
		rating_mats = mean(rating_mats, 3);
	case 2 % Average across participants within cluster
		
		% Correlate ratings among participants
		
		% Collapse colour dimensions
		rating_vecs = reshape(rating_mats, [size(rating_mats, 1)*size(rating_mats, 2) size(rating_mats, 3)]);
		% Correlations between participants
		rating_corrs = corr(rating_vecs);
		% Convert correlation to distance
		rating_dists = 1-rating_corrs;
		
		% Create dendrogram from distances
		clusterDistance_method = 'average';
		distances_p = squareform(rating_dists); % convert to pdist vector form
		tree = linkage(rating_dists, clusterDistance_method);
		
		% Get order of leaves in dendrogram
		f = figure('visible', 'off');
		[h, T, outperm] = dendrogram(tree, size(tree, 1)+1);
		close(f);
		
		% Specify participant clusters (from visual inspection)
		switch rating_type
			case 'preference'
				pgroups = {outperm(1:11), outperm(12:end)}; % preference ratings
			case 'similarity'
				pgroups = {outperm(1:17), outperm(18:end)}; % similarity ratings
		end
		group_labels = {'clust1', 'clust2'};
		
		% Average per participant cluster
		rating_mats_groupMean = nan(size(rating_mats, 1), size(rating_mats, 2), length(pgroups));
		for g = 1 : length(pgroups)
			gmean = mean(rating_mats(:, :, pgroups{g}), 3);
			rating_mats_groupMean(:, :, g) = gmean;
		end
		
		% Treat each group as one participant
		rating_mats = rating_mats_groupMean;
	
end

%% Diverging colour scale
% For preference ratings

cmap = flipud(cbrewer('div', 'RdBu', 100));
cmap(cmap < 0) = 0; % for some reason cbrewer is giving negative values...?

%% Plot

figure;
%set(gcf, 'Position', [0 0 1920 1080]);
set(gcf, 'Position', get(0, 'Screensize'));
set(gcf, 'Color', 'w');

if strcmp(rating_type, 'preference') & (strcmp(process_type, 'raw') | strcmp(process_type, 'remap'))
	colormap(cmap);
else
	colormap viridis
end

for p = 1 : size(rating_mats, 3)
	switch participant_mean
		case 0
			subplot(4, 5, p);
		case 2
			subplot(1, 2, p);
	end
	imagesc(rating_mats(:, :, p), clim);
	ax = gca();
	cbar = colorbar;
	set(cbar, 'YTick', (clim(1) : clim(2)/2 : clim(2)));
	axis square
	
	switch participant_mean
		case 0
			title([rating_type newline 'subject' num2str(p)], 'interpreter', 'none');
		case 2
			title(group_labels{p});
	end
	colourTickLabels(ax, cbar, rgb, 0);
end

%% Check violations of transitivity
% Check using distance from middle rating (0)
% Thresholds
%	0
%	-0.5 to +0.5
%	-1.5 to +1.5
%	-2.5 to +2.5
%	-3.5 to +3.5
%
% Only consider a preference if a participant rated outside the
% thresholded range
%	If the rating is within the range, then consider it as no preference
%	for either a or b
%		Consider it as no preference? i.e. a=b?
%		Consider it as left not preferred? i.e. !(a<=b)?
%	Is this any different from just single threshold (instead of range)?

%% Check violations of transitivity
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

%% Check violation of transitivity
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

%%
% Plot total violation counts at each threshold

violation_counts = permute(sum(sum(sum(violation_mat, 1), 2), 3), [4 5 1 2 3]);

figure;
set(gcf, 'Color', 'w');

switch participant_mean
	case 0
		
		% Show each participant
		subplot(1, 2, 1);
		imagesc(violation_counts);
		cbar = colorbar;
		colormap inferno
		
		ylabel(cbar, 'violation count');
		set(gca, 'XTick', (1:numel(b_thresholds)), 'XTickLabel', b_thresholds);
		xlabel([relation_string '(a,b) \leq x']);
		ylabel('participant');
		title('Violation counts per participant');
		
		% Show average across participants
		subplot(1, 2, 2);
		plot(mean(violation_counts, 1));
		axis tight
		
		set(gca, 'XTick', (1:numel(b_thresholds)), 'XTickLabel', b_thresholds);
		xlabel([relation_string '(a,b) \leq x']);
		ylabel('violation count')
		title(['Mean across N=' num2str(size(violation_counts, 1))]);
		
	case {1, 2}
		
		plot(b_thresholds, violation_counts);
		axis tight
		
		tmp1 = linspace(min(b_thresholds), 0, 4);
		tmp2 = linspace(0, max(b_thresholds), 4); tmp2 = tmp2(2:end);
		set(gca, 'XTick', [tmp1 tmp2]);
		
		xlabel([relation_string '(a,b) \leq x']);
		ylabel('violation count')
		title(['Violation count (averaged ratings)']);
		
		if participant_mean == 2
			legend(group_labels, 'Location', 'best');
		end
		
end

%% 
% Plot (a,b), (b,c), (a,c) satisfaction matrices

switch rating_type
	case 'similarity'
		switch process_type
			case 'distance'
				b = find(b_thresholds==3.5);
		end
	case 'preference'
		switch process_type
			case 'remap'
				b = find(b_thresholds==0);
				b = find(b_thresholds==min(abs(b_thresholds)));
		end
end

p = 1;

p_label = ['participant ' num2str(p)];

if participant_mean == 1
	p = 1;
	p_label = ['averaged ratings'];
end

figure;
set(gcf, 'Color', 'w');

% Show preference ratings
subplot(1, 4, 1);
imagesc(rating_mats(:, :, p));
cbar = colorbar;

switch rating_type
	case 'similarity'
		switch process_type
			case 'raw'
			case 'distance'
				ylabel(cbar, 'dissimilarity')
		end
	case 'preference'
		switch process_type
			case 'raw'
			case 'remap'
				ylabel(cbar, 'pref rating');
				colormap(gca, cmap);
			case 'distance'
		end
end

title([p_label]);
xlabel('right colour');
ylabel('left colour');

% Show (a,b)
subplot(1, 4, 2);
imagesc(bmats(:, :, p, b));
cbar = colorbar;
title([relation_string '(a,b) \leq ' num2str(b_thresholds(b))]);
xlabel('b');
ylabel('a');

% Show (b,c)
subplot(1, 4, 3);
imagesc(bmats(:, :, p, b));
cbar = colorbar;
title([relation_string '(b,c) \leq ' num2str(b_thresholds(b))]);
xlabel('c');
ylabel('b');

% Show (a,c)
subplot(1, 4, 4);
imagesc(bmats(:, :, p, b));
cbar = colorbar;
title([relation_string '(a,c) \leq ' num2str(b_thresholds(b))]);
xlabel('c');
ylabel('a');

%%
% Plot (a,b), (b,c), (a,c) valid count

figure;
set(gcf, 'color', 'w');

% Show (a,b)
subplot(1, 3, 1);
imagesc(squeeze(sum(valid_mat(:, :, :, p, b), 3)));
cbar = colorbar;
ylabel(cbar, [relation_string '(a,b) \leq ' num2str(b_thresholds(b)) ' & ' relation_string '(b,c) \leq ' num2str(b_thresholds(b))])
xlabel('b');
ylabel('a');
title([p_label newline 'cond1,cond2 count']);

% Show (b,c)
subplot(1, 3, 2);
imagesc(squeeze(sum(valid_mat(:, :, :, p, b), 1)));
cbar = colorbar;
ylabel(cbar, [relation_string '(a,b) \leq ' num2str(b_thresholds(b)) ' & ' relation_string '(b,c) \leq ' num2str(b_thresholds(b))])
xlabel('c');
ylabel('b');
title([p_label newline 'cond1,cond2 count']);

% Show (a,c)
subplot(1, 3, 3);
imagesc(squeeze(sum(valid_mat(:, :, :, p, b), 2)));
cbar = colorbar;
ylabel(cbar, [relation_string '(a,b) \leq ' num2str(b_thresholds(b)) ' & ' relation_string '(b,c) \leq ' num2str(b_thresholds(b))])
xlabel('c');
ylabel('a');
title([p_label newline 'cond1,cond2 count']);

%%
% Plot (a,b), (b,c), (a,c) violation count

figure;
set(gcf, 'color', 'w');

% Show (a,b)
subplot(1, 3, 1);
imagesc(squeeze(sum(violation_mat(:, :, :, p, b), 3)));
cbar = colorbar;
ylabel(cbar, ['violations at thresh=' num2str(b_thresholds(b))])
xlabel('b');
ylabel('a');
title([p_label newline 'violation count']);

% Show (b,c)
subplot(1, 3, 2);
imagesc(squeeze(sum(violation_mat(:, :, :, p, b), 1)));
cbar = colorbar;
ylabel(cbar, ['violations at thresh=' num2str(b_thresholds(b))])
xlabel('c');
ylabel('b');
title([p_label newline 'violation count']);

% Show (a,c)
subplot(1, 3, 3);
imagesc(squeeze(sum(violation_mat(:, :, :, p, b), 2)));
cbar = colorbar;
ylabel(cbar, ['violations at thresh=' num2str(b_thresholds(b))])
xlabel('c');
ylabel('a');
title([p_label newline 'violation count']);

%%
% Plot percentage of violations at each threshold

violation_counts = permute(sum(sum(sum(violation_mat, 1), 2), 3), [4 5 1 2 3]);
valid_counts = permute(sum(sum(sum(valid_mat, 1), 2), 3), [4 5 1 2 3]);
violation_perc = violation_counts ./ valid_counts;

% Note - nans can occur when count of first two conditions being true is 0
violation_perc(isnan(violation_perc)) = 0;

figure;
set(gcf, 'Color', 'w');

if participant_mean == 0
	
	% Show each participant
	subplot(1, 2, 1);
	imagesc(violation_perc);
	cbar = colorbar;
	colormap inferno
	
	ylabel(cbar, 'violation portion');
	set(gca, 'XTick', (1:numel(b_thresholds)), 'XTickLabel', b_thresholds);
	xlabel([relation_string '(a,b) \leq x']);
	ylabel('participant');
	title('Violation portion per participant');
	
	% Show average across participants
	subplot(1, 2, 2);
	plot(mean(violation_perc, 1));
	axis tight
	
	set(gca, 'XTick', (1:numel(b_thresholds)), 'XTickLabel', b_thresholds);
	xlabel([relation_string '(a,b) \leq x']);
	ylabel('violation count')
	title(['Mean across N=' num2str(size(violation_perc, 1))]);
	
elseif participant_mean == 1
	
	plot(b_thresholds, violation_perc);
	axis tight
	
	tmp1 = linspace(min(b_thresholds), 0, 4);
	tmp2 = linspace(0, max(b_thresholds), 4); tmp2 = tmp2(2:end);
	set(gca, 'XTick', [tmp1 tmp2]);
	
	xlabel([relation_string '(a,b) \leq x']);
	ylabel('violation portion')
	title(['Violation portion (averaged ratings)']);
	
end

%% Shuffle ratings and check for violations, for comparison

nShuffles = 100;

shuffled_violations = nan([nShuffles size(violation_mat)]);
for sh = 1 : nShuffles
	
	shuffled_mats = shuffleRatings(rating_mats);

	[shuffled_viols, shuffled_thresholds] = transitivityViolations(shuffled_mats);
	
	shuffled_violations(sh, :, :, :, :, :) = shuffled_viols;
	
end

% Count violations

% participants x thresholds x shuffles x X x Y x Z
shuffled_violations = permute(shuffled_violations, [5 6 1 2 3 4]);
dims = size(shuffled_violations);
shuffled_violations = reshape(shuffled_violations, [dims(1:3) prod(dims(4:end))]);

% Count
shuffled_violations = sum(shuffled_violations, 4);

%%

% Note - shuffled_thresholds will have the same values as b_thresholds
%	Because - thresholds are determined by unique values in the rating
%		matrices, and the rating values are the same shuffled or not
switch rating_type
	case 'similarity'
		switch process_type
			case 'distance'
				b = find(shuffled_thresholds==3.5);
				b = find(shuffled_thresholds-3.5 == min(abs(shuffled_thresholds-3.5)));
		end
	case 'preference'
		switch process_type
			case 'remap'
				b = find(shuffled_thresholds==0);
				b = find(shuffled_thresholds==min(abs(shuffled_thresholds)));
		end
end


%%

figure;

histogram(shuffled_violations(1, b, :));

%% 
% For each participant, plot histogram of (shuffled) violation counts

figure
set(gcf, 'Color', 'w');
for p = 1 : size(shuffled_violations, 1)
	
	subplot(4, 5, p);
	
	cdfplot(shuffled_violations(p, b, :));
	
	hold on;
	
	% Plot actual
	line(repmat(sum(sum(sum(violation_mat(:, :, :, p, b), 3), 2), 1), [1 2]), ylim, 'Color', 'r');
	
	% Plot 5%
	line(repmat(prctile(shuffled_violations(p, b, :), 5), [1 2]), ylim, 'Color', 'k', 'LineStyle', '--');
	
	title(['preference' newline 'participant ' num2str(p)]);
	xlabel('violations');
	ylabel('cdf');
end

% Create legend in new axes
subplot(4, 5, 4*5);
cdf_line = plot(nan, nan);
actual_line = line([nan nan], [nan nan], 'Color', 'r');
thresh_line = line([nan nan], [nan nan], 'Color', 'k', 'LineStyle', '--');
set(gca, 'Visible', 'off');
legend([cdf_line actual_line, thresh_line], {'cdf', 'actual', '5%'}, 'Location', 'best');

%title(['averaged ratings (POP)' newline 'thresh=' num2str(shuffled_thresholds(b))]);

%%

%{
This is more to do with triangle inequality than with transitivity

If pref(a,b) <= pref(b,c) & pref(b,c) <= pref(a,c),
	then pref(a,b) <= pref(a,c)

Instead of

If a <= b <= c, then a <= c

%% Check violation of transitivity
% transitivity: if x <= y <= z, then x <= z
%
% x is pref(a, b)
% y is pref(b, c)
% z is pref(a, c)
%
% If pref(a,b) <= pref(b,c) & pref(b,c) <= pref(a,c)
%	then pref(a,b) <= pref(a,c)
%	(directly comparing ratings, instead of first binarising)

% Set of colour indexes
colours = (1:size(rating_mats, 1));

% x*y*z*participants
violation_mat = zeros(numel(colours), numel(colours), numel(colours), size(rating_mats, 3));
valid_mat = zeros(size(violation_mat)); % for counting where the first two conditions are true

for p = 1 : size(rating_mats, 3)
	for x = colours
		for y = colours(colours~=x)
			for z = colours(colours~=x & colours~=y)
				
				if rating_mats(x,y,p) <= rating_mats(y,z,p) && rating_mats(y,z,p) <= rating_mats(x,z,p)
					% Then the first two conditions are satisfied
					
					valid_mat(x, y, z, p) = 1;
					
					if ~(rating_mats(x,y,p) <= rating_mats(x,z,p))
						% Then there is a violation
						
						violation_mat(x, y, z, p) = 1;
						
					end
					
				end
				
			end
		end
	end
end

%% Illustrate specific a,b,c combination
% Fix (a,b)
% Show pref(a,b) (constant)
% Show pref(b,c) (as function of c)
% Show pref(a,x) (as function of c)
% Highlight relevant rows in rating matrix

% For participant 1, preference ratings
p = 1;
a = 4;
b = 6;

% For participant 1, similarity distances
p = 1;
a = 4;
b = 6;

% For participant mean, preference ordered preference ratings
%a = 11;
%b = 9;


figure;
set(gcf, 'Color', 'w');

% Show ratings matrix and higlight relevant cells
% Note - imagesc() and rectangle() x,y are flipped (x corresponds to rows
%	in imagesc())
ax1 = subplot(1, 3, 1);
imagesc(rating_mats(:, :, p), clim);
set(gca, 'TickDir', 'out');
axis square

cbar = colorbar;
set(cbar, 'YTick', (clim(1) : clim(2)/2 : clim(2)));

title([p_label]);
xlabel('right colour');
ylabel('left colour');

%{
if strcmp(rating_type, 'preference') & (strcmp(process_type, 'raw') | strcmp(process_type, 'remap'))
	colormap(cmap);
else
	colormap viridis
end
%}
switch rating_type
	case 'similarity'
		switch process_type
			case 'raw'
			case 'distance'
				ylabel(cbar, 'dissimilarity')
		end
	case 'preference'
		switch process_type
			case 'raw'
			case 'remap'
				ylabel(cbar, 'pref rating');
				colormap(gca, cmap);
			case 'distance'
		end
end

hold on

% Highlight (b,c) for all c
rectangle('Position', [0.5, b-0.5, size(rating_mats,2), 1], ...
          'EdgeColor', 'm', 'LineWidth', 2, 'LineStyle', '-');
% Highlight (a,c) for all c
rectangle('Position', [0.5, a-0.5, size(rating_mats,2), 1], ...
          'EdgeColor', 'g', 'LineWidth', 2, 'LineStyle', '-');
% Highlight (a,b)
rectangle('Position', [b-0.5, a-0.5, 1, 1], ...
          'EdgeColor', 'r', 'LineWidth', 3);

%{
% Plot dummies for a legend
hPixel = plot(nan, nan, 'r', 'LineWidth', 3);
hRowB   = plot(nan, nan, 'm', 'LineWidth', 2);
hRowA   = plot(nan, nan, 'g', 'LineWidth', 2);

legend([hPixel, hRowB, hRowA], ...
       {'(a,b)', '(b,c)', '(a,c)'}, ...
       'Location', 'southoutside');
%}

% Show pref(b,c), pref(a,c) as function of c
ax2 = subplot(1, 3, [2 3]);
ab = line([1 12], [rating_mats(a, b, p) rating_mats(a, b, p)], 'Color', 'r', 'LineWidth', 3); % Show fixed (a,b)

hold on

% Show pref(b,c)
bc = plot((1:12)+0.2, rating_mats(b, :, p), 'Color', 'm', 'LineWidth', 2);
% Show pref(a,c)
ac = plot((1:12)+0.4, rating_mats(a, :, p), 'Color', 'g', 'LineWidth', 2);

% Highlight where (a,b) <= (b,c) and (b,c) <= (a,c)

% Join the three lines so we can check visually see the pattern of
% conditions
for c = 1 : size(rating_mats, 2)
	
	if rating_mats(a,b,p) <= rating_mats(b,c,p) && rating_mats(b,c,p) <= rating_mats(a,c,p)
		highlight_colour = [0 0 0];
	else
		highlight_colour = [0.75 0.75 0.75];
	end
	
	plot(...
		[c c+0.2 c+0.4],...
		[rating_mats(a,b,p) rating_mats(b,c,p) rating_mats(a,c,p)],...
		'Color', highlight_colour);
end

% Plot dummies for legend
relevant_highlight = plot(nan, nan, 'Color', [0 0 0]);
irrelevant_highlight = plot(nan, nan, 'Color', [0.75 0.75 0.75]);

legend([ac bc ab relevant_highlight irrelevant_highlight],...
	{'(a,c)', '(b,c)', '(a,b)', '(a,b)\leq(b,c)\leq(a,c)', '!(a,b)\leq(b,c)\leq(a,c)'},...
	'Location', 'best');

axis tight

xlabel('c');

pos1 = get(ax1, 'InnerPosition');
pos2 = get(ax2, 'OuterPosition');
pos2(2) = pos1(2); pos2(4) = pos1(4);
set(ax2, 'OuterPosition', pos2);

%}