%%



%{

See if there is a consistent first, second, third, favourite colour across
participants

object_order should be 'preference_ordered'

%}

%% Load data

source_dir = 'data_mat/';
source_file = 'data.mat';

loaded = load([source_dir source_file]);

%%

% Add exclusion of participants based on low double pass correlation
% r < .3

% 'similarity' or 'preference'
rating_type = 'preference';

% 'raw' or 'remap' or 'antisym' or 'distance' - 'remap', 'antisym' only for rating_type 'preference'
process_type = 'antisym';

% '' or 'preference_ordered'
object_order = ''; % 'preference_ordered';

% 0 = no average; 1 = average ratings across participants; 2 = average within participant clusters
% Note - check hardcoded clusters in the code in the following sections
participant_mean = 1;

data = loaded.data;

[colour_hexes, colour_positions, colours_rgb] = getColours();

%% Extract rating matrices

rating_mats = getRatings(data, rating_type, colour_positions);

%% Convert to distances

relation_string = '';

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
			case 'antisym'
				rating_mats = rating2dist(rating_mats, rating_type, 'antisym');
				clim = [-3.5 3.5];
				relation_string = 'apref';
			case 'distance'
				rating_mats = rating2dist(rating_mats, rating_type, 'distance');
				clim = [0 3.5];
				relation_string = 'prefdist';
				% Rescale to match similarity distances
				clim = [0 7];
				rating_mats = rating_mats .* 2;
			case 'distOld'
				rating_mats = rating2dist(rating_mats, rating_type, 'distance_old');
				clim = [0 3.5];
				relation_string = 'prefdistOld';
		end
end

% Reorder rating matrices based on overall preference for each stimulus
switch object_order
	case ''
		% Do nothing
		% Todo - create equivalent variable for colour order as for
		%	preferenced_ordered, with the default stimulus order
		pColor_orders = repmat((1:size(colours_rgb, 1))', [1 size(rating_mats, 3)]);
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

%% Output ratings mats if required

output_ratings = 0;

out_string = [rating_type '_' process_type];

if output_ratings == 1
	
	out_dir = ['ot' filesep];
	out_file = [out_string '.mat'];
	
	save([out_dir out_file], 'rating_mats');
	
	disp(['saved: ' out_dir out_file]);
	
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

if strcmp(rating_type, 'preference') & any(strcmp(process_type, {'raw', 'remap', 'antisym'}))
	colormap(cmap);
else
	colormap viridis
	colormap(cmap(51:end, :));
end

for p = 1 : size(rating_mats, 3)
	switch participant_mean
		case 0
			subplot(4, 5, p);
		case 1
			subplot(4, 5, 1); % To keep the axis size the same
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
			title([relation_string newline 'subject' num2str(p)], 'interpreter', 'none');
		case 1
			title([relation_string newline 'N=19'], 'interpreter', 'none');
		case 2
			title([relation_string newline group_labels{p}]);
	end
	
	colours_rgb_tmp = colours_rgb(pColor_orders(:, p), :);
	colourTickLabels(ax, cbar, colours_rgb_tmp, 0);
end

%% Print figures if required

print_fig = 0;

set(gcf, 'Color', 'w');

if print_fig == 1
	
	figure_name = ['figures/' rating_type '_' process_type '_mean' num2str(participant_mean) '_ratingMats_raw'];
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end

%% Show MDS (for distances)

figure;
set(gcf, 'Position', get(0, 'Screensize'));
set(gcf, 'Color', 'w');
for p = 1 : size(rating_mats, 3)
	switch participant_mean
		case 0
			subplot(4, 5, p);
		case 1
			subplot(4, 5, p);
		case 2
			subplot(1, 2, p);
	end
	
	Y = mdscale(rating_mats(:, :, p), 2);
	
	scatter(Y(:, 1), Y(:, 2), 100, colours_rgb, 'filled', 'MarkerFaceAlpha', 0.8);
	
	switch participant_mean
		case 0
			title([relation_string newline 'subject' num2str(p)], 'interpreter', 'none');
		case 2
			title([relation_string newline group_labels{p}]);
	end
	
	set(gca, 'XTick', [], 'YTick', []);
	axis square
	
end

%% Print figures if required

print_fig = 0;

set(gcf, 'Color', 'w');

if print_fig == 1
	
	figure_name = ['figures/' rating_type '_' process_type '_mean' num2str(participant_mean) '_mds_raw'];
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end

%% Convert rgb to hsv

colours_hsv = rgb2hsv(colours_rgb);
colours_hsv(:, 1) = colours_hsv(:, 1) * 360;

%% Show original colours in hue circle

figure;
set(gcf, 'Color', 'w');

xs = sind(colours_hsv(:, 1));
ys = cosd(colours_hsv(:, 1));

ax = axes;

plotHueCircle(ax);

scatter(xs, ys, 500, colours_rgb, 'filled');

axis square

%% Plot order of preferred colours for each participant

figure;
set(gcf, 'Color', 'w');

for p = 1 : size(rating_mats, 3)
	
	c_xs = nan(size(colours_hsv, 1), 1);
	c_ys = nan(size(c_xs));
	for c = 1 : size(colours_hsv, 1)
		
		% Get the Cth preferred colour from each participant
		% Get the hue angle
		c_angles = colours_hsv(pColor_orders(c, p), 1);
		
		% Convert to (x,y) coordinate/vector
		xs = sind(c_angles);
		ys = cosd(c_angles);
		
		% Average vectors together (for multiple participants)
		c_xs(c) = mean(xs);
		c_ys(c) = mean(ys);
		
	end
	
	ax = subplot(4, 5, p);
	
	plotHueCircle(ax);
	
	text(c_xs, c_ys, arrayfun(@num2str, (1:size(colours_hsv, 1)), 'UniformOutput', false));
	
	xlim([-1 1]);
	ylim([-1 1]);
	
	axis square
	
end

%% Plot a shape for each participant

figure;
set(gcf, 'Color', 'w');

for p = 1 : 1%size(rating_mats, 3)
	
	c_xs = nan(size(colours_hsv, 1), 1);
	c_ys = nan(size(c_xs));
	for c = 1 : size(colours_hsv, 1)
		
		% Get the Cth preferred colour from each participant
		% Get the hue angle
		c_angles = colours_hsv(pColor_orders(c, p), 1);
		
		% Convert to (x,y) coordinate/vector
		xs = sind(c_angles);
		ys = cosd(c_angles);
		
		% Average vectors together (for multiple participants)
		c_xs(c) = mean(xs);
		c_ys(c) = mean(ys);
		
	end
	
	ax = subplot(4, 5, p);
	
	plotHueCircle(ax);
	
	% Join the points together with lines, highlight the starting and end
	% points
	plot(0.9*c_xs, 0.9*c_ys, 'k', 'LineWidth', 2);
	
	text(0.9*c_xs([1 end]), 0.9*c_ys([1 end]), arrayfun(@num2str, [1 size(colours_hsv, 1)], 'UniformOutput', false));
	
	xlim([-1 1]);
	ylim([-1 1]);
	
	axis square
	
end

%% Plot preference order encoded by vector length
% Rotate each plot so that the first preference points upwards

figure;
set(gcf, 'Color', 'w');

for p = 1 : size(rating_mats, 3)
	
	c_xs = nan(size(colours_hsv, 1), 1);
	c_ys = nan(size(c_xs));
	for c = 1 : size(colours_hsv, 1)
		
		% Get the Cth preferred colour from each participant
		% Get the hue angle
		c_angles = colours_hsv(pColor_orders(c, p), 1);
		
		% Convert to (x,y) coordinate/vector
		xs = sind(c_angles);
		ys = cosd(c_angles);
		
		% Average vectors together (for multiple participants)
		c_xs(c) = mean(xs);
		c_ys(c) = mean(ys);
		
	end
	
	ax = subplot(4, 5, p);
	
	plotHueCircle(ax, colours_rgb);
	
	c_lengths = linspace(1, 0.2, numel(c_xs));
	for c = 1 : size(colours_hsv, 1)
		
		% Plot all the way to the circumference
		plot([0 c_xs(c)], [0 c_ys(c)], 'k:', 'LineWidth', 0.5);
		
		% Plot preference as length
		plot([0 c_xs(c)*c_lengths(c)], [0 c_ys(c)*c_lengths(c)], 'k', 'LineWidth', 1);
	end
	
	xlim([-1 1]);
	ylim([-1 1]);
	axis vis3d
	
	% Note - 0 degrees is at the top (i.e. (x,y)=(0,1))
	rotate_angle = 360 - colours_hsv(pColor_orders(1, p), 1);
	view([rotate_angle 90]);
	
end

%% Join lines to create a shape, per participant

% We need to know the order of colours in the hue circle to do this

%% Version with stimulus hues equally spaced
% And join the lines to create a polygon

% Assign angles for each colour
colour_angles = (0 : 360/size(colours_rgb, 1) : 360);
colour_angles = colour_angles(1:end-1); % drop the last one, 360

figure;
set(gcf, 'Position', get(0, 'Screensize'));
set(gcf, 'Color', 'w');

for p = 1 : size(rating_mats, 3)
	
	c_xs = nan(size(colours_hsv, 1), 1);
	c_ys = nan(size(c_xs));
	for c = 1 : size(colours_hsv, 1)
		
		% Get the Cth preferred colour from each participant
		c_angles = colour_angles(pColor_orders(c, p));
		
		% Convert to (x,y) coordinate/vector
		xs = sind(c_angles);
		ys = cosd(c_angles);
		
		% Average vectors together (for multiple participants)
		c_xs(c) = mean(xs);
		c_ys(c) = mean(ys);
		
	end
	
	ax = subplot(4, 5, p);
	hold on;
	
	% Plot the original colours, equidistant
	ring_xs = sind(colour_angles);
	ring_ys = cosd(colour_angles);
	scatter(ring_xs, ring_ys, 100, colours_rgb, 'filled');
	
	c_lengths = linspace(1, 0.2, numel(c_xs));
	for c = 1 : size(colours_hsv, 1)
		
		% Plot all the way to the circumference
		plot([0 c_xs(c)], [0 c_ys(c)], 'k:', 'LineWidth', 0.5);
		
		% Plot preference as length
		plot([0 c_xs(c)*c_lengths(c)], [0 c_ys(c)*c_lengths(c)], 'k', 'LineWidth', 1);
		
	end
	
	% Join up the lines
	%plot(c_xs, c_ys);
	
	% Join up the lines into a polygon (needs same colour order across
	% participants)
	[tmp, tmpi] = sort(pColor_orders(:, p));
	plot(...
		[c_xs(tmpi).*c_lengths(tmpi)'; c_xs(tmpi(1)).*c_lengths(tmpi(1))],...
		[c_ys(tmpi).*c_lengths(tmpi)'; c_ys(tmpi(1)).*c_lengths(tmpi(1))],...
		'LineWidth', 1, 'Color', 'k');
	
	xlim([-1 1]);
	ylim([-1 1]);
	axis vis3d
	
	% Note - 0 degrees is at the top (i.e. (x,y)=(0,1))
	rotate_angle = 360 - colour_angles(pColor_orders(1, p));
	view([rotate_angle 90]);
	
end

%% Print figures if required

print_fig = 0;

set(gcf, 'Color', 'w');

if print_fig == 1
	
	figure_name = ['figures/' rating_type '_' process_type '_mean' num2str(participant_mean) '_shapes_raw'];
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end

%% Version with stimulus hues equally spaced
% Plot the actual mean rating (not the preference rank)
% (uses preference_ordered)

% Assign angles for each colour
colour_angles = (0 : 360/size(colours_rgb, 1) : 360);
colour_angles = colour_angles(1:end-1); % drop the last one, 360

figure;
set(gcf, 'Position', get(0, 'Screensize'));
set(gcf, 'Color', 'w');

for p = 1 : size(rating_mats, 3)
	
	c_xs = nan(size(colours_rgb, 1), 1);
	c_ys = nan(size(c_xs));
	for c = 1 : size(colours_rgb, 1)
		
		% Get the Cth preferred colour from each participant
		c_angles = colour_angles(pColor_orders(c, p));
		
		% Convert to (x,y) coordinate/vector
		xs = sind(c_angles);
		ys = cosd(c_angles);
		
		% Average vectors together
		c_xs(c) = mean(xs);
		c_ys(c) = mean(ys);
		
	end
	
	ax = subplot(4, 5, p);
	hold on;
	
	% Plot the original colours, equidistant
	ring_xs = sind(colour_angles);
	ring_ys = cosd(colour_angles);
	scatter(ring_xs*7, ring_ys*7, 100, colours_rgb, 'filled');
	
	% Need to transform [-3.5 to 3.5] to [7 0] (length represents
	% preference strength
	c_lengths = mean(abs(rating_mats(:, :, p)-3.5), 2)';
	for c = 1 : size(colours_rgb, 1)
		
		% Plot all the way to the circumference
		plot([0 c_xs(c)], [0 c_ys(c)], 'k:', 'LineWidth', 0.5);
		
		% Plot preference rating (mean) as length
		plot([0 c_xs(c)*c_lengths(c)], [0 c_ys(c)*c_lengths(c)], 'k', 'LineWidth', 1);
		
	end
	
	% Join up the lines
	%plot(c_xs, c_ys);
	
	% Join up the lines into a polygon (needs same colour order across
	% participants)
	[tmp, tmpi] = sort(pColor_orders(:, p));
	plot(...
		[c_xs(tmpi).*c_lengths(tmpi)'; c_xs(tmpi(1)).*c_lengths(tmpi(1))],...
		[c_ys(tmpi).*c_lengths(tmpi)'; c_ys(tmpi(1)).*c_lengths(tmpi(1))],...
		'LineWidth', 1, 'Color', 'k');
	
	%xlim([-1 1]);
	%ylim([-1 1]);
	axis vis3d
	
	% Note - 0 degrees is at the top (i.e. (x,y)=(0,1))
	rotate_angle = 360 - colour_angles(pColor_orders(1, p));
	view([rotate_angle 90]);
	
end

%% Version with stimulus hues equally spaced
% Plot the actual mean rating (not the preference rank)
% Directly use the rating from each colour

% Assign angles for each colour
colour_angles = (0 : 360/size(colours_rgb, 1) : 360);
colour_angles = colour_angles(1:end-1); % drop the last one, 360

figure;
set(gcf, 'Position', get(0, 'Screensize'));
set(gcf, 'Color', 'w');

for p = 1 : size(rating_mats, 3)
	
	c_xs = nan(size(colours_rgb, 1), 1);
	c_ys = nan(size(c_xs));
	
	c_xs = sind(colour_angles);
	c_ys = cosd(colour_angles);
	
	ax = subplot(4, 5, p);
	hold on;
	
	% Plot the original colours, equidistant
	ring_xs = sind(colour_angles);
	ring_ys = cosd(colour_angles);
	scatter(ring_xs*7, ring_ys*7, 100, colours_rgb, 'filled');
	
	% Need to transform [-3.5 to 3.5] to [7 0] (length represents
	% preference strength
	c_lengths = mean(abs(rating_mats(:, :, p)-3.5), 2)';
	for c = 1 : size(colours_rgb, 1)
		
		% Plot all the way to the circumference
		plot([0 c_xs(c)], [0 c_ys(c)], 'k:', 'LineWidth', 0.5);
		
		% Plot preference rating (mean) as length
		plot([0 c_xs(c)*c_lengths(c)], [0 c_ys(c)*c_lengths(c)], 'k', 'LineWidth', 1);
		
	end
	
	% Join up the lines
	%plot(c_xs, c_ys);
	
	% Join up the lines into a polygon (needs same colour order across
	% participants)
	plot(...
		[c_xs.*c_lengths c_xs(1).*c_lengths(1)],...
		[c_ys.*c_lengths c_ys(1).*c_lengths(1)],...
		'LineWidth', 1, 'Color', 'k');
	
	%xlim([-1 1]);
	%ylim([-1 1]);
	axis vis3d
	
	% Note - 0 degrees is at the top (i.e. (x,y)=(0,1))
	[~, top_colour] = max(c_lengths);
	rotate_angle = 360 - colour_angles(top_colour);
	view([rotate_angle 90]);
	
end

%% Version with stimulus hues equally spaced
% Plot the rating rank, obtained from the ratings for each colour
% Directly use the rating from each colour

% Assign angles for each colour
colour_angles = (0 : 360/size(colours_rgb, 1) : 360);
colour_angles = colour_angles(1:end-1); % drop the last one, 360

figure;
set(gcf, 'Position', get(0, 'Screensize'));
set(gcf, 'Color', 'w');

for p = 1 : size(rating_mats, 3)
	
	c_xs = nan(size(colours_rgb, 1), 1);
	c_ys = nan(size(c_xs));
	
	c_xs = sind(colour_angles);
	c_ys = cosd(colour_angles);
	
	ax = subplot(4, 5, p);
	hold on;
	
	% Plot the original colours, equidistant
	ring_xs = sind(colour_angles);
	ring_ys = cosd(colour_angles);
	scatter(ring_xs, ring_ys, 100, colours_rgb, 'filled');
	
	% Need to transform [-3.5 to 3.5] to [7 0] (length represents
	% preference strength
	c_lengths = mean(abs(rating_mats(:, :, p)-3.5), 2)';
	% Convert to ranks
	[~, c_ranks] = sort(c_lengths, 'descend');
	c_lengths = linspace(1, 0.2, numel(c_xs));
	c_lengths(c_ranks) = c_lengths;
	for c = 1 : size(colours_rgb, 1)
		
		% Plot all the way to the circumference
		plot([0 c_xs(c)], [0 c_ys(c)], 'k:', 'LineWidth', 0.5);
		
		% Plot preference rating (mean) as length
		plot([0 c_xs(c)*c_lengths(c)], [0 c_ys(c)*c_lengths(c)], 'k', 'LineWidth', 1);
		
	end
	
	% Join up the lines
	%plot(c_xs, c_ys);
	
	% Join up the lines into a polygon (needs same colour order across
	% participants)
	plot(...
		[c_xs.*c_lengths c_xs(1).*c_lengths(1)],...
		[c_ys.*c_lengths c_ys(1).*c_lengths(1)],...
		'LineWidth', 1, 'Color', 'k');
	
	%xlim([-1 1]);
	%ylim([-1 1]);
	axis vis3d
	
	% Note - 0 degrees is at the top (i.e. (x,y)=(0,1))
	[~, top_colour] = max(c_lengths);
	rotate_angle = 360 - colour_angles(top_colour);
	view([rotate_angle 90]);
	
end

%%

% c is the Cth most preferred colour
c_xs = nan(size(colours_hsv, 1), 1);
c_ys = nan(size(c_xs));
for c = 1 : size(colours_hsv, 1)
	
	% Get the Cth preferred colour from each participant
	% Get the hue angle
	c_angles = colours_hsv(pColor_orders(c, :), 1);
	
	% Convert to (x,y) coordinate/vector
	xs = sind(c_angles);
	ys = cosd(c_angles);
	
	% Average vectors together
	c_xs(c) = mean(xs);
	c_ys(c) = mean(ys);

end

figure;
set(gcf, 'Color', 'w');

text(c_xs, c_ys, arrayfun(@num2str, (1:size(colours_hsv, 1)), 'UniformOutput', false));

xlim([-1 1]);
ylim([-1 1]);

%%

% Clusters based on raw preference ratings
pClusters = {...
	[1 15 12 7 11 5 8 10 4 13 18],...
	[2 19 6 3 16 9 14 17]...
	};

pCluster_coords = cell(size(pClusters));

for pClust = 1 : numel(pClusters)
	pCluster_coords{pClust} = struct();
	
	c_xs = nan(size(colours_hsv, 1), 1);
	c_ys = nan(size(c_xs));
	for c = 1 :size(colours_hsv, 1)
		
		c_angles = colours_hsv(pColor_orders(c, pClusters{pClust}), 1);
		
		% Convert to (x,y) coordinate/vector
		xs = sind(c_angles);
		ys = cosd(c_angles);
		
		c_xs(c) = mean(xs);
		c_ys(c) = mean(ys);
		
	end
	
	% These are the mean coordinates for per ranked colours, for each
	% cluster
	pCluster_coords{pClust}.xs = c_xs;
	pCluster_coords{pClust}.ys = c_ys;
	
end

figure;
set(gcf, 'Color', 'w');

for pClust = 1 : numel(pClusters)
	
	ax = subplot(1, numel(pClusters), pClust);
	
	plotHueCircle(ax, colours_rgb);
	
	hold on;
	
	text(pCluster_coords{pClust}.xs, pCluster_coords{pClust}.ys, arrayfun(@num2str, (1:size(colours_hsv, 1)), 'UniformOutput', false));

	xlim([-1 1]);
	ylim([-1 1]);
	
	axis square
	
	title(['cluster' num2str(pClust)]);
	xlabel('sin(hue)');
	ylabel('cos(hue)');
end

%%

figure;

% Define a circular domain
[x, y] = meshgrid(-1:.01:1);
[theta, rho] = cart2pol(x, y);

% Set saturation to be based on the radius, capped at 1
saturation = min(1, rho); 

% Define constant brightness and hue based on angle
hue = (theta + pi) / (2 * pi); % Shift hue from [-pi, pi] to [0, 1]
value = 0.9; % Constant value

% Create the color image in HSV
hsvImage = cat(3, hue, saturation, ones(size(hue)) * value);

% Convert to RGB
rgbImage = hsv2rgb(hsvImage);

% Mask out the background (outside the radius)
rgbImage(repmat(rho > 1, [1, 1, 3])) = 1;

x_start = -1; % X-coordinate of the first column
x_end = 1;  % X-coordinate of the last column
y_start = -1; % Y-coordinate of the first row
y_end = 1;  % Y-coordinate of the last row

% Display the image
imshow(rgbImage, 'XData', [x_start, x_end], 'YData', [y_start, y_end]);
title('HSV Color Wheel (Radius 1)');