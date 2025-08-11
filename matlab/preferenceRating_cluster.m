%% Description

%{

Cluster subjects based on correlation between preference rating matrices

%}

%% Load data

source_dir = 'data_mat/';
source_file = 'data.mat';

loaded = load([source_dir source_file]);

%%

rating_type = 'preference';
process_type = 'remap'; % 'raw' or 'remap' or 'distance' - 'remap' only for rating_type 'preference'

data = loaded.data;

[colours, colour_positions, colours_rgb] = getColours();

%% Extract rating matrices

rating_mats = getRatings(data, rating_type, colour_positions);

%% Convert to distances

switch rating_type
	case 'similarity'
		
		switch process_type
			case 'raw'
				dist_mats = rating_mats;
			case 'distance'
				dist_mats = rating2dist(rating_mats, rating_type, []);
		end
		
		clim = [0 7];
		
	case 'preference'
		
		switch process_type
			case 'raw'
				dist_mats = rating_mats;
				clim = [0 7];
			case 'remap'
				dist_mats = rating2dist(rating_mats, rating_type, 'remap');
				clim = [-3.5 3.5];
			case 'distance'
				dist_mats = rating2dist(rating_mats, rating_type, 'distance');
				clim = [0 3.5];
		end
end

%% Diverging colour scale
% For preference ratings

cmap = flipud(cbrewer('div', 'RdBu', 100));
cmap(cmap < 0) = 0; % for some reason cbrewer is giving negative values...?

%% Plot

figure;

if strcmp(process_type, 'remap')
	colormap(cmap);
else
	colormap viridis
end

for p = 1 : length(data)
	subplot(4, 5, p);
	imagesc(dist_mats(:, :, p), clim);
	cbar = colorbar;
	set(cbar, 'YTick', (clim(1) : clim(2)/2 : clim(2)));
	title([rating_type newline 'subject' num2str(p)], 'interpreter', 'none');
	axis square
end

%% Correlate ratings among participants

% Collapse colour dimensions
rating_vecs = reshape(rating_mats, [length(colours)*length(colours) size(rating_mats, 3)]);

% Correlations between participants
rating_corrs = corr(rating_vecs);

% Convert correlation to distance
rating_dists = 1-rating_corrs;

%% Euclidean distances among participants
%{
% Collapse colour dimensions
rating_vecs = reshape(rating_mats, [length(colours)*length(colours) size(rating_mats, 3)]);

% Distances between participants
rating_dists = squareform(pdist(rating_vecs', 'euclidean'));
%}
%% Create dendrogram from distances

clusterDistance_method = 'average';

distances_p = squareform(rating_dists); % convert to pdist vector form

tree = linkage(rating_dists, clusterDistance_method);

%% Plot dendrogram

figure;
set(gcf, 'color', 'w');
[h, T, outperm] = dendrogram(tree, size(tree, 1)+1);
xlabel('participant');
axis tight

%% Plot distance matrices

figure;
set(gcf, 'color', 'w');

subplot(1, 2, 1);
imagesc(rating_dists); c = colorbar;
title(c, '1-r');
set(gca, 'XTick', (1:size(rating_dists, 1)));
set(gca, 'YTick', (1:size(rating_dists, 1)));
xlabel('participant');
ylabel('participant');
axis square;

subplot(1, 2, 2);
imagesc(rating_dists(outperm, outperm)); c = colorbar;
title(c, '1-r');
set(gca, 'XTick', (1:size(rating_dists, 1)), 'XTickLabel', outperm);
set(gca, 'YTick', (1:size(rating_dists, 1)), 'YTickLabel', outperm);
xlabel('participant');
ylabel('participant');
axis square;

%% Plot matrices using the dendrogram order

figure;
set(gcf, 'color', 'w');

if strcmp(rating_type, 'preference') & (strcmp(process_type, 'raw') | strcmp(process_type, 'remap'))
	colormap(cmap);
else
	colormap viridis
end

for p_counter = 1 : length(data)
	p = outperm(p_counter);
	subplot(4, 5, p_counter);
	imagesc(dist_mats(:, :, p), clim);
	cbar = colorbar;
	set(cbar, 'YTick', (clim(1) : clim(2)/2 : clim(2)));
	title([rating_type newline 'subject' num2str(p)], 'interpreter', 'none');
	axis square
end

%% Plot average for each cluster

% Get N clusters from the tree
% Note that this loses the order of the dendrogram (where similar
% participants are adjacent)
%	vector values give the cluster which that index belongs to
% pgroups = cluster(tree, 'MaxClust', 2);

% From visual inspection of the histogram
switch rating_type
	case 'preference'
		pgroups = {outperm(1:11), outperm(12:end)}; % preference ratings
	case 'similarity'
		pgroups = {outperm(1:17), outperm(18:end)}; % similarity ratings
end

figure;
set(gcf, 'color', 'w');

if strcmp(rating_type, 'preference') & (strcmp(process_type, 'raw') | strcmp(process_type, 'remap'))
	colormap(cmap);
else
	colormap viridis
end

for g = 1 : length(pgroups)
	gmean = mean(dist_mats(:, :, pgroups{g}), 3);
	
	subplot(1, length(pgroups), g);
	imagesc(gmean, clim);
	cbar = colorbar;
	set(cbar, 'YTick', (clim(1) : clim(2)/2 : clim(2)));
	title([rating_type newline 'cluster' num2str(g)], 'interpreter', 'none');
	axis square
end

%% Output clusters

out_dir = 'participantClusters\';

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

out_file = ['clusters_' rating_type '_' process_type '.mat'];

participants = loaded.files;
clusters = pgroups;


save_clusters = 0;
if save_clusters == 1
	save([out_dir out_file],...
		'participants', 'clusters',...
		'rating_type', 'process_type');
end

