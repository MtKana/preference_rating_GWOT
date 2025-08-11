%% Description

%{

%}

%% Load data

source_dir = 'data_mat/';
source_file = 'data.mat';

loaded = load([source_dir source_file]);

%%

rating_type = 'preference'; % 'similarity' or 'preference'
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

if strcmp(rating_type, 'preference') & (strcmp(process_type, 'raw') | strcmp(process_type, 'remap'))
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
