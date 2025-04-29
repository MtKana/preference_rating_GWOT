%% Description

%{

Read csv data files and save into one .mat data file

%}

%% Get list of files

source_dir = '../raw_data/kana_colourpreferencequalia-master/data/';

% Get list of files in data folder
files = cellstr(ls([source_dir '*.csv']));

%% Read data from files

data = cell(length(files), 1);
for f = 1 : length(files)
	
	data{f} = readtable([source_dir files{f}]);
	
end

%% Save as .mat

out_dir = 'data_mat/';
out_file = 'data.mat';

save([out_dir out_file], 'data', 'files');