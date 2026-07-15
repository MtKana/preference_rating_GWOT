function [] = plotHueCircle(ax, colours_rgb)
%plotHueCircle
% Plots a hue circle centred at origin with radius 1
%
% Inputs:
%	ax = axes handle
% Outputs:

axes(ax);

hold on;

% Define the number of points to use for the circle
numPoints = 1000;

% Create angles for the circle, ranging from 0 to 360 degrees
angle = linspace(0, 360, numPoints);

% Calculate x and y coordinates for a unit circle
x = sind(angle);
y = cosd(angle);

% Generate a colormap representing the full range of hues in HSV
% The 'hsv' function in MATLAB creates a colormap where hue varies from 0 to 1
% (corresponding to 0 to 360 degrees), while saturation and value are 1.
colors = hsv(numPoints);

% Plot each point of the circle with its corresponding hue color
for k = 1 : numPoints
    plot(x(k), y(k), '.', 'MarkerSize', 20, 'Color', colors(k,:));
    hold on; % Keep the current plot for adding more points
end

% Plot specific colours
if nargin == 2
	
	colours_hsv = rgb2hsv(colours_rgb);
	colours_hsv(:, 1) = colours_hsv(:, 1) * 360;
	
	xs = sind(colours_hsv(:, 1));
	ys = cosd(colours_hsv(:, 1));
	scatter(xs, ys, 100, colours_rgb, 'filled');
end

end