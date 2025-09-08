function [ax2, cbar_x, cbar_y] = colourTickLabels(ax1, cb1, colours, transform)
% colourTickLabels
%
% Add colorbars as tick labels to a plot
%
% Inputs:
%	ax1 = axis handle
%	cb1 = colorbar handle for ax1
%	colours = Nx3 matrix, specifies RGB colours
%	transform = 0; 1 to reduce changing of figure size to accommodate
%		the new colorbars
% Outputs:
%	ax2 = axis handle for the dummy axis used for the x ticks

% Get inner position of the axes
ax_pos = tightPosition(ax1);

%{
% Resize axes to accommodate new colorbars
ax_pos(3) = ax_pos(3) - ax_pos(1); % Reduce width to accommodate colorbar to the right
ax_pos(1) = ax_pos(1)*2; % Shift right
ax_pos(4) = ax_pos(4) - ax_pos(2); % Reduce height to account for new starting height
ax_pos(2) = ax_pos(2)*2; % Shift up to accommodate a colorbar below
%}

switch transform
	case 1
		trans = 0.25;
	case 0
		trans = 0;
end
ax_pos(3) = ax_pos(3) - (ax_pos(1)*trans); % Reduce width to accommodate colorbar to the right
ax_pos(1) = ax_pos(1) + (ax_pos(1)*trans); % Shift right
ax_pos(4) = ax_pos(4) - (ax_pos(2)*trans); % Reduce height to account for new starting height
ax_pos(2) = ax_pos(2) + (ax_pos(2)*trans); % Shift up to accommodate a colorbar below
set(ax1, 'Position', ax_pos);

% Create dummy axis over the desired axis
ax2 = axes('Visible', 'off');
colormap(ax2, colours);

% axis square to match ax1
if all(ax1.PlotBoxAspectRatio == 1)
	axis(ax2, 'square');
end

% Add colorbars
cbar_x = colorbar(ax2, 'southoutside');
cbar_y = colorbar(ax2, 'westoutside');

% Match position to ax1
set(ax2, 'InnerPosition', ax_pos);
drawnow;
linkprop([ax1, ax2], 'Position');

ax2_pos = tightPosition(ax2);

% Remove x, y ticks to move colorbars closer
set(ax1, 'XTick', [], 'YTick', []);
set(ax2, 'XTick', [], 'YTick', []);

% Adjust colorbar positions so that the distance from the main axes matches
%	the default colorbar on the right
% Note - this causes matlab to unlink the position attribute, so the
%	height/width can end up not matching the axes height
move_cbar_closer = 0;
if move_cbar_closer == 1
	
	% distance from right edge to right colorbar
	cdist = cbar_pos(1) - (ax_pos(1)+ax_pos(3));
	
	cbar2_pos = get(cbar_x, 'Position');
	cbar2_pos(1) = ax_pos(2) - cdist - cbar2_pos(2);
	cbar2_pos(2) = ax2_pos(1);
	cbar2_pos(4) = ax2_pos(3);
	set(cbar_x, 'Position', cbar2_pos);
	
	cbar3_pos = get(cbar_y, 'Position');
	cbar3_pos(1) = ax_pos(1) - cdist - cbar3_pos(3);
	cbar3_pos(2) = ax2_pos(2);
	cbar3_pos(4) = ax2_pos(4);
	set(cbar_y, 'Position', cbar3_pos);
	
end

set(cbar_x, 'XTick', []);
set(cbar_y, 'YTick', []);

set(cbar_y, 'YDir', 'reverse'); % Make 1 start from the top

xlabel(cbar_x, 'right colour');
ylabel(cbar_y, 'left colour');

end

