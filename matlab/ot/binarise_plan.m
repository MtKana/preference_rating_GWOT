function [plan_binary] = binarise_plan(plan, makeplot)
%BINARISE_PLAN
% Binarise a transport plan - select the cell with maximum weight as the
% match
%
% Inputs:
%	plan = NxN matrix
%	makeplot = int; if 1, creates a figure showing the binarised plan
%
% Outputs:
%	plan_binary = NxN matrix

% Binarise the plan - select maximum weight in each row
[M, I] = max(plan, [], 2);
plan_binary = zeros(size(plan));
for stim = 1 : size(plan, 1)
	plan_binary(stim, I(stim)) = 1;
end

if makeplot == 1
	figure;
	set(gcf, 'Color', 'w');
	
	subplot(1, 2, 1);
	imagesc(plan);
	colorbar;
	title(['original']);
	axis square;
	colormap inferno
	
	subplot(1, 2, 2);
	imagesc(plan_binary);
	colorbar;
	title(['binarised']);
	axis square;
	colormap inferno
end

end

