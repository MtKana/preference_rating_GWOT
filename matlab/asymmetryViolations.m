function [violation_mat] = asymmetryViolations(rating_mats, sign_only)
%asymmetryViolations
% antisymmetry / skew-symmetry: (a,b) = -(b,a)
%
% Inputs:
%	rating_mats = stim x stim x participants matrix
%	sign_only = 1/0
%		If 1, then ignore magnitude of ratings
% Outputs:
%	violation_mat = stim x stim x participants x thresholds matrix
%		Holds 1 where antisymmetry was violated
%		Note, stim x stim should be symmetric

violation_mat = nan(size(rating_mats));

if sign_only == 1
	rating_mats = sign(rating_mats);
end

for p = 1 : size(rating_mats, 3)
	violation_mat(:, :, p) = rating_mats(:, :, p) ~= -rating_mats(:, :, p)';
end

end

