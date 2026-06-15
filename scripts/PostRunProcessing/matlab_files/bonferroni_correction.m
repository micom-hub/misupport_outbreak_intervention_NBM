% Perform the Bonferroni correction on the PRCC significance values.
function [bonferroni_significance] = bonferroni_correction(uncorrected_significance)

% Inputs:

% uncorrected_significance: An OxP vector of the significance values (p-values)
% for the P varied parameters for the O model outputs.

% The number of tests is the number of model outputs being tested times the
% number of varied parameters.
[O, P] = size(uncorrected_significance);
tests = O * P;

bonferroni_significance = uncorrected_significance * tests;

end % function bonferroni_correction



