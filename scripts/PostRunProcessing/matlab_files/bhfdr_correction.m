% Perform the Benjamini and Hochberg False Discovery Rate (BHFDR) correction on
% the PRCC significance values.
% See: https://en.wikipedia.org/wiki/False_discovery_rate
function [bhfdr_significance, bhfdr_significant_prcc, bhfdr_significant_significance] = bhfdr_correction(uncorrected_prcc, uncorrected_significance, alpha, paramHeaders, dataHeaders)

% In comments below:
%
% P is the number of varied parameters.
%
% O is the number of model outputs.
%
% ---------------------------------------------------------------------
% Function arguments
% ---------------------------------------------------------------------
%
% uncorrected_significance: The uncorrected PRCC significance values, an OxP
% matrix of the significance values for the PRCC
% values in uncorrected_prcc.
%
% alpha: The alpha value to use as the cutoff for significant PRCCs.
%
% paramHeaders is a 1xP cell array of strings. Used for debug print.
%
% dataHeaders is an Ox1 cell array of strings. Used for debug print.
%
% ---------------------------------------------------------------------
% Function outputs
% ---------------------------------------------------------------------
%
% bhfdr_significance: An OxP matrix that is the uncorrected_significance corrected
% using the BHFDR correction.
%
% bhfdr_significant_prcc: An OxP matrix that is the same as uncorrected_prcc, except
% when an uncorrected prcc value is not significant, according
% to the BHFDR test, that element of bhfdr_significant_prcc is 1.
%
% bhfdr_significant_signifcance: An OxP matrix that is the same as bhfdr_significance,
% except when a significance value is not significant, according
% to the BHFDR test, that element of bhfdr_significant_signifcance is 1.

[O, P] = size(uncorrected_significance);
test_count = O * P;

% Create a matrix that has OxP rows and 4 columns.
%
% Column 1 is the uncorrected significance values.
% Column 2 is the corrected significance values.
%
% Columns 3 and 4 are the index for the uncorrected significance value in the
% array uncorrected_significance.
% 
% uncorrected_significance(1,1 ) corrected_significance 1 1
% uncorrected_significance(1,2 ) corrected_significance 1 2
% .         .
% .         .
% .         .
% uncorrected_significance(1,P ) corrected_significance 1 P
% uncorrected_significance(2,1 ) corrected_significance 2 1
% uncorrected_significance(2,2 ) corrected_significance 2 2
% .         .
% .         .
% .         .
% uncorrected_significance(2,P ) corrected_significance 2 P
% .
% .
% .
% uncorrected_significance(O,1 ) corrected_significance O 1
% uncorrected_significance(O,2 ) corrected_significance O 2
% .         .
% .         .
% .         .
% uncorrected_significance(O,P ) corrected_significance O P

% Transform the uncorrected significance values into a linear vector.
% Remember the i and j index from the original uncorrected significance
% value matrix, so we can created a matrix of the corresponding corrected
% values.
uncorrected_vector = zeros(test_count, 3);
k = 0;
for i = 1:O

    for j = 1:P
        k = k + 1;
        uncorrected = uncorrected_significance(i, j);
        uncorrected_vector(k, :) = [uncorrected, i, j];
    end

end

% Sort the rows of the uncorrected_vector by the uncorrected significance values.
uncorrected_vector = sortrows(uncorrected_vector, 1);

% Define the corrected significance values.
fdr = zeros(test_count, 4);
for k = 1:test_count
    uncorrected =  uncorrected_vector(k, 1);
    corrected = (uncorrected * test_count) / k;
    i =  uncorrected_vector(k, 2);
    j =  uncorrected_vector(k, 3);
    fdr(k, :) = [uncorrected, corrected, i, j];
end

% Find the largest index, k, such that corrected(k) is <= alpha.
%fid = fopen('bhfdr-rank-order.csv', 'wt');
maxK = 0;
for k = 1:test_count
    i = fdr(k, 3);
    j = fdr(k, 4);
    %fprintf(fid, '%s,%s,%g,%d,%g\n', dataHeaders{i+1}, paramHeaders{j}, fdr(k, 1), k, fdr(k, 2));
    if fdr(k, 2) <= alpha
        maxK = k;
    end
end
%fclose(fid);

% Define the result arrays.
bhfdr_significance = ones(O, P);
bhfdr_significant_prcc = ones(O, P);
bhfdr_significant_significance = ones(O, P);
for k = 1:test_count

    i = fdr(k, 3);
    j = fdr(k, 4);
    bhfdr_significance(i, j) = fdr(k, 2);

    if k <= maxK
        bhfdr_significant_prcc(i, j) = uncorrected_prcc(i, j);
        bhfdr_significant_significance(i, j) = fdr(k, 2);
    end
end

end % function bhfdr_correction
