% Calculation of PRCCs and their significance (uncorrected p-value)
%
% This script will perform PCC rather than PRCC if the optional argument,
% analysisType, is 'Pearson'.

% Original Author: Simeone Marino, May 29 2007
% Modified: Nicholas Cilfone 2012-2013

function [prcc, significance, pcc_ztest, pcc_ztest_signif] = ...
    performPrcc(lhsMatrix, data, analysisType)

% In comments below:
%
% R: The number of runs. If the data is not averaged over replications for
%    each experiment it is the number of experiments times the number of
%    replications. If the data is averaged over the replications for each
%    experiment it is the number of experiments.
%
% P: The number of varied parameters.
%
% O: The number of model outputs.
%
% TP: The number of timepoints.
%
% Inputs:
%     lhsMatrix: An RxP matrix of varied parameter values.
%
%     data: An O x R x TP matrix of the model outputs for each run for each
%           timepoint.
%
%     analysisType: An optional argument for the the type of analysis requested
%                   on the call to function partialcorr. The default value is
%                   'Spearman', which results in performing a PRCC analysis.
%                   If specified as 'Pearson' then a PCC analysis is performed.
%
% Outputs:
%     prcc: A TP x O x P matrix of the PRCC values for the timesteps and
%           model outputs specified in Y.
%
%     significance: A TP x O x P matrix of the significance values for the
%                   PRCC values in prcc.
%
%     pcc_ztest: A TP x O x P matrix of the Z-test values for the
%                timesteps and model outputs specified in Y.
%
%     pcc_ztest_signif: A TP x O x P matrix of the Z-test significance
%                       values for the PCC values in pcc_ztest.

arguments
    lhsMatrix
    data
    analysisType = 'Spearman'
end

if ~strcmp(analysisType, 'Spearman') && ~strcmp(analysisType, 'Pearson')
    ident = 'PERFORM:PRCC:ERROR:invalidAnalysisType';
    fmt = 'The analysis type of %s is invalid. It must be either Spearman or Pearson.';
    msg = sprintf(fmt, analysisType);
    exception = MException(ident, msg);
    throw(exception);
end


[R, P] = size(lhsMatrix);
[O, dataR, TP] = size(data);

if dataR ~= R
    ident = 'PERFORM:PRCC:ERROR:invalidRunCount';
    fmt = 'The number of runs (rows) from the LHS matrix, %d, ';
    fmt = [fmt, ' does not match the number of runs from the data, %d.'];
    msg = sprintf(fmt, R, dataR);
    exception = MException(ident, msg);
    throw(exception);
end

% Allocate the result arrays.
prcc = zeros(TP, O, P);
significance = zeros(TP, O, P);
pcc_ztest = zeros(TP, O, P);
pcc_ztest_signif = zeros(TP, O, P);

% Process each model output
for i = 1:O

    % Y is R x TP.
    %
    % Extract a model output from the data array. This has to be done
    % differently depending on the timesteps.
    if TP == 1
        % One timestep. data(i, :, :) is 1xR. It needs to be transposed to Rx1
        % for passing to the partialcorr function. squeeze(data(i, :, :)) is
        % also 1xR, so squeeze is not useful in this case.
        Y = data(i, :, :)';
    else
        % More than 1 timestep. data(i, :, :) is 1xRxTP. squeeze makes it
        % RxTP as required by the partialcorr function.
	    Y = squeeze(data(i, :, :));
    end

    % Process each varied parameter and each varied initial condition.
    for j = 1:P
    
        % Z is the LHS matrix of varied parameter values with the column
        % for varied parameter j removed.
        Z = lhsMatrix;
        Z(:, j) = [];
    
        % Calculate PRCCs and significances. lhsMatrix(:,j), Y, Z all must
        % have R rows.
        [rho, p] = partialcorr(lhsMatrix(:,j), Y, Z, 'type', analysisType);
    
        % Store the results for this varied parameter for this output.
        % rho and p are 1 x TP.
        prcc(:, i, j) = rho;
        significance(:, i, j) = p;

	% Calculate Z-test which needs PCC, not PRCC.
	[rho, ~] = partialcorr(lhsMatrix(:,j), Y, Z);
        % Equation 8 in doi:10.1016/j.jtbi.2008.04.011
        fisher_transform = @(r) 0.5 * log(abs((1 + r) ./ (1 - r)));
        % Equation 10 in doi:10.1016/j.jtbi.2008.04.011
        [h, p] = ztest(fisher_transform(rho), 0, 1);
        pcc_ztest(:, i, j) = h;
        pcc_ztest_signif(:, i, j) = p;
    end

end

end % function prcc










