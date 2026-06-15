function prccResult = performPrccWithCorrections(prccResult, analysisType)

% Perform PRCC analysis, including corrections.
%
% In comments below:
%
% TP is the number of timepoints.
%
% R is number of runs, same as NR is nte settings file.
%
% P is the number of varied parameters and varied initial conditions.
%
% O is the number of of model outputs, the same as the number of model
%   equations.
%
% Inputs:
%
%     prccResult: A structure with the following fields:
%
%         analysisTimePoints: A TP x 1 vector of timestep values.
%
%         lhsMatrix: An R x P matrix of varied parameter and varied initial
%                    condition values.
%
%         alpha: The value used to determine significance. If the significance
%                value for a prcc value is < alpha then that prcc value is
%                significant.
%
%         modelOutput: A O x R x TP matrix of model output values.
%
%         modelOutputNames: An O x 1 vector of data column names.
%
%         paramNames: A P x 1 vector of parameter and ivaried initial conditions.
%
%     analysisType: An optional argument for the the type of analysis requested
%                   on the call to function partialcorr. The default value is
%                   'Spearman', which results in performing a PRCC analysis.
%                   If specified as 'Pearson' then a PCC analysis is performed.
%
% Outputs:
%
%     prccResult has had the following fields added:
%
%     uncorrectedPrcc: A TP x O x P matrix of the PRCC values.
%
%     uncorrectedSignificance: A TP x O x P matrix of the significance values
%                              for the PRCC values in uncorrectedPrcc.
%
%     uncorrectedSignificantPrcc: A TP x O x P matrix of the PRCC values.
%                                 The same as uncorrectedPrcc, except entries
%                                 for PRCC values that are not significant have
%                                 been set to 1.
%
%     uncorrectedSignificantSignifcance: A TP x O x P matrix. The same as
%                                        uncorrectedSignificance, except that
%                                        significance values for PRCC values
%                                        that are not significant have been
%                                        set to 1.
%
%     uncorrectedPccZtest: A TP x O x P matrix of integer values indicating
%                          whether uncorrectedPccZtestSignificance p-values
%                          are lower than 0.05.  To use a significance
%                          level other than 0.05, threshold the values of
%                          uncorrectedPccZtestSignificance.
%
%     uncorrectedPccZtestSignificance: A TP x O x P matrix of Z-test p-values
%                                      using equation 10 in Marino 2008
%                                      doi:10.1016/j.jtbi.2008.04.011
%
%     bonferroniSignificance: A TP x O x P matrix of signficance values
%                             corrected using the Bonferroni correction.
%
%     bonferroniSignificantPrcc: A TP x O x P matrix of PRCC values.
%                                The same as uncorrectedPrcc, except entries
%                                for PRCC values that are not significant 
%                                according to alpha and bonferroniSignificance
%                                have been set to 1.
%
%     bonferroniSignificantSignifcance: A TP x O x P matrix. The same as
%                                       bonferroniSignificance, except that
%                                       bonferroniSignificance values that
%                                       are not significant have been set to 1.
%
%     bhfdrSignificance: A TP x O x P matrix of signficance values
%                        corrected using the BHFDR (Benjamini-Hochberg)
%                        correction.
%
%     bhfdrSignificantPrcc: A TP x O x P matrix of PRCC values.
%                           The same as uncorrectedPrcc, except entries
%                           for PRCC values that are not significant 
%                           according to alpha and bhfdrSignificance
%                           have been set to 1.
%
%     bhfdrSignificantSignifcance: A TP x O x P matrix. The same as
%                                  bhfdrSignificance, except that
%                                  bhfdrSignificance values that are not
%                                  significant have been set to 1.

arguments
    prccResult
    analysisType = 'Spearman'
end

if ~strcmp(analysisType, 'Spearman') && ~strcmp(analysisType, 'Pearson')
    ident = 'PERFORM:PRCC:ERROR:invalidAnalysisType';
    fmt = 'The analysis type of %s is invalid. It must be either Spearman or Pearson.';
    msg = sprintf(fmt, analysisType);
    exception = MException(ident, msg);
    throw(exception);
end


prccResult = performUncorrectedPrcc(prccResult, analysisType);
prccResult = performBonferroniCorrections(prccResult);
prccResult = performBhfdrCorrections(prccResult);

end %function performPrccWithCorrections

function prccResult = performUncorrectedPrcc(prccResult, analysisType)

% Perform the uncorrected PRCC analysis.
%
% Then define which PRCC values and significance
% values are significant according to the alpha value.
%
% Then perform PRCC corrections, including significance for the corrected PRCC
% values.
%
% Inputs:
%
%     prccresult: A structure with the following fields.
%
%         lhsMatrix: An R x P matrix of varied parameter and varied initial
%                    condition values.
%
%         modelOutput: An O x R x TP matrix of data (model output) values.
%
%         alpha: The value used to determine significance. If the significance
%                value for a prcc value is < alpha then that prcc value is
%                significant.
%
%    analysisType: The type of analysis to perform, either 'Spearman' or 'Pearson'
%
% Outputs:
%
%     prccResult: The results of the uncorrected PRCC have been added as
%                 the following fields to this structure.
%
%     uncorrectedPrcc: A TP x O x P matrix of the PRCC values.
%
%     uncorrectedSignificance: A TP x O x P matrix of the significance values
%                              for the PRCC values in uncorrectedPrcc.
%
%     uncorrectedSignificantPrcc: A TP x O x P matrix of the PRCC values.
%                                 The same as uncorrectedPrcc, except entries
%                                 for PRCC values that are not significant have
%                                 been set to 1.
%
%     uncorrectedSignificantSignifcance: A TP x O x P matrix. The same as
%                                        uncorrectedSignificance, except that
%                                        significance values for PRCC values
%                                        that are not significant have been
%                                        set to 1.
%
%       uncorrectedPccZtest: A TP x O x P matrix of integer values indicating
%                            whether uncorrectedPccZtestSignificance p-values
%                            are lower than 0.05.  To use a significance
%                            level other than 0.05, threshold the values of
%                            uncorrectedPccZtestSignificance.
%
%       uncorrectedPccZtestSignificance: A TP x O x P matrix of Z-test p-values
%                                        using equation 10 in Marino 2008
%                                        doi:10.1016/j.jtbi.2008.04.011
%

[prccResult.uncorrectedPrcc, ...
    prccResult.uncorrectedSignificance, ...
    prccResult.uncorrectedPccZtest, ...
    prccResult.uncorrectedPccZtestSignificance] = ...
    performPrcc(prccResult.lhsMatrix, prccResult.modelOutput, analysisType);

[prccResult.uncorrectedSignificantPrcc, prccResult.uncorrectedSignificantSignificance] = ...
    determineSignificance(prccResult.uncorrectedPrcc, prccResult.uncorrectedSignificance, prccResult.alpha);

end % function performUncorrectedPrcc

function prccResult = performBonferroniCorrections(prccResult)

% Perform the Bonferroni PRCC corrections.
%
% Adjust the uncorrected significance values for each timepoint using the
% Bonferroni correction. Then define Bonferroni correctd PRCC values using
% alpha and the corrected significance values.
%
% Inputs:
%
%     prccResult which has the following fields (among others not used here):
%
%         analysisTimePoints: A vector of timepoints for PRCC analysis.
%
%         uncorrectedPrcc: A TP x O x P matrix of uncorrected PRCC values.
%
%         uncorrectedSignificance: A TP x O x P matrix of the significance
%                                  values for the PRCC values in uncorrectedPrcc.
%
%         alpha: The alpha value to use as the cutoff for significant PRCCs.
%
% Outputs:
%
%    prccResult: Updated with the following new fields defined.
%
%        bonferroniSignificance: A TP x O x P matrix of signficance values
%                                corrected using the Bonferroni correction.
%
%        bonferroniSignificantPrcc: A TP x O x P matrix of PRCC values.
%                                   The same as uncorrectedPrcc, except entries
%                                   for PRCC values that are not significant 
%                                   according to alpha and bonferroniSignificance
%                                   have been set to 1.
%
%        bonferroniSignificantSignifcance: A TP x O x P matrix. The same as
%                                          bonferroniSignificance, except that
%                                          bonferroniSignificance values that
%                                          are not significant have been set to 1.

TP = size(prccResult.analysisTimePoints, 1);

prccResult.bonferroniSignificance = zeros(size(prccResult.uncorrectedSignificance));
for i = 1:TP
    uncorrectedSignificance = squeeze(prccResult.uncorrectedSignificance(i,:,:));
    prccResult.bonferroniSignificance(i,:,:) = bonferroni_correction(uncorrectedSignificance);
end

[prccResult.bonferroniSignificantPrcc, prccResult.bonferroniSignificantSignifcance] = ...
        determineSignificance(prccResult.uncorrectedPrcc, prccResult.bonferroniSignificance, prccResult.alpha);

end %function performBonferroniCorrections

function prccResult = performBhfdrCorrections(prccResult)

% Perform the BHFDR (Benjamini-Hochberg False Data Recovery) correction for
% each timepoint.
%
% Inputs:
%
%     prccResult which has the following fields (among others not used here):
%
%         analysisTimePoints: A vector of timepoints for PRCC analysis.
%
%         uncorrectedPrcc: A TP x O x P matrix of uncorrected PRCC values.
%
%         uncorrectedSignificance: A TP x O x P matrix of the significance
%         values for the PRCC values in uncorrectedPrcc.
%
%         alpha: The alpha value to use as the cutoff for significant PRCCs.
%
%         paramNames is a 1xP cell array of strings. Used for debug print.
%
%         modelOutputNames is an O x 1 cell array of strings. Used for debug
%         print. Used for debug print.
%
% Outputs:
%
%    prccResult: Updated with the following new fields defined.
%
%        bhfdrSignificance: A TP x O x P matrix of signficance values
%        corrected using the BHFDR correction.
%
%        bhfdrSignificantPrcc: A TP x O x P matrix that is the same as
%        uncorrectedPrcc, except when an uncorrected prcc value is not
%        significant, according to the BHFDR test, that element of
%        bhfdr_significant_prcc is 1.
%
%        bhfdrSignificantSignifcance: A TP x O x P matrix that is the same as
%        bhfdrSignificance, except when a significance value is not
%        significant, according to the BHFDR test, that element of
%        bhfdrSignificantSignifcance is 1.

TP = size(prccResult.analysisTimePoints, 1);
resultMatrixSize = size(prccResult.uncorrectedSignificance);

prccResult.bhfdrSignificance = zeros(resultMatrixSize);
prccResult.bhfdrSignificantPrcc = zeros(resultMatrixSize);
prccResult.bhfdrSignificantSignificance = zeros(resultMatrixSize);
for i = 1:TP
    uncorrectedPrcc = squeeze(prccResult.uncorrectedPrcc(i,:,:));
    uncorrectedSignificance= squeeze(prccResult.uncorrectedSignificance(i,:,:));

    [prccResult.bhfdrSignificance(i,:,:), ...
     prccResult.bhfdrSignificantPrcc(i,:,:), ...
     prccResult.bhfdrSignificantSignificance(i,:,:)] = bhfdr_correction( ...
                                              uncorrectedPrcc, ...
                                              uncorrectedSignificance, ...
                                              prccResult.alpha, ...
                                              prccResult.paramNames, ...
                                              prccResult.modelOutputNames);
                                              
end

end %function performBhfdrCorrections
