function result = prcc_analysis(...
    odeHostSim, granOutputIdx, lnbOutputIdx, ...
    timesteps, onlyInitialGrans, ...
    granFilter, alpha, analysisType, varargin)

% Perform PRCC analysis of a HostSim LHS.
%
% In the comments below:
%
% G: The number of granulomas across all hosts 
%
% R: The total number of HostSim runs.
%
% IG: The number of primary granulomas across all hosts; i.e. number of 
%     initial grans * the number of hosts
%
% P: The number of varying granuloma parameters + number of varying lymph/blood
%    parameters.onlyInitialGrans
%
% O: The number of model granuloma outputs to include (the length of
%    granOutputIdx) + the number of lymph/blood outputs to include (the length
%    of lnbOutputIdx) + all additional outcome measurements in varargin (LN + Blood)
%
% gO: The number of model granuloma outputs to include (the length of
%    granOutputIdx) + additional granuloma outcomes in varargin.
%
% lnO: The number of lymph/blood outputs to include (the length
%    of lnbOutputIdx) + additional LN/Blood outcomes in varargin.
%
% TP: The number of timepoints being analyzed.

% Inputs:
%
% odeHostSim: A structure that is the result of a HostSim LHS.
%
% granOutputIdx: A row vector of indices into the lungOutputFinal matrix of each host.
%
% lnbOutputIdx: A row vector of indices into the LNOutputFinal matrix of each host.
%
% timesteps: The timesteps to perform the analysis for, a row vector of timestep
%            values. These should be members of odeHostSim.analysisTimePoints.
%
% onlyInitialGrans: A logical value. If true then only include data for the
%                   initial granulomas of each host, otherwise include data
%                   for all granulomas of each host.
%
% granFilter: If non-empty a H x IG boolean array, where H is the number of
%             hosts and IG is the number of initial granulomas. It is 1 for
%             an initial granuloma to be included  and is 0 for an
%             initial granuloma to exclude. If non-empty then
%             onlyInitialGrans is considered true (regardless of the
%             onlyInitialGrans argument value). If empty then granulomas
%             are selected for analysis without regard to any filtering
%             and onlyInitialGrans can be true or false.
%
% alpha: The alpha value to use as the cutoff for significant PRCCs.
%
% analysisType: "Spearman" or "Pearson".  Enter "Spearman" for PRCC, "Pearson" for PCC.
%
% Either granOutputIdx or lnbOutputIdx may be empty, but not both.
%
% Optional arguments: 
%    Non-state variable outcomes at different scales.
%    prcc_analysis(..., '<Scale>', '<Type>', {'<metricName1>',...,'<metricNameN>'})
%    Allows one to add other outcomes from LN/Gran objects to analyze, instead of
%    just state variables.  
%
%         Valid inputs for <Scale> are 'LNB' and 'gran'. 
%         Valid inputs for '<Type>' at time of writing are:
%               gran: 'downstream' or 'geometry'
%               LNB: 'downstream'
%         Valid inputs for <metricName#> are any subfield of
%         ...hostList{i}.<Gran/LNB>CellArray{1}.(<Type>), 
%           e.g. if ...GranCellArray{1}.downstream.totalBugs exists
%           then 'totalBugs' is valid for '<metricName#>'. 
%
% Example Call: 
% >result = prcc_analysis(odeHostSim, [15,16,17],[], 1:200, true, [], 0.05, ...
%           'gran','downstream',{'totalBugs'},...
%           'gran','geometry',{'diamGran','caseumRatio'},...
%           'LNB','downstream',{'petCTPrediction'})
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs: Summary: results structure containing 
%          results.gran and results.host, each containing their own
%          PRCCs and LHS matrices. Results also contains various fields
%          containing information common to both gran/lnb.
%
% result: A structure with the following fields:
%
%     lhsMatrix: A matrix of the varying parameter values, of size G x P. Each
%                granuloma from the same host has the same lymph/blood varying
%                parameters.
% 
%     outputs: A matrix of size G x O, where G is the number of granulomas across
%                  all hosts and  Each granuloma from the same
%                  host hasthe same lymph/blood outputs.
%
%     analysisTimePoints: a column vector of time points to be analyzed;
%         equal to timesteps (transpose)
%
%     granFieldNames, LNBFieldNames, extraOutFieldTypesGran, extraOutFieldTypesLNB, 
%         which contain <Type> and <metricName#> in a ragged cell matrix structure;
%         as well as totalExtraOutputFields(Gran/LNB) which contain the total number
%         of <metricName#> optional variables passed in at both gran and LNB scales.
%
%     granOutputNames, lnbOutputNames, allOutputNames - cell arrays of strings
%         containing names of all output names, both state variable and non-
%         state-variable names. 
%
%     paramNames: A cell array of size 1 x P of the names of the granuloma and
%             lymph/blood varying parameters.  
%
%     granOutputIdx, lnbOutputIdx, onlyInitialGrans, alpha
%        - Stored in the result structure directly from function arguments
%
%     gran - A structure which contains the PRCC results at the 
%           granuloma scale. It includes the fields:
%
%         alpha, paramNames, lhsMatrix, analysisTimePoints
%         - Each inherited from results as above.  
%
%         modelOutputNames - Names of granuloma outcomes; equal to
%                           result.granOutputNames for compatibility.
%
%         modelOutput - A matrix of size gO x G x TP containing granuloma outputs
%                    for each granuloma in time. note that if onlyInitialGrans
%                    was chosen, this will be of size gO x IG x TP
%    
%         uncorrectedPrcc: A gO x P matrix that is the PRCC values from the PRCC
%                          analysis.
%    
%         uncorrectedSignificance: An O x P matrix that is the significance values for
%                                  the PRCC values.
%
%         uncorrectedSignificantPrcc: A copy of uncorrectedPrcc, except any element
%                                     that is not significant according to the alpha
%                                     test, comparing uncorrectedSignificance to
%                                     alpha, is NaN.
%    
%         uncorrectedSignificantSignifcance: A copy of uncorrectedPrcc, except any
%                                            element that is not significant
%                                            according to the alpha test is NaN.
%    
%         bonferroniSignificance: A version of uncorrectedSignificance, after the
%                                 Bonferroni corrections are performed.
%    
%         bonferroniSignificantPrcc: Analogous to uncorrectedSignificantPrcc, but
%                                    based on an alpha test using
%                                    bonferroniSignificance.
%    
%         bonferroniSignificantSignifcance: Analagous to uncorrectedSignificantSignifcance,
%                                           but based on a alpha test using
%                                           bonferroniSignificance.
%    
%         bhfdrSignificance: A version of uncorrectedSignificance, after the
%                                 Benjamini-Hochberg False Data recovery
%                                 corrections are performed.
%    
%         bhfdrSignificantPrcc: Analogous to uncorrectedSignificantPrcc, but
%                                    based on an alpha test using
%                                    bhfdrSignificance.
%    
%         bhfdrSignificantSignifcance: Analagous to uncorrectedSignificantSignifcance,
%             but based on a alpha test using
%             bhfdrSignificance.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  result also contains the substructure:
%
%       host - a structure analogous to result.gran, with the following changes: 
%
%       result.host.lhsMatrix is of size R x P, taken from
%                 result.gran.lhsMatrix to ensure no copied LN parameter values.
%                 This is concatenated with the hostGran values.
%
%       modelOutput - a matrix of size lnO x R x TP containing lymph node / blood
%                    outputs for each lymph node in time.
%
%       Otherwise, lnb has the same fields as gran, including:
%           modelOutputNames, alpha, paramNames, 
%           uncorrectedPrcc, uncorrectedSignificance, 
%           uncorrectedSignificantPrcc, uncorrectedSignificantSignifcance,
%           bonferroniSignificance, bonferroniSignificantPrcc,
%           bonferroniSignificantSignifcance, bhfdrSignificance,
%           bhfdrSignificantPrcc, bhfdrSignificantSignifcance
%           
%           all calculated the same way / analogously to gran, only using
%           its own modelOutput and LHS matrix.
%


% Validate variable inputs
fprintf('Checking inputs.\n')
[gran_tsIdx, lnb_tsIdx, errorCount] = checkInputs(odeHostSim, granOutputIdx, lnbOutputIdx, ...
    timesteps, onlyInitialGrans, alpha);
[~, TP] = size(timesteps);

if (errorCount > 0)
    fprintf("%d Error(s) in checkInputs. Aborting...", errorCount);
    result = [];
    return;
end

% Set up result structure by making result.<fields>, result.gran, and result.host
result = setupResultStruct(odeHostSim, granOutputIdx, lnbOutputIdx, ...
    timesteps, onlyInitialGrans, ...
    granFilter, alpha, varargin);

if (isa(result, 'char'))
    fprintf("Error(s) encountered after setupResultStruct.\n");
    fprintf(result);
    fprintf("Aborting...\n");
    return
end

% Get the matrix of the model outputs on both the LNB and Granuloma scales
[result.gran.modelOutput, result.host.modelOutput, result.granPK.modelOutput, ...
    result.hostGranPK.modelOutput, result.PLPK.modelOutput] = ...
    getOutputs(odeHostSim, granOutputIdx, ...
    lnbOutputIdx, TP, gran_tsIdx, lnb_tsIdx, onlyInitialGrans, granFilter,...
    result.extraOutFieldTypeGran, result.granFieldNames, ...
    result.totalExtraOutputFieldsGran,...
    result.extraOutFieldTypeLNB, result.LNBFieldNames, ...
    result.totalExtraOutputFieldsLNB ...
    );

% Perform gran-scale PRCC on the gran substructure with the full LHS matrix
result.gran = performPrccWithCorrections(result.gran, analysisType);

% Perform host-scale PRCC on the host substructure with the host-scale LHS matrix
result.host = performPrccWithCorrections(result.host, analysisType);

% if (odeHostSim.clusterSampling)
%     result.hostGran = performPrccWithCorrections(result.hostGran);
% end

%Drug PRCC
if (odeHostSim.drug)
    %result.granPK = performPrccWithCorrections(result.granPK);
    %result.hostGranPK = performPrccWithCorrections(result.hostGranPK);
    %result.PLPK = performPrccWithCorrections(result.PLPK);
end

end % prcc_analysis

function [gran_tsIdx, lnb_tsIdx, errorCount] = checkInputs(odeHostSim, granOutputIdx, ...
    lnbOutputIdx, timesteps, ...
    onlyInitialGrans, alpha)

% Check the user supplied inputs for validity.
%
% odeHostSim: A structure that is the result of a HostSim LHS.
%
% granOutputIdx: A vector of indices into the lungOutputFinal matrix of each host.
%
% lnbOutputIdx: A vector of indices into the LNOutputFinal matrix of each host.
%
% timesteps:  A vector of timesteps to perform the analysis for.
%
% onlyInitialGrans: A logical value. If true then only include data for the
%                   initial granulomas of each host, otherwise include data
%                   for all granulomas of each host.
%
% alpha: The alpha value to use as the cutoff for significant PRCCs.
%
% Outputs:
%
% gran_tsIdx: a vector of indices into odehostsim.analysistimepoints. it has the
%        same number of elements as timesteps. It is used to subset outputs
%        of granuloma.
%
% lnb_tsIdx: a vector of indices into odehostsim.analysistimepoints. it has the
%        same number of elements as timesteps. It is used to subset outputs from
%        the LN and blood. It is shifted by 1 because outputs in LNB are indexed
%        from timestep 1, not 0.
%
% errorCount: The number of input errors detected.

errorCount = 0;

if isempty(odeHostSim.hostList)
    errorCount = errorCount + 1;
    fprintf('The supplied odeHostSim has an empty hostList\n')
    % No point in continuing if the hostList is empty.
    return
end

if isempty(granOutputIdx) && isempty(lnbOutputIdx)
    errorCount = errorCount + 1;
    errmsg = 'Both the granuloma output indices';
    errmsg = [errmsg 'and lymph/blood output indices are empty.\n'];
    fprintf(errmsg)
    % No point in continuing if both granOutputIdx and lnbOutputIdx are
    % empty.
    return
end

[~, ~, numGranOutputs] = size(odeHostSim.hostList{1, 1}.lungOutputFinal);
errorCount = checkOutputIndices(numGranOutputs, granOutputIdx, 'granOutputIdx', errorCount);

[~, numLNOutputs] = size(odeHostSim.hostList{1, 1}.LNOutputFinal);
errorCount = checkOutputIndices(numLNOutputs, lnbOutputIdx, 'lnbOutputIdx', errorCount);


timestepError = false;
if ~isnumeric(timesteps)
    errorCount = errorCount + 1;
    timestepError = true;
    fprintf('The supplied timesteps is not numeric.\n')
end

if ~timestepError && ndims(timesteps) ~= 2 %#ok<ISMAT> 
    errorCount = errorCount + 1;
    timestepError = true;
    fprintf('The supplied timesteps has %d dimensions, instead of 2 dimensions\n', ndims(timesteps))
end

[rows, cols] = size(timesteps);
if ~timestepError && rows ~= 1
    errorCount = errorCount + 1;
    timestepError = true;
    fprintf('The supplied timesteps is not a row vector, it has %d rows\n', rows)
end

if ~timestepError 
    gran_tsIdx = [];
    lnb_tsIdx = [];
    badTimesteps = [];
    for i=1:cols
        [isin, idx] = ismember(timesteps(i), odeHostSim.analysisTimePoints);
        if isin
            gran_tsIdx(end+1) = idx; %#ok<AGROW> 
            lnb_tsIdx(end+1) = idx; %#ok<AGROW>
        else
            badTimesteps(end+1) = timesteps(i); %#ok<AGROW> 
        end
    end

    if ~isempty(badTimesteps)
        errorCount = errorCount + 1;
        fprintf('The following timesteps are not a member of odeHostSim.analysisTimePoints.\n')
        fprintf('%f ', badTimesteps)
        fprintf('\n')
    end
end

if ~islogical(onlyInitialGrans)
    errorCount = errorCount + 1;
    fprintf('The supplied onlyInitialGrans is not a a logical (true/false) value.\n')
end

alphaError = false;
if ~isscalar(alpha)
    errorCount = errorCount + 1;
    alphaError = true;
    fprintf('The supplied alpha is not a scalar.\n')
end

if ~alphaError && ~isnumeric(alpha)
    errorCount = errorCount + 1;
    alphaError = true;
    fprintf('The supplied alpha is not numeric.\n')
end

if ~alphaError && alpha <= 0
    errorCount = errorCount + 1;
    fprintf('The supplied alpha, %f, is <= 0.\n', alpha)
end

end % checkInputs

function errorCount = checkOutputIndices(numOutputs, outputIndices, name, errorCount)

% Check a vector of indices into an output array for being in bounds of the
% array.  This can be for either the granuloma output indices or the
% lymph/blood output indices.

% Inputs:
%
% numOutputs: The size of the array dimension for model outputs in the array
%             the output indices are to be used for.
%
% outputIndices: The output indices to check.
%
% name: A string name for the output indices, used for error messages.
%
% errorCount: The current number of input errors detected.
%
% Outputs:
%
% errorCount: The number of errors detected, updated for any errors detected by
%             this function.
%

if ~isnumeric(outputIndices)
    errorCount = errorCount + 1;
    fprintf('The supplied %s is not numeric.\n', name)
    return
end

[rows, ~] = size(outputIndices);
if rows > 1
    errorCount = errorCount + 1;
    fprintf('The supplied %s has more than 1 row.\n', name)
    fprintf('It should be a single number or a row vector of numbers\n')
    return
end

ints = outputIndices(floor(outputIndices) == outputIndices);
if length(ints) ~= length(outputIndices)
    errorCount = errorCount + 1;
    fprintf('The supplied %s is not all integer values.\n', name)
    return
end

numBad = sum(outputIndices <= 0);
if numBad > 0
    errorCount = errorCount + 1;
    fprintf('%d elements of %s are <= 0\n', numBad, name);
end

numBad = sum(outputIndices > numOutputs);
if numBad > 0
    errorCount = errorCount + 1;
    fmt = '%d elements of %s are greater than the number of';
    fmt = [fmt, ' outputs, %d\n'];
    fprintf(fmt, numBad, name, numOutputs)
end

end % checkOutputIndices

function lhsMatrix = getParams(odeHostSim, onlyInitialGrans, granFilter, hostGranMatrix)

% Create an LHS matrix of values of the varying parameters.

% Inputs:
%
% odeHostSim: A structure that is the result of a HostSim LHS.
%
% onlyInitialGrans: A logical value. If true then only include data for the
%                   initial granulomas of each host, otherwise include data
%                   for all granulomas of each host.
%
% granFilter: If non-empty a H x IG boolean array, where H is the number of
%             hosts and IG is the number of initial granulomas. It is 1 for
%             an initial granuloma to be included  and is 0 for an
%             initial granuloma to exclude. If non-empty then
%             onlyInitialGrans is considered true (regardless of the
%             onlyInitialGrans argument value). If empty then granulomas
%             are selected for analysis without regard to any filtering
%             and onlyInitialGrans can be true or false.
%
% hostGranMatrix: if true, will use hostGran LHS matrix rather than
%             granuloma LHS values matrix.
%
% Outputs:
%
% lhsMatrix: A matrix of the varying parameter values, of size G x P, where G is
%            the number of granulomas across all hosts and P is the number of
%            varying granuloma parameters + number of varying lymph/blood
%            parameters. Each granuloma from the same host has the same
%            lymph/blood varying parameters.

lhsMatrix = [];
[granParamCount, ~] = size(odeHostSim.granParamInfo);
[lnbParamCount, ~] = size(odeHostSim.lnbParamInfo);
% For backwards compatibility, a message is printed to screen
% This prevents the same message from being printed to screen
% repeatedly.
LNPrintbit = true;
granPrintbit = true;
for i=1:odeHostSim.NR

    % Get the lymph node varying parameter values for this host.
    lnbParams = [];
    for k=1:lnbParamCount
        if odeHostSim.lnbParamInfo{k,1}.varied
            paramName = odeHostSim.lnbParamInfo{k,1}.name;
            try 
                if (isfield(odeHostSim.hostList{i,1}.LNBloodCellArray{1,1}.initialConditions, paramName))
                    paramValue = odeHostSim.hostList{i,1}.LNBloodCellArray{1,1}.initialConditions.(paramName);
                else
                    paramValue = odeHostSim.hostList{i,1}.LNBloodCellArray{1,1}.(paramName);
                end
            catch 
                % In case you try to analyze old runs that didn't have the initialConditions field.
                if (LNprintbit)
                    fprintf("Field initialConditions does not exist in LN agent class.\n");
                    fprintf("Defaulting to old behavior...\n");
                    LNprintbit = false;
                end
                paramValue = odeHostSim.hostList{i,1}.LNBloodCellArray{1,1}.(paramName);
            end
            lnbParams = [lnbParams, paramValue]; %#ok<AGROW> 
        end
    end 

    if (~hostGranMatrix)
        % Process the granulomas for this host. Quit after processing the
        % initial grans, if that was requested. If requested only use a
        % granuloma that passes the T cell crash test.
        [hostGranCount, ~, ~] = size(odeHostSim.hostList{i,1}.lungOutputFinal);
        for j=1:hostGranCount

            if onlyInitialGrans && j > odeHostSim.numGransStart
                % Only doing initial grans and past the last initial gran.
                break
            elseif ~isempty(granFilter) && granFilter(i, j) == 0
                % Skip this gran because it didn't pass gran filer.
                continue
            end

            granParams = [];
            for k=1:granParamCount        
                if odeHostSim.granParamInfo{k,1}.varied
                    paramName = odeHostSim.granParamInfo{k,1}.name;
                    try 
                        if (isfield(odeHostSim.hostList{i,1}.GranCellArray{j}.initialConditions, paramName))
                            paramValue = odeHostSim.hostList{i,1}.GranCellArray{j}.initialConditions.(paramName);
                        else
                            paramValue = odeHostSim.hostList{i,1}.GranCellArray{j}.(paramName); 
                        end
                    catch 
                        % In case you try to analyze old runs that didn't have the initialConditions field.
                        if (granPrintbit) 
                            fprintf("Field initialConditions does not exist in granuloma agent class.\n");
                            fprintf("Defaulting to old behavior...\n");
                            granPrintbit = false;
                        end
                        paramValue = odeHostSim.hostList{i,1}.GranCellArray{j}.(paramName); 
                    end
                    granParams = [granParams, paramValue]; %#ok<AGROW> 
                end
            end 

            % Each granuloma in this host has its own granuloma paramters but they
            % all share the same lymph/blood parameters.

            %CM - copying the lnbParams here hostGranCount times; so we're
            %aware.
            lhsMatrix = [lhsMatrix; lnbParams, granParams]; %#ok<AGROW> 
        end
    else % hostGranMatrix is true
        % Process average granulomas for the host scale.
        for j = 1:odeHostSim.numGransStart
            granParams = [];
            for k=1:granParamCount        
                if odeHostSim.granParamInfo{k,1}.varied
                    paramName = odeHostSim.granParamInfo{k,1}.name;
                    paramValue = odeHostSim.lhsvaluesHostGran(k,i);
                    granParams = [granParams, paramValue]; %#ok<AGROW> 
                end
            end 

            % Each granuloma in this host has its own granuloma paramters but they
            % all share the same lymph/blood parameters.

            %CM - copying the lnbParams here nuGranStart times; so we're
            %aware. We're going to subsample this out. This can be simplified
            %at some point, but it takes nearly no time to subsample.
            lhsMatrix = [lhsMatrix; lnbParams, granParams]; %#ok<AGROW> 
        end
    end
end

end % getParams

function [granOutputs, hostOutputs, granPKOutputs, ...
    hostGranPKOutputs, PLPKOutputs] = ...
    getOutputs(odeHostSim, granOutputIdx, ...
    lnbOutputIdx, TP, gran_tsIdx, lnb_tsIdx, onlyInitialGrans, granFilter,...
    extraOutFieldTypeGran, granFieldNames, totalExtraOutputFieldsGran,...
    extraOutFieldTypeLNB, LNBFieldNames, totalExtraOutputFieldsLNB...
    )

% Create a matrix of granuloma and lymph/blood outputs, for the specified
% timestep, across all granulomas of all hosts, with only those granuloma
% outputs specified by granOutputIdx and only those lymph/blood outputs
% specified by lnbOutputIdx.
% ----------------------------
% gO - number of granuloma outputs
% lnO - number of lymph node and blood compartment outputs
% R - the number of HostSim runs
% G - the total number of granulomas across all HostSim runs
% TP - The number of timepoints being analyzed.

% Inputs:
%
% odeHostSim: A structure that is the result of a HostSim LHS.
%
% granOutputIdx: A vector of indices into the lungOutputFinal matrix of each
%                host.
%
% hostOutputIdx: A vector of indices into the LNOutputFinal matrix of each host.
%
% TP - The number of timepoints being analyzed.
%
% onlyInitialGrans: A logical value. If true then only include data for the
%                   initial granulomas of each host, otherwise include data
%                   for all granulomas of each host.
%
% granFilter: If non-empty a H x IG boolean array, where H is the number of
%             hosts and IG is the number of initial granulomas. It is 1 for
%             an initial granuloma to be included  and is 0 for an
%             initial granuloma to exclude. If non-empty then
%             onlyInitialGrans is considered true (regardless of the
%             onlyInitialGrans argument value). If empty then granulomas
%             are selected for analysis without regard to any filtering
%             and onlyInitialGrans can be true or false.
%
%             extraOutFieldType(Gran/LNB), (gran/LNB)FieldNames, totalExtraOutputFields(Gran/LNB)
%             - cell arrays additional output types and fieldnames to read out of odeHostSim for 
%               additional analysis. More details of these are in the header of prcc_analysis(...)
%
%
% Outputs:
%
%       granOutputs - A matrix of size gO x G x TP containing granuloma outputs
%                    1:gO for each granuloma in time.
%      
%       hostOutputs - a matrix of size lnO + gO x R x TP containing lymph node / blood
%                    outputs for each lymph node in time.

% odeHostSim.hostList{i, 1}.lungOutputFinal is numGrans X numTimesteps x
% numOutputs. numGrans might be >= odeHostSim.numGransStart, since
% dissemination may create additional grans.  Therefore it is possible that not
% all hosts have the same number of granulomas.

% Unpack variable inputs
% AnalyzeThis contains fieldnames for extras to be saved.

numBaseGranOutputs = numel(granOutputIdx);
numBaseLNBOutputs = numel(lnbOutputIdx);
granPKOutputs = [];
hostGranPKOutputs = [];
PLPKOutputs = [];

%Compute total number of grans...
if onlyInitialGrans
    numGransAllHosts = odeHostSim.numGransStart * odeHostSim.NR;
else
    numGransAllHosts = 0;
    for i = 1:odeHostSim.NR
        numGransAllHosts = numGransAllHosts + ...
            size(odeHostSim.hostList{i}.lungOutputFinal,1);
    end
end
% Preallocate output bin sizes...
granOutputs = zeros(numBaseGranOutputs + totalExtraOutputFieldsGran, ...
    numGransAllHosts, numel(gran_tsIdx));
hostGranOutputs = zeros(numBaseGranOutputs + totalExtraOutputFieldsGran, ...
    odeHostSim.NR, numel(gran_tsIdx));
% Used for linearizing ragged gran counts between hosts
hostGransProcessed = 0;
lnbOutputs = zeros(numBaseLNBOutputs + totalExtraOutputFieldsLNB,...
    odeHostSim.NR, numel(lnb_tsIdx));

%Iterate through all hosts to collect their outputs...
for i=1:odeHostSim.NR

    % Get the lymph node outputs for this host.
    lnbOutputs(1:numBaseLNBOutputs, i, :) = ...
        squeeze(odeHostSim.hostList{i,1}.LNOutputFinal(lnb_tsIdx, lnbOutputIdx))';

    % Add the extra LNB outputs for this host.

    % There are two dimensions for keeping track of the LN, blood, etc. 
    % First are types of outputs in  "extraOutFieldTypeLNB" - e.g. "downstream" vs. "geometry"
    % The second is the fieldname within that; e.g. "downstream.petCTPrediction".
    % To linearize the ragged matrix, we have "totalIdx".
    % Since the first N LNBoutputs come from state variables, we don't start at 0.
    totalIdx = numBaseLNBOutputs;

    if (numel(extraOutFieldTypeLNB) > 0)
        %if there are any extra outputs to add...
        for exLNIdx = 1:numel(extraOutFieldTypeLNB)
            %for each type of extra output...
            for fieldname = 1:numel(LNBFieldNames{exLNIdx})
                %for each metric within that field...
                totalIdx = totalIdx + 1;
                lnbOutputs(totalIdx, i, :) = ...
                    odeHostSim.hostList{i,1}.LNBloodCellArray{1}.(extraOutFieldTypeLNB{exLNIdx}).(LNBFieldNames{exLNIdx}{fieldname})(lnb_tsIdx);
            end
        end
    end

    % Get the requested model outputs for each granuloma for the specified
    % timestep.
    [hostGranCount, ~, ~] = size(odeHostSim.hostList{i,1}.lungOutputFinal);
    %Increment previous host gran count...
    for j = 1:hostGranCount
        %for each granuloma in this host...

        % Check whether this granuloma qualifies.
        if onlyInitialGrans && j > odeHostSim.numGransStart
            % Only doing initial grans and past the last initial gran.
            break
        elseif ~isempty(granFilter) && granFilter(i, j) == 0
            % Skip this gran because it didn't pass the T cell crash test.
            continue
        end
        thisGran = hostGransProcessed + 1;

        % Add the state variable granuloma outputs to granOutputs
        for baseGranIdx = 1:numBaseGranOutputs
            granOutputs(baseGranIdx, thisGran, :) = odeHostSim.hostList{i,1}.lungOutputFinal(j, gran_tsIdx, granOutputIdx(baseGranIdx));
            hostGranOutputs(baseGranIdx, i, :) = hostGranOutputs(baseGranIdx, i, :) + granOutputs(baseGranIdx, thisGran, :);
        end

        % Add in the extra outputs...
        %granExtraOutputs = zeros(1, totalExtraOutputFieldsGran); 
        totalIdx = numBaseGranOutputs;
        for exGranIdx = 1:numel(extraOutFieldTypeGran)
            for fieldname = 1:numel(granFieldNames{exGranIdx})
                totalIdx = totalIdx + 1;
                granOutputs(totalIdx, thisGran, :) = ...
                    odeHostSim.hostList{i,1}.GranCellArray{j}.(extraOutFieldTypeGran{exGranIdx}).(granFieldNames{exGranIdx}{fieldname})(gran_tsIdx);
                hostGranOutputs(totalIdx, i, :) = hostGranOutputs(totalIdx, i, :) + granOutputs(totalIdx, thisGran, :);
            end
        end


        %Iterate the number of grans for the linearized indexing of grans.
        hostGransProcessed = hostGransProcessed + 1;
    end
end

hostOutputs = zeros( numBaseLNBOutputs + totalExtraOutputFieldsLNB + ...
    numBaseGranOutputs + totalExtraOutputFieldsGran, ...
    odeHostSim.NR, numel(gran_tsIdx));

if size(lnbOutputs,1) >= 1
    hostOutputs(1:size(lnbOutputs,1), :,:) ...
        = lnbOutputs;
end
if size(hostGranOutputs,1) >= 1
    hostOutputs(1 + size(lnbOutputs,1): size(lnbOutputs,1) + size(hostGranOutputs,1),:,:) ...
        = hostGranOutputs;
end

end % getOutputs

function [paramNames, granOutputNames, lnbOutputNames, allOutputNames]...
    = formatNames(odeHostSim, granOutputIdx, ...
    lnbOutputIdx, granExtraOutputNames,...
    LNExtraOutputNames)

% Create cell arrays of varying parameter names and model output names,
% for those model outputs selected.

%
% P - the number of varying granuloma parameters + number 
%     of varying lymph/blood parameters.
%
% gO - the number of model granuloma outputs to include 
%
% lnO - the number of lymph/blood outputs to include.
%%%%%%%%%%%%%%%%%
% Inputs:
%
% odeHostSim: A structure that is the result of a HostSim LHS run.
%
% granOutputIdx: A vector of indices into the lungOutputFinal matrix of each
%                host.
%
% lnbOutputIdx: A vector of indices into the LNOutputFinal matrix of each host.
%
% (gran/LN)ExtraOutputNames - Cell array of size 1 x <# types of extra
% output#>.  Each cell X contains a cell array of all field names that
% are contained by output type X; these are to be unpacked here and
% linearized into granOutputNames and lnbOutputNames respectively.
%
%%%%%%%%%%%%%%%%%%%
% Outputs:
%
% paramNames: A cell array of size P f the names of the granuloma and
%             lymph/blood varying parameters.  
% 
% granOutputNames: A cell array of size gO of the names of the selected granuloma
%              model outputs. 
%
% lnbOutputNames: A cell array vector of size lnO of the names of the selected
%              of the selected lymph/blood model outputs. 
%
% allOutputNames: A cell array of size lnO + gO of the names of the 
%              selected granuloma and lymph/blood model outputs.
%
%              State variable outputs have generic generated names,
%              since the model outputs are elements of GranulomaAgentClass 
%              and LNClass output vectors, and are not distinguished from other 
%              properties of those classes.
%                   

[granParamCount, ~] = size(odeHostSim.granParamInfo);
granParamNames = cell(0);
for k=1:granParamCount        
    if odeHostSim.granParamInfo{k,1}.varied
        paramName = odeHostSim.granParamInfo{k,1}.name;
        granParamNames{end+1} = paramName; %#ok<AGROW> 
    end
end 

lnbParamNames = cell(0);
[lnbParamCount, ~] = size(odeHostSim.lnbParamInfo);
for k=1:lnbParamCount
    if odeHostSim.lnbParamInfo{k,1}.varied
        paramName = odeHostSim.lnbParamInfo{k,1}.name;
        lnbParamNames{end+1} = strcat('lnb_', paramName); %#ok<AGROW> 
    end
end 

paramNames = [lnbParamNames, granParamNames];

%Preallocate strings arrays for the number of granuloma and lnb outputs
num_gran_sv_outputs = numel(granOutputIdx);
num_gran_extra_outputs = numel(granExtraOutputNames);
num_lnb_sv_outputs = numel(lnbOutputIdx);
num_lnb_extra_outputs = numel(LNExtraOutputNames);
granOutputNames = strings(1, num_gran_sv_outputs);
lnbOutputNames = strings(1, num_lnb_sv_outputs);

[SV_names, ~] = hostSim_Gran_State_Names();
for SV_subID = 1:numel(granOutputIdx);
    SV_ID = granOutputIdx(SV_subID);
    granOutputNames(SV_subID) = SV_names(SV_ID); %#ok<AGROW> 
end

[SV_names, ~] = hostSim_LNB_State_Names();
for SV_subID = 1:numel(lnbOutputIdx)
    SV_ID = lnbOutputIdx(SV_subID);
    lnbOutputNames{SV_ID} = SV_names(SV_ID); %#ok<AGROW> 
end

%Append the extra output names
for i_sub = 1:num_gran_extra_outputs
    i = i_sub + num_gran_sv_outputs;
    for j = 1:numel(granExtraOutputNames{i_sub})
        granOutputNames(i) = string(granExtraOutputNames{i_sub}{j}); %#ok<AGROW> 
    end
end
for i_sub = 1:num_lnb_extra_outputs
    i = i_sub + num_gran_sv_outputs;
    for j = 1:numel(LNExtraOutputNames{i_sub})
        lnbOutputNames(i) = string(LNExtraOutputNames{i_sub}{j}); %#ok<AGROW> 
    end
end

allOutputNames = [lnbOutputNames, granOutputNames];

end % formatNames

function [result] = setupResultStruct(odeHostSim, granOutputIdx, lnbOutputIdx, ...
    timesteps, onlyInitialGrans, granFilter, alpha, prcc_analysis_varargs)
% Defines and populates a results structure for PRCC analysis on gran and host scales.
%
% input: 
%      odeHostSim - contains output of a HostSim run
%
%      granOutputIdx, lnbOutputIdx, timesteps, onlyInitialGrans, granFilter, alpha,
%      - All exactly the same as the function prcc_analysis atop this file.
%
%      prcc_analysis_varargs - The cell array varargs that were passed into prcc_analysis.  
%
%
% output: 
%    result - a new structure containing the following fields:
%            
%          granOutputIdx, lnbOutputIdx, onlyInitialGrans, alpha
%                   -Same as input, now stored in result
%
%          analysisTimePoints - a vector of TP x 1 time points for analysis;
%                   Same as timepoints (transposed) for compatibility with
%                   external scripts.
%
%           paramNames, granOutputNames, lnbOutputNames, allOutputNames 
%           - stored in results; details are in the function "formatNames"
%             in this file.
%        
%           extraOutFieldTypeGran extraOutFieldTypeLNB granFieldNames LNBFieldNames 
%           - stored in results; details are in the function "unpackVarargs" 
%             in this file
%
%           lhsMatrix
%           - stored in results; details are in the function "getParams" 
%             in this file. Note that this is the granuloma-scale lhsMatrix.
%             host-scale lhs matrix is result.host.lhsMatrix%
%
%           gran - A structure which will contain the PRCC results at the 
%                 granuloma scale. it is set up here to include the fields:
%
%                    alpha, paramnames, lhsmatrix, analysistimepoints
%                    - each inherited from results as above.  
%
%                    modeloutputnames - copied from results.granoutputnames
%                    named this way in order to be compatible with external 
%                    scripts.
%           
%           host - A structure which will contain the PRCC results at the 
%                 host scale. It is set up here to include the fields:
%
%                    alpha, paramNames, analysisTimePoints
%                    - Each inherited from results as above.  
%
%                    lhsMatrix - This is the host-scale LHS matrix which
%                               contains only one LHS sampling of the 
%                               lymph node / blood parameters per row,
%                               instead of being copied for each granuloma.
%
%                    modelOutputNames - copied from results.lnbOutputNames
%                    Named this way in order to be compatible with external 
%                    scripts.

%Initialize the result structure.
result = struct();
%Unpack varargs - get names of extra outputs and store them in result.
result = unpackVarargs(odeHostSim, result, prcc_analysis_varargs);
if (isa(result, 'char'))
    return;
end
%Define which components of lungOutputFinal we want to pull from in host/LNB
result.granOutputIdx = granOutputIdx;
result.lnbOutputIdx = lnbOutputIdx;
%Get the number of extra fields that were intervention/geometry/downstream we wanted to pull for gran/LNB
result.totalExtraOutputFieldsGran = calcExtraOutputFields(result.extraOutFieldTypeGran, result.granFieldNames);
result.totalExtraOutputFieldsLNB = calcExtraOutputFields(result.extraOutFieldTypeLNB, result.LNBFieldNames);
%setup time points that we will analyze.
result.analysisTimePoints = timesteps';
%Set flag - are we looking at disseminated grans or only the initial granulomas?
result.onlyInitialGrans = onlyInitialGrans;
%p-value cutoff alpha is set.
result.alpha = alpha;
[paramNames, granOutputNames, lnbOutputNames, allOutputNames] ...
    = formatNames(odeHostSim, granOutputIdx, lnbOutputIdx, result.granFieldNames, result.LNBFieldNames);
result.paramNames = paramNames;
result.granOutputNames = granOutputNames;
result.lnbOutputNames = lnbOutputNames;
result.allOutputNames = allOutputNames;

% In case we return without defining the result structure, due to input errors.
result.lhsMatrix = [];

% Get the matrix of varying parameters (granuloma scale). This is the same regardless of
% timestep.
fprintf('Getting LHS matrix (varying parameter values).\n')
result.lhsMatrix = getParams(odeHostSim, onlyInitialGrans, granFilter, false);

% These are needed for the bhfdr_correction function.
fprintf('Getting parameter names.\n')
result.gran.modelOutputNames = granOutputNames;
%result.hostGran.modelOutputNames = granOutputNames;
result.host.modelOutputNames = [lnbOutputNames, granOutputNames];;

% Copy parameters from result into result.gran and result.host
result = copyCommonParametersDown(result, 'gran');
%result = copyCommonParametersDown(result, 'hostGran');
result = copyCommonParametersDown(result, 'host');

result.host.lhsMatrix = hostScaleLHSMatrix(odeHostSim, result);
end

function result = copyCommonParametersDown(result, name)
% Modify results.gran and results.host to share parameters
% that are in common between granuloma and lymph node / blood scales.
% Here, name is 'gran' or 'lnb'.
result.(name).alpha = result.alpha;
result.(name).paramNames = result.paramNames;
result.(name).lhsMatrix = result.lhsMatrix;
result.(name).analysisTimePoints = result.analysisTimePoints;
end % function copyCommonParametersDown

function hostLHSMatrix = hostScaleLHSMatrix(odeHostSim, result) %#ok<INUSD> 
% Retreive the usable LNB LHS matrix from the 
% granuloma LHS matrix.
% 
% H - Number of hosts
%
% P - Number of parameters
%
% Input: odeHostSim object
%        results - currently unused, but it'd be nice to leave it here
%                  in case we update this. 
%
% Output: LNBLHSMatrix - a matrix of H x P parameters used to analyze
%         the PRCC values specifically of host-scale outcomes..

%"true" and "granFilter = []" here ensures that we're
%able to correctly retrieve information from the granuloma-scale
%LHS matrix.
tempMatrix = getParams(odeHostSim, true, [], true);
subsampleRange = 1:odeHostSim.numGransStart:size(tempMatrix,1);
hostLHSMatrix = tempMatrix(subsampleRange,:);
end % function hostScaleLHSMatrix

function result = unpackVarargs(odeHostSim, result, prcc_analysis_varargs)
% Take varargs from prcc_analysis and enter them into results structure
% 
% input: odeHostSim object for checking that the fields exist
% 
%        result structure to be appended
% 
%        prcc_analylsis_varargs to be unpacked.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% output: result - structure updated to contain the following fields:
%
%          extraOutFieldTypeGran - gFT x 1 Cell array of names containing extra
%                                   granuloma output types being accessed, e.g. 
%                                   'downstream' or 'geometry'
%
%          extraOutFieldTypeLNB - lnFT x 1 Cell array of names containing extra
%                                    LN/B output types being accessed, e.g. 
%                                    'downstream'
%
%          granFieldNames - lnFT x 1 Cell array, with the ith cell
%           containing an array of names associated to the ith type
%          
%          LNBFieldNames - lnFT x 1 Cell array, with the ith cell
%           containing an array of names associated to the ith type
%
%           Example: extraOutFieldTypeGran = {'downstream', 'geometry'}
%                    granFieldNames = {{'totalBugs', 'totalMacs'}, {'diamGran'}}
extraOutFieldTypeGran = {};
extraOutFieldTypeLNB = {};
result.granFieldNames = {};
result.LNBFieldNames = {};
for argc = 1:numel(prcc_analysis_varargs)
    % The field type should be 'gran', 'downstream' or 'geometry' in varargs{argc+1}
    % followed by {'Field1', 'Field2'...} in varargs{argc+2}

    % If we're unpacking a granuloma output type...
    if (isequal(prcc_analysis_varargs{argc}, 'gran'))
        extraOutFieldTypeGran{end+1} = prcc_analysis_varargs{argc+1}; %#ok<AGROW> 
        result.granFieldNames{end+1} = prcc_analysis_varargs{argc+2};
        % If the field type does not exist, report an error
        if (~ismember(extraOutFieldTypeGran{end},...
                fieldnames(odeHostSim.hostList{1}.GranCellArray{1})))
            result = sprintf('Error: Gran field type %s does not exist\n',extraOutFieldTypeGran{end});
            return;
        else 
            % Else the field type exists
            for j = 1:numel(result.granFieldNames{end})
                if (~ismember(result.granFieldNames{end}{j},...
                        fieldnames(odeHostSim.hostList{1}.GranCellArray{1}.(extraOutFieldTypeGran{end})) ...
                        ))
                    result = ...
                        sprintf('Error: Gran field %s.%s does not exist\n',...
                        extraOutFieldTypeGran{end},result.granFieldNames{end}{j});
                    return;
                end
            end
        end
        % Else, we're unpacking LNB names...
    elseif (isequal(prcc_analysis_varargs{argc}, 'LNB'))
        extraOutFieldTypeLNB{end+1} = prcc_analysis_varargs{argc+1}; %#ok<AGROW> 
        result.LNBFieldNames{end+1} = prcc_analysis_varargs{argc+2};
        if(~ismember(extraOutFieldTypeLNB{end},...
                fieldnames(odeHostSim.hostList{1}.LNBloodCellArray{1})))
            result = sprintf('Error: LNB field %s does not exist\n',...
                extraOutFieldTypeLNB{end});
            return;
        else
            for j = 1:numel(result.LNBFieldNames{end})
                if (~ismember(result.LNBFieldNames{end}{j},fieldnames(odeHostSim.hostList{1}.LNBloodCellArray{1}.(extraOutFieldTypeLNB{end}))))
                    result = sprintf('Error: LNB field %s.%s does not exist\n',extraOutFieldTypeLNB{end},result.LNBFieldNames{end}{j});
                    return;
                end
            end
        end 
    end % if gran; else LNB
end % for all args

% Append the results structure to hold the names of the 
% additional output names.
result.extraOutFieldTypeLNB = extraOutFieldTypeLNB;
result.extraOutFieldTypeGran = extraOutFieldTypeGran;
end % function unpackVarargs

function totalExtraOutputFields = calcExtraOutputFields(extraOutputFieldType, fieldNameCell)
% Calculates the total number of extra outputs across all field types, 
% e.g. totalExtraOutputFields = 3 if you are taking 2 outputs from 
% (granuloma agent).downstream and one from (granuloma agent).geometry.
totalExtraOutputFields = 0;
for i = 1:numel(extraOutputFieldType)
    totalExtraOutputFields = totalExtraOutputFields +...
        numel(fieldNameCell{i});
end
end
