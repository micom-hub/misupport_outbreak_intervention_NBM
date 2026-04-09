function prccResult = lhsPrccFromCsv(inputFileName, alpha, outputFileName)

%   Read a CSV file made from make-lhs-prcc.py, perform PRCC analysis,
%   including corrections.  The results are saved in the Matlab workspace,
%   and can be optionally written to a .mat file or a set of CSV files, or both.
%
%   The CSV file read can be for multiple model timesteps.
%
%   In comments below:
%
%   TP is the number of timepoints.
%
%   R is number of runs. If the data is not averaged over replications for
%   each experiment it is the number of experiments times the number of
%   replications times the number of timepoints.  If the data is averaged
%   over the replications for each experiment it is the number of experiments
%   times the number of timepoints.
%
%   P is the number of varied parameters and varied initial conditions.
%
%   O is the number of data columns (number of model outputs). This does not
%   include the first two columns of the CSV file, which are for experiment
%   number and time step.
%
%   Inputs:
%
%       inputFileName: Name of the output from make-lhs-prcc.py
%                      This is assumed to have a substring of the form
%                      stat-cols-([0-9]+) where ([0-9]+) is a string of
%                      digits for the number of columns that are statistics,
%                      i.e. model outputs (also including the experiment number
%                      and model timestep). The remaining columns are the
%                      values of the varied parameters.
%
%       alpha: The alpha value to use as the cutoff for significant PRCCs.
%
%       outputFileName: An optional file name to save the PRCC result structure
%                       to. If not specified, or if empty, the result structure
%                       is not saved to a file. In either case it is always left
%                       in the workspace.
%
%    Outputs:
%
%       Structure prccResult in the Matlab workspace, which can be optionally
%       saved in a .mat file if function argument outputFileName is specified.
%
%       prccResult has the following fields:
%
%       analysisTimePoints: A TP x 1 vector of timestep values.
%                           This was obtained from the input file.
%
%       lhsMatrix: An R x P matrix of varied parameter and varied initial
%                  condition values.
%                  This was obtained from the input file.
%
%       modelOutput: A O x R x TP matrix of model output values.
%                    This was obtained from the input file.
%
%       modelOutputNames: An O x 1 vector of data column names from the CSV
%                         header line. This was obtained from the input file.
%
%       paramNames: A P x 1 vector of parameter and initial condition column
%                   names from the CSV header line.
%                   This was obtained from the input file.
%
%       vcInfo: A structure with version control information for the version
%               control project this script is part of.
%
%       alpha: The alpha function argument.
%
%       uncorrectedPrcc: A TP x O x P matrix of the PRCC values.
%
%       uncorrectedSignificance: A TP x O x P matrix of the significance values
%                                for the PRCC values in uncorrectedPrcc.
%
%       uncorrectedSignificantPrcc: A TP x O x P matrix of the PRCC values.
%                                   The same as uncorrectedPrcc, except entries
%                                   for PRCC values that are not significant have
%                                   been set to 1.
%
%       uncorrectedSignificantSignifcance: A TP x O x P matrix. The same as
%                                          uncorrectedSignificance, except that
%                                          significance values for PRCC values
%                                          that are not significant have been
%                                          set to 1.
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
%       bonferroniSignificance: A TP x O x P matrix of signficance values
%                               corrected using the Bonferroni correction.
%
%       bonferroniSignificantPrcc: A TP x O x P matrix of PRCC values.
%                                  The same as uncorrectedPrcc, except entries
%                                  for PRCC values that are not significant 
%                                  according to alpha and bonferroniSignificance
%                                  have been set to 1.
%
%       bonferroniSignificantSignifcance: A TP x O x P matrix. The same as
%                                         bonferroniSignificance, except that
%                                         bonferroniSignificance values that
%                                         are not significant have been set to 1.
%
%       bhfdrSignificance: A TP x O x P matrix of signficance values
%                          corrected using the BHFDR (Benjamini-Hochberg)
%                          correction.
%
%       bhfdrSignificantPrcc: A TP x O x P matrix of PRCC values.
%                             The same as uncorrectedPrcc, except entries
%                             for PRCC values that are not significant 
%                             according to alpha and bhfdrSignificance
%                             have been set to 1.
%
%       bhfdrSignificantSignifcance: A TP x O x P matrix. The same as
%                                    bhfdrSignificance, except that
%                                    bhfdrSignificance values that are not
%                                    significant have been set to 1.

prccResult = setup(inputFileName);

% Get the scripts directory version control info, so we can know which version
% of the scripts directory was used for this run, to aid reproducibility.  Use
% this script's name as the output prefix, to distinuish the version control
% information for a run of this script with version control information from a
% run of another script.
callerFullPath = mfilename('fullpath');
prccResult.vcInfo = saveVersionControlInfo(callerFullPath, '', 'lhsPrccFromCsv.m');

prccResult.alpha = alpha;
prccResult = performPrccWithCorrections(prccResult);

% If requested, save prccResult to a .mat file.
if exist('outputFileName', 'var')
    if ~isempty(outputFileName)
        save(outputFileName, 'prccResult')
    end
end

end % function lhsPrccFromCsv

function  prccResult = setup(inputFileName)

% Read a CSV file and split it up into the data (model output) and the LHS
% matrix (varied parameters and varied initial conditions). Format data and the
% LHS matrix in the format needed for performing PRCC analysis.

% Inputs:
%
%     inputFileName: The name of the CSV file to be read.
%
% Outputs:
%
%     prccResult: A structure with the following fields.
%
%     analysisTimePoints: A TP x 1 vector of timestep values.
%
%     lhsMatrix: An R x P matrix of varied parameter and varied initial
%                condition values.
%
%     modelOutput: A O x R x TP matrix of data (model output) values.
%
%     modelOutputNames: An O x 1 vector of data column names from the CSV
%                       header line.
%
%     paramNames: A P x 1 vector of parameter and initial condition column
%                   names from the CSV header line.

%& Get the number of statistics columns from the file name.
[tokens, matches] = regexp(inputFileName, 'stat-cols-([0-9]+)', 'tokens', 'match');
if length(tokens) == 0
    msg = 'File name "%s" does not contain the number of statistics columns';
    msg = sprintf(msg, inputFileName);
    error('lhsPrccFromCsv:FatalError', msg);
end
numDataVar = str2num(string(tokens(1)));
if numDataVar < 3
    msg = 'The number of data (non-parameter) columns is < 3.\n';
    msg = [msg, 'The first 2 columns are experiment and time step.\n'];
    msg = [msg, 'If < 3 data colums then there are no model output columns.\n'];
    error('lhsPrccFromCsv:FatalError', msg);
end

% Get the output directory to use from the input file name.
[path, ~, ~] = fileparts(inputFileName);
if path == ""
    path = ".";
end
path = strcat(path, "/");

% Read data and headers from specified file
% "1,0": 1 means skip the first row, the header row,
%        0 means don't skip any columns.
tempMatrix = csvread(inputFileName,1,0);
[~, numCols] = size(tempMatrix);

% Read the header line from the input file as text.
fid = fopen(inputFileName);
hdrs = textscan(fid, '%s', numCols, 'delimiter' , ',');
headers = hdrs{1}';

% Get the unique timesteps. Each timestep is present for each run, so we
% reduce that to the unique timesteps. Column 2 of the CSV file has the
% timestep.
prccResult.analysisTimePoints = unique(tempMatrix(:, 2));

% Define the Data matrix
% This assumes all columns after data (model outputs) are parameters.

% dataCsv is (R*TP) x O.  This assumes that the first two columns contain the
% experiment number and timestep, which are not part of the data proper.
dataCsv = tempMatrix(:, 3:numDataVar);

% Define the LHS matrix - the matrix of varied parameters.
% This assumes all columns after data are parameters.
% lhsMatrix is RxP where R is the number of runs and P is the number of
% varied parameters. P = numCols - numDataVar.
% The parameter columns in tempMatrix have the parameter set for a run
% repeated TP times, since the same parameters are used for a run for each
% timestep of the run. We get the unique parameter sets, one set per run.
%
% Include the timestep column. 
nonUniqueLhsMatrix = tempMatrix(:, [2,numDataVar+1:numCols]);
%
% Select the parameter sets for the first timestep. This will be the unique
% parameter sets. We can't use the Matlab unique function because that
% sorts its results. We don't want to change the order of the rows, they
% need to stay in the same order to match the data rows.
lhsMatrix = nonUniqueLhsMatrix(nonUniqueLhsMatrix(:,1) == prccResult.analysisTimePoints(1), :);
%
% Delete the timestep column, keep just the parameter values.
lhsMatrix(:, 1) = [];
prccResult.lhsMatrix = lhsMatrix;

TP = size(prccResult.analysisTimePoints, 1);
R = size(lhsMatrix, 1);
prccResult.modelOutput = reformatData(dataCsv, TP, R);

% Setup the headers for the model output columns.
% This assumes that the first two columns contain the experiment number and
% timestep, which are not headers for the data proper.
prccResult.modelOutputNames(:, 1) = headers(3:numDataVar);

% Setup the headers for the parameter columns
% This assumes all columns after data are parameters
prccResult.paramNames(:, 1) = headers(numDataVar+1:numCols);

end %function setup()

function data = reformatData(dataCsv, TP, R)

% Reformat the data as read from a CSV file to a form usable for PRCC analysis.
%
% Inputs:
%
%     dataCsv: An (R*TP) x O matrix of data (model output) values, as read
%              from the CSV file.
%
%     TP: The number of timepoints represented in dataCsv.
%
%     R: The number of runs represented in dataCsv.
%
% Outputs:
%
%     data: An O x R x TP matrix, a reformatted version of dataCsv.
%
%
% For example, for O = 2, R = 3 and TP = 2.
% dataCsv: (R*TP) x O = 6 x 2
%     1 11
%     2 12
%     3 21
%     4 22
%     5 31
%     6 32
%
% data: O x R x TP = 2 x 3 x 2
% data(1,:,:):
%     1 2
%     3 4
%     5 6
%     
% data(2,:,:):
%    11 12 
%    21 22
%    31 32

[dataRows, O] = size(dataCsv);
if dataRows ~= R * TP
    ident = 'lhsPrccFromCsv:reformatData:Error:invalidDataRowCount';
    fmt = 'The number of rows in the CSV data, %d, does not match the';
    fmt = [fmt, ' number of runs, %d, times the number of timepoints, %d,'];
    fmt = [fmt, ' which is %d.'];
    msg = sprintf(fmt, dataRows, R, TP, R*TP);
    exception = MException(ident, msg);
    throw(exception);
end

t = 0;
r = 1;
data = zeros(O, R, TP);
for i = 1:dataRows
    t = t + 1;
    if t > TP
        t = 1;
        r = r + 1;
    end

    data(:, r, t) = dataCsv(i, :);
end

end % function reformatData

