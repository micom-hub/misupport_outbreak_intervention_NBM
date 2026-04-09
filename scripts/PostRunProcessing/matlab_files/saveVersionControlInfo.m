function vcInfo = saveVersionControlInfo(callerFullPath, outputDirectory, outputPrefix)

% A function to save the version control information for the source code being
% run.
%
% Input:
%
% callerFullPath: The file path of the source code being run. This must be
%                 supplied by the caller because this function isn't
%                 necessarily, and typically won't be, in the same location.
%                 It should be the output of function mfilepath('fullpath'),
%                 so it will also have the name of the calling function.
%
% outputDirectory: The output directory to store the version control info
%                  into. This will be empty if the current directory is
%                  the output directory.
%
% outputPrefix: A character string that is prepended to the file name of each
%               output file. It can be empty.
%
% Output:
%
% The version control info is stored in file version-control-info.txt in
% directory outputDirectory.
%
% vcInfo: A structure with string fields, one field for each version control
%         command run to get version information, and also fields for numerc
%         counts of the modified and unversioned files in the source code
%         directory.
%

% This is needed when exporting some of our code, such as our standalone ODE
% LHS code, to places external to the Kirschner lab, since they won't
% necessarily use svn for version control.
vcInfo = '';
%return % Uncomment if not using svn (Subversion) for version control.


% Remove the calling function name from the code path.
codePath = regexprep(callerFullPath, '/[^/]+$', '');

% If outputDirectory is empty then the current directory is the output
% directory. When running version control commands we need to cd to
% codePath, but the output of the commands needs to be written to the
% output directory, so an explicit output directory is needed.
if isempty(outputDirectory)
    outputDirectory = [pwd '/'];
end

% So we can correctly append a file name to the output directory.
if ~ endsWith(outputDirectory, '/')
    outputDirectory = [outputDirectory '/'];
end


outputFileName = [outputDirectory, outputPrefix, '-version-control-info.txt'];

% Run the status command before the info command. Otherwise, when running in
% the model source directory the output file created by the info command will
% appear as an unversioned file.
svnStatusResult = runVCInfoCommand(codePath, outputFileName, 'svn status', false);
vcInfo.svnStatus = svnStatusResult;

svnInfoResult = runVCInfoCommand(codePath, outputFileName, 'svn info', true);
vcInfo.svnInfo = svnInfoResult;

[vcInfo.unversionedCount, vcInfo.modifiedCount] = getStatusCounts(svnStatusResult);

end % function saveVersionControlInfo


function cmdResult = runVCInfoCommand(codePath, outputFileName, command, append)

% Run a version control command an append it's output to a file and return it's
% output as the function return value.
%
% Input:
%
% codePath: The file path of the source code being run. It is the location
%           where the version control commands are to be run.
%
% outputFileName: The file to store the version control info into.
% 
% command: The version control command to run.
%
% append: A boolean value. It should be false for the first version control
%         command to be run, so it overwrites any existing outputFileName file
%         and true for subsequent commands so their output gets appended to
%         file outputFileName.
%
% Output:
%
% The output from the version control command is appended to file
% outputFileName.
%
% cmdResult: Contains the output of the version control command.


% When append is false the echo command in the call to system will
% overwrite the output file, to clear any contents from a prior run of the
% calling program. Otherwise have it append to the output file.
echoOption = '>';
if append
    echoOption = '>>';
end

% Needed on some systems so the PATH is properly set, so the svn command will
% be found. This is known to be needed on recent MacOS systems - even though
% Bash is the shell, the Matlab system command does not perform the Bash shell.
% This seems to be due to invoking Matlab from an icon rather than from a
% terminal command.
sysType = computer;
if strcmp(sysType, 'GLNXA64')
	bashConfig = 'source $HOME/.bashrc; ';
elseif strcmp(sysType, 'MACI64') || strcmp(sysType, 'MACA64')
	bashConfig = 'source $HOME/.bash_profile; ';
else
    msgfmt = 'saveVersionControlInfo error, unknown sysType %s\n';
    msg = sprintf(msgfmt, sysType);
    throw(MException('saveVersionControlInfo:BadSysType', msg));
end

% Capture the command output into a variable, then write the command and the
% command output to the output file. This is for the case of running a model in
% its source directory, so the output file will be created in the source
% directory, and running an svn status command.  If the command is written to
% the output file prior to running the command, then the output file will exist
% when the command is run, and for an svn status command will appear as an
% unversioned file. This means there will always be an unversioned file.
%
% tee so the command output goes to the file and also ends up in cmdResult.
% Always have tee append (-a) since it follows an echo command to the same
% file.

% Do 'env LD_LIBRARY_PATH=""' before the svn command. With some newer
% versions of Matlab, running an svn command from Matlab causes the
% following error otherwise:
% svn: symbol lookup error: /lib/x86_64-linux-gnu/libsvn_subr-1.so.1: undefined symbol: apr_crypto_block_cleanup
% This is known to happend with Matlab 2025a on Ubuntu 22.04.

fmt = '%s cd %s; cr=$(env LD_LIBRARY_PATH="" %s); echo "%s" %s %s; echo "$cr" | tee -a %s';
sysCmd = sprintf(fmt, bashConfig, codePath, command, command, echoOption, outputFileName, outputFileName);
[status, cmdResult] = system(sysCmd);
if status ~= 0
     msgfmt = 'saveVersionControlInfo error running command:\n%s\nstatus: %d\nresult: %s\n';
     msg = sprintf(msgfmt, sysCmd, status, cmdResult);
     throw(MException('saveVersionControlInfo:CmdFailure', msg));
end

% splitlines to change cmdResult from a single charaacter array into a
% cell array with an element for each line of text, for readability.
cmdResult = splitlines(cmdResult);

end

function [unversionedCount, modifiedCount] = getStatusCounts(svnStatusResult)

% Get the number of unversioned files and the number of versioned files
% that have been modified.
%
% Input:
%
% svnStatusResult: A cell array of character arrays. Each element is one
%                  line of output from an svn status command. The first
%                  character of each line indicates the status. "?" means
%                  unversioned and anything else means modified in some
%                  way - 'M' means already in version control but modified,
%                  'A' means not yet in version control but will be added
%                  on the next commit, 'D' means in version control but
%                  will be deleted on the next commit, etc. (other less
%                  used status codes). If a file is in version control but
%                  has not been modified in any way it won't appear in the
%                  svn status output.
%
% Output:
%
% unversionedCount: The number of files with a status code of '?'.
%
% modifiedCount: The number of files with a status code other than '?'.
% 

unversionedCount = 0;
modifiedCount = 0;
for i  = 1:length(svnStatusResult)
    if isempty(svnStatusResult{i})
        continue
    end
    
    if strcmp(svnStatusResult{i}(1), '?')
        unversionedCount = unversionedCount + 1;
    else
        modifiedCount = modifiedCount + 1;
    end
end
end
