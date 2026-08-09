function [flopTotal,Details] = FLOPS(fileName,MATfileName,profileStruct)
%FLOP Count the floating-point operations for a MATLAB script or function
%
% Syntax:
%   
%   Standard syntax:  FLOPS(fileName,MATfileName,profileName)
%   Workable syntax:  FLOPS(fileName,MATfileName)
%   Risky    syntax:  FLOPS('MyScriptName') for MATLAB scripts
%                     FLOPS('MyFunName(A,B,C)') for a MATLAB function
%
% Description:
%
%   Instead of FLOPS counting at runtime, this program scans each line of
%   the codes. From the matrix sizes, the program infers the required
%   FLOPS. To analyze MATLAB codes, users provide variable sizes and the
%   number of execution times of each line.
%
%   Before using this program, do the following
%   1) Run the codes and store all the variables in a MAT file.
%      The easiest way is to add a line at the end of your codes:
%      save MATfileName
%      
%   2) Run the codes using the profiler tool
%      profile on
%      (run your MATLAB codes here)
%      profileStruct = profile('info');
%
%   Then you can use the program to count FLOPs.
%   3) FLOP(fileName,MATfileName,profileStruct)
%      where fileName is the MATLAB codes for counting FLOPs
%            MATfileName is the MAT file name that stores variables
%            profileStruct is the results obtained from profile tools 
%
%
% Input Arguments:
%
%   fileName - A string that specifies the MATLAB file name for FLOP counts
%
%   fileNameMAT - A string that specifies the MAT file name that stores variables
%
%   profileStruct - A struct returned by profile('info')
%
%
% Output Arguments:
%
%   flopTotal -  a scalar, total floating-point operations
%
%   Details   -  a n-by-1 struct with three fields:
%                o FLOPS: floating-point operations of that line of codes
%                o Multiplier: number of times the line is executed
%                o Detail: Accounting details in a r-by-2 cell array
%                          The first column is the operator/function name
%                          The second column is FLOPS for that operator
%
%
% Notes: 
%
% o It is not feasible to count absolutely all floating point.
%   The counting method loosely follows the convention of MATLAB 5.
%   However, users may add or override rules in ExtendedRules.xlsx
%
% o If profileStruct is not provided, the program will try to analyze the
%   FOR loops to guess how many times each line is executed. However,
%   it is impossible to identify whether codes within a IF...END block are
%   executed or not. Therefore, the program could over-estimate the FLOPs
%   because some lines of codes may not be executed.
%
% o It is risky to call this function with only one input argument: a
%   MATLAB script name or a function signiture. The program will create a
%   copy of the MATLAB codes and add save(...) at the end of the file. Then
%   the program profiles the new codes and run the standard FLOPS syntax
%   with three input arguments. Such automatic algorithm may or may not
%   work on your computer.
%
%
% Limitations:
%
% o Variable sizes cannot change anywhere in the codes.
%
% o Subfunction and nested functions are not supported.
%
% o Full matrices without special structures are assumed in linear algebra 
%   operations.
%
%
% Written by Hang Qian
% Contact: matlabist@gmail.com


clc

% Read MATLAB script/function that needs FLOPS analysis
if nargin < 1
    error('MATLAB and MAT file names are required.')
end

% Convenient but risky way to use this program
if nargin == 1
    [flopTotal,Details] = OneStep(fileName);
    return
end

if ~ischar(fileName)
    error('The first input argument should be a string that indicates a MATLAB script/function name.')
end

% Load MATLAB codes text into a cell array
fileName = strrep(fileName,'.m','');
TXTcell = readText(fileName);
TXTcell = strtrim(TXTcell);
for justRepeat = 1:30
    TXTcell = strrep(TXTcell,'  ',' ');
end
nline = length(TXTcell);

% Concatenate codes that were in multiple lines
try
    for m = (nline-1):-1:1
        if length(TXTcell{m}) >= 3  &&  strcmp(TXTcell{m}(end-2:end),'...')
            disp(['Lines around ',num2str(m),' are combined to a single line.'])
            TXTcell{m} = [TXTcell{m}(1:end-3), TXTcell{m+1}];
            TXTcell{m+1} = '';
        end
    end
catch
    disp('Unable to concatenate multiple lines. ')
end

% Check the MAT file that stores variables
if nargin < 2
    error('Please run %s first and store all variables in a MAT file.',fileName)
end

if ~ischar(MATfileName)
    error('The second input argument should be a string that indicates the MAT file name.')
end

% Generate varList, which is a r-by-3 cell array
% First column: variable names. Intermediate variables have the form GenVar88
% Second column: variable sizes in the form [nrow,ncol]
% Third column: variable values of intermediate variables
try   
    varInfo = whos('-file',MATfileName);
    varNames = {varInfo(:).name};
    varSizes = {varInfo(:).size};
    nvar = length(varNames);
    varList = cell(nvar,3);
    varList(:,1) = varNames;
    varList(:,2) = varSizes;
catch
    error('Unable to load variables in the MAT file')
end

% Load the profile struct and obtain the number of execution times
flopMultiplier = zeros(nline,1);
isProfile = false;
if nargin > 2        
    if ~isstruct(profileStruct)
        error('The third input argument should be a struct returned by profile(''info'').')
    end
    
    % Get execution times of the main function
    AIOstruct = profileStruct.FunctionTable;
    for m = 1:length(AIOstruct)
        if strcmp(AIOstruct(m).FunctionName,fileName)
            indLines = AIOstruct(m).ExecutedLines(:,1);
            if max(indLines) > nline
                break
            end            
            flopMultiplier(indLines) =  AIOstruct(m).ExecutedLines(:,2);
            isProfile = true;
            break
        end
    end 
    
    % Get execution times of the subfunctions
    for m = 1:length(AIOstruct)
        if strncmp(AIOstruct(m).FunctionName,[fileName,'>'],length([fileName,'>']))
            indLines = AIOstruct(m).ExecutedLines(:,1);
            if max(indLines) > nline
                break
            end            
            flopMultiplier(indLines) =  AIOstruct(m).ExecutedLines(:,2);            
        end
    end
end

if ~isProfile
    flopMultiplier = ones(nline,1);
    multiplierQueue = NaN(20,1);
    controlCount = 0;
    warning('profileStruct is unavailable or corrupted. Analyze FOR loop instead, but it is less accurate and likely to over-estimate FLOPs.')
end

% flopEachLine is the identified FLOPS for each line of the codes
% flopMultiplier is larger than one when the codes are in a FOR loop
% flopDetail keeps record of everything counted in a cell array
flopEachLine = zeros(nline,1);
flopDetail = cell(nline,1);

% Load extended rules in the EXCEL file ExtendedRules.xlsx
if exist('ExtendedRules.xlsx','file') == 2
    [~,Header]=xlsread('ExtendedRules.xlsx');    
    NewRule = Header(2:end,1:2);
    emptyMask = strcmp(NewRule(:,1),'');
    NewRule = NewRule(~emptyMask,:);
else
    NewRule = cell(0,2);
end

% Analyze each line of the codes
for m = 1:nline
    
    % Load the current line
    txt = TXTcell{m};    
    fprintf('\nLine %d: %s\n',m,txt);
            
    % Skip comments, disp, fprintf and other lines that do not need FLOPS count
    if isempty(txt) || strncmp(txt,'%',1) ...
            || strncmp(txt,'disp(',5) || strncmp(txt,'fprintf(',8) ...
            || strncmp(txt,'warning(',8) || strncmp(txt,'error(',6) ...
            || strncmp(txt,'load(',5) || strncmp(txt,'save(',5)            
        continue
    end
    
    try
        
        % Remove comments after the codes
        locComments = strfind(txt,'; %');
        if ~isempty(locComments)
            txt = txt(1:locComments(1)+1);
        end
        
        % If profileStruct is not available, analyze FOR loop for the number of
        % execution times
        if ~isProfile
            % Identify FOR loops and the number of iteration
            if strncmp(txt,'for ',4)
                
                % loopVec is something like 1:nobs, 1:2:100
                loopVec = txt(strfind(txt,'=')+1:end);
                nIter = length(virtualEval(MATfileName,loopVec));
                
                controlCount = controlCount + 1;
                multiplierQueue(controlCount) = nIter;
                flopMultiplier(m:end) = flopMultiplier(m:end) .* multiplierQueue(controlCount);
                continue
            end
            
            % Handle other control statements
            % Put the multiplier in the queue as one, so it is easier to handle END
            if strncmp(txt,'if ',3) || strncmp(txt,'switch ',7) || strncmp(txt,'function ',9)...
                    || strncmp(txt,'while ',6) || strcmp(txt,'try')
                controlCount = controlCount + 1;
                multiplierQueue(controlCount) = 1;
                continue
            end
            
            % The END indicates codes after that line will no long subject to the
            % current multiplier, and the most recent multiplier in the queue retires
            if strcmp(txt,'end') || strcmp(txt,'end;')...
                    || ~isempty(strfind(txt,';end')) || ~isempty(strfind(txt,'; end'))
                flopMultiplier(m:end) = flopMultiplier(m:end) ./ multiplierQueue(controlCount);
                multiplierQueue(controlCount) = NaN;
                controlCount = controlCount - 1;
                continue
            end
        end
        
        % Avoid another call to a function
        if strncmp(txt,'function ',9)
            continue
        end
        
        % Replace transpose by intermediate variables
        [txt,varList] = transpose2var(txt,varList,MATfileName);
        
        % Start the FLOP counter for the current line
        flopCounter = 0;
        flopDetails = cell(0,2);
        
        % Search for brackets. Each time, the most inner bracket is replaced by
        % its output variables. FLOPS will be counted for math expressions and
        % some statistics functions. Essentially, it breaks down a long
        % expression into several expressions by creating intermediate
        % variables. The intermediate variables will be added to the varList
        % with their size information.
        % In addition, such algorithm can also handle {} and [], so we add a
        % FOR loop to first handle cell array, then matrix concatenation, and
        % lastly the function/indexed matrix.
        for bracketType = 3:-1:1
            for justRepeat = 1:100
                
                % Analyze the brackets
                try
                    [~,~,LeftLoc,RightLoc] = bracket(txt,bracketType);
                catch
                    % Omit unbalanced brackets
                    break
                end
                if isempty(LeftLoc)
                    break
                end
                
                % Only work on the most inner bracket for each iteration
                loc1 = LeftLoc(end,1);
                loc2 = RightLoc(end,1);
                
                % Search backward from the left bracket to obtain the function name
                % If funName is empty, it is a math expression
                [isFun,funName] = getFunName(txt,loc1);
                if isFun
                    % Extract the input arguments of the function
                    funInputArg = txt(loc1+1 : loc2-1);
                    
                    % Count FLOPs if the input arguments contain math expression
                    % such as sum(D*D',2)
                    [flopCounter,flopDetails] = countMath(flopCounter,flopDetails,varList,MATfileName,funInputArg,NewRule);
                    
                    % Count flops for functions such as sum, sin, chol
                    [flopCounter,flopDetails] = funCount(flopCounter,flopDetails,varList,MATfileName,funName,funInputArg,NewRule);
                    
                    % Evaluate the entire function or indexed matrix
                    % Replace the function by an intermediate variable
                    % Keep a record of the intermediate variable in the varList
                    [txt,varList] = virtualEvalPlus(txt,varList,MATfileName,loc1-length(funName),loc2);
                    
                else
                    % Extract the math expression, such as (B+C*2)
                    mathExpress = txt(loc1+1 : loc2-1);
                    
                    % Replace bracket math expression by artificial variables
                    % Count flops for +-*/\
                    [flopCounter,flopDetails] = countMath(flopCounter,flopDetails,varList,MATfileName,mathExpress,NewRule);
                    
                    % Evaluate the entire math expression
                    % Replace the expression by an intermediate variable
                    % Keep a record of the intermediate variable in the varList
                    [txt,varList] = virtualEvalPlus(txt,varList,MATfileName,loc1,loc2);
                    
                end
            end
        end
        
        % Finally, it is a math expression after all brackets are removed
        [flopEachLine(m),flopDetail{m}] = countMath(flopCounter,flopDetails,varList,MATfileName,txt,NewRule);
        
        if m < 10
            fprintf('        FLOPs = %d * %d = %d\n',flopEachLine(m),flopMultiplier(m),flopEachLine(m)*flopMultiplier(m))
        elseif m < 100
            fprintf('         FLOPs = %d * %d = %d\n',flopEachLine(m),flopMultiplier(m),flopEachLine(m)*flopMultiplier(m))
        else
            fprintf('          FLOPs = %d * %d = %d\n',flopEachLine(m),flopMultiplier(m),flopEachLine(m)*flopMultiplier(m))
        end
    
    %
    catch
        warning('Unable to parse this line of codes. Skip to the next line.')
        continue
    end
    %}
    
end

flopTotal = nansum(flopEachLine .* flopMultiplier);
fprintf('\nTotal FLOPs = %d\n',flopTotal);

Details = struct('FLOPS',num2cell(flopEachLine),'Multiplier',num2cell(flopMultiplier),'Detail',flopDetail);

end


%-------------------------------------------------------------------------
function [flopTotal,Details] = OneStep(signiture)
%OneStep Fast but risky way to prepare inputs for FLOPS
%
% Syntax:
%
%   [flopTotal,Details] = OneStep('MyScriptName')
%   [flopTotal,Details] = OneStep('MyFunName(A,B,C)')
%
% Description:
%
%   1. Create a temporary copy of the MATLAB codes
%   2. Add save(...) to the new codes
%   3. Profile the new codes
%   4. Call FLOPS() to count floating point operations
%
% Input Arguments:
%
%   signiture - For MATLAB scripts, this is the file name of the scripts
%               For a MATLAB function, this is the function signiture, say
%               'MyFunName(A,B,C)', where the actual input matrices A,B,C
%               must exist in the current workspace.
%
% Output Arguments:
%
%   flopTotal -  a scalar, total floating-point operations
%
%   Details   -  a n-by-1 struct with three fields:
%                o FLOPS: floating-point operations of that line of codes
%                o Multiplier: number of times the line is executed
%                o Detail: Accounting details in a r-by-2 cell array
%                          The first column is the operator/function name
%                          The second column is FLOPS for that operator

if ~ischar(signiture)
    error('The input argument must be a string, either a MATLAB script name or function signiture.')
end

% Distinguish scripts from functions
% Identify the file name
locBracket = strfind(signiture,'(');
if isempty(locBracket)
    fileName = signiture;
else
    fileName = signiture(1:locBracket-1);
end
fileName = strrep(fileName,'.m','');
fileNameTemp = [fileName,'Temp'];

% Scan the text
TXTcell = readText(fileName);

% Run the profile tool first
profile on
evalin('base',[signiture,';']);
profileStruct = profile('info');
AIOstruct = profileStruct.FunctionTable;

% Save variables after the last line of each function/subfunction/scripts
count = 0;
saveLine = zeros(3,1);
for m = 1:length(AIOstruct)
    if strcmp(AIOstruct(m).FunctionName,fileName) ...
            || strncmp(AIOstruct(m).FunctionName,[fileName,'>'],length([fileName,'>']))
        lastLine = max(AIOstruct(m).ExecutedLines(:,1));
        count = count + 1;
        if strncmp(TXTcell(lastLine),'end',3)
            saveLine(count) = lastLine-1;
        else
            saveLine(count) = lastLine;
        end
    end
end

% Write new lines: save('fileNameTemp','-append')
% Also create an empty MAT file fileNameTemp
save(fileNameTemp,'');
writeTxt = {['save(''' fileNameTemp ''',''-append'')']};
for m = count:-1:1
    TXTcell = [TXTcell(1:saveLine(m)); writeTxt; TXTcell(saveLine(m)+1:end)];
end

% Generate a MATLAB script or function
fileID = fopen([fileNameTemp,'.m'],'w');
for m = 1:length(TXTcell)
    fprintf(fileID,'%s\n',TXTcell{m});
end
fclose(fileID);

% Refresh function and file system path caches
% Hopefully, the newly created fileNameTemp can be recognized
rehash

% Call the regular syntax to count FLOPS
codesTemp = strrep(signiture,fileName,fileNameTemp);
codesTemp = [codesTemp,';'];
profile on
evalin('base',codesTemp);
profileStruct = profile('info');
[flopTotal,Details]  = FLOPS(fileNameTemp,fileNameTemp,profileStruct);

% Remove the temporary files created by this program
% delete([fileNameTemp,'.m'])
% delete([fileNameTemp,'.mat'])

end

%-------------------------------------------------------------------------
function TXT = readText(fileName)

%readText Scan a text file and load the contents in a cell array.
%
% Syntax:
%
%   TXT = readText(fileName)
%
% Description:
%
%   By partial matching the file name in the current folder, the program
%   scans the text and then load the contents in a cell array. Each cell
%   element contains a line of codes. If there are multiple matches, MATLAB
%   codes take the priority, followed by the TXT file.
%
% Input Arguments:
%
%   fileName - A string that specifies the MATLAB file name. 
%              It can be a full name with extension, or a partial name.
%
% Output Arguments:
%
%   TXT -  a n-by-1 cell array that contains each line of the codes.
%
% Written by Hang Qian
% Contact: matlabist@gmail.com


if ~ischar(fileName)
    error('File name must be a string.')
end

% Open the file
if ~isempty(strfind(fileName,'.'))
    % File name contains an extension, thus no ambiguity
    fid = fopen(fileName,'r');
else
    % Search current folder to match the file name
    % Partial match is supported
    listing = dir;
    nameAIO = {listing(:).name}';
    mask = strncmp(nameAIO,fileName,length(fileName));
    nmatch = sum(mask);
    if nmatch == 0
        mask = strncmpi(nameAIO,fileName,length(fileName));
        nmatch = sum(mask);
    end
    
    switch nmatch
        case 0
            % It works when fileName contains full path without extension
            fid = fopen([fileName,'.m'],'r');
            if fid == -1
                fid = fopen([fileName,'.txt'],'r');
            end
        case 1
            % Successful match
            nameAIOcut = nameAIO(mask);
            fid = fopen(nameAIOcut{1},'r');
        otherwise
            % Multiple match: MATLAB file takes priority and then txt file
            nameAIOcut = nameAIO(mask);
            maskMATLAB = strcmp(nameAIOcut(max(1,end-1):end),'.m');
            if any(maskMATLAB)
                nameAIOcut = nameAIOcut(maskMATLAB);
            else
                maskTXT = strcmp(nameAIOcut(max(1,end-3):end),'.txt');
                if any(maskTXT)
                    nameAIOcut = nameAIOcut(maskTXT);
                end                    
            end
            fid = fopen(nameAIOcut{1},'r');
    end
end
    
if fid == -1
    error('Unable to open the file.')
end

% Read MATLAB codes
vecASCII = fread(fid,Inf,'*uint8');

% Ideally, the codes contain new-line markers (10) and return markers (13).
% The codes can also correctly displayed in Notepad. However, occasionally,
% codes only have new-line markers (10) without return markers (13). Such
% file will display as a single line in Notepad.
if ~isempty(find(vecASCII==13,1))
    vecASCII = [13;10; vecASCII ; 13;10];
    ENTER_KEY = find(vecASCII==13);
    nline = length(ENTER_KEY) - 1;
    TXT = cell(nline,1);
    for m = 1:nline
        val = vecASCII(ENTER_KEY(m)+2:ENTER_KEY(m+1)-1)';
        TXT{m} = native2unicode(val);
    end
else
    vecASCII = [10; vecASCII ; 10];
    ENTER_KEY = find(vecASCII==10);
    nline = length(ENTER_KEY) - 1;
    TXT = cell(nline,1);
    for m = 1:nline
        val = vecASCII(ENTER_KEY(m)+1:ENTER_KEY(m+1)-1)';
        TXT{m} = native2unicode(val);
    end
end

% If the above algorithm fails, try to use text scan
if nline <= 1
    TXT = textscan(fid,'%s','Delimiter','\n');
    if size(TXT,1) == 1
        TXT = TXT{1};
    end
    nline = size(TXT,1);
    if nline <= 1
        warning('Unable to read the text in the file.')
    end
end

% Close the file
fclose(fid);

end

%-------------------------------------------------------------------------
% Evaluate an expression in a separate function space
% Error evaluation will return an empty matrix
function value = virtualEval(MATfileName,expression,varList)

% Load MAT file variables
load(MATfileName);

if isempty(strfind(expression,'GenVar'))
    % Try to evaluate the expression.
    try
        value = eval(expression);
    catch
        value = [];
    end
else
    % Load variables in the varList
    if nargin > 2
        for m = 1:size(varList,1)
            if strncmp(varList{m,1},'GenVar',6)
                eval(['GenVar' num2str(m) ' = varList{' num2str(m) ',3};'])
            end
        end
    end
    
    % Try to evaluate the expression.
    try
        value = eval(expression);
    catch
        value = [];
    end    
end

end


%-------------------------------------------------------------------------
% Evaluate an expression in a separate function space
% Replace the expression by an intermediate variable
% Keep a record of the intermediate variable in the varList
function [txt,varList] = virtualEvalPlus(txt,varList,MATfileName,locStart,locEnd)
OutputMatrix = virtualEval(MATfileName,txt(locStart:locEnd),varList);
nvar = size(varList,1);
GenVarName = ['GenVar',num2str(nvar+1)];
txt = [txt(1:locStart-1),GenVarName,txt(locEnd+1:end)];
varList{nvar+1,1} = GenVarName;
varList{nvar+1,2} = size(OutputMatrix);
varList{nvar+1,3} = OutputMatrix;
end


%-------------------------------------------------------------------------
% Check if the txt is a valid variable name (letter, number, underline)
function legal = isLegalName(txt)
native = int8(txt);
numberInd = (native >= 48) & (native <= 57);
upperInd = (native >= 65) & (native <= 90);
lowerInd = (native >= 97) & (native <= 122);
letterInd = upperInd | lowerInd;
underlineInd = (native == 95);
legal = numberInd | letterInd | underlineInd;
end


%-------------------------------------------------------------------------
% Search backward from the left bracket to obtain the function name 
% e.g. randn(3,1), size(A)
% If the left bracket is a math bracket, say A * (B+C), name will be empty
function [isFun,funName] = getFunName(txt,locBracket)
locStart = 1;
for m = locBracket-1:-1:1
    if ~isLegalName(txt(m))
        locStart = m + 1;
        break
    end
end
funName = txt(locStart:locBracket-1);
isFun = ~isempty(funName);
end


%-------------------------------------------------------------------------
% Search backward and forward from an operator to obtain the variable names
% e.g. ABC + nobs
function [varName1,varName2,locStart1,locEnd2] = getMathVarName(txt,loc)

% Backward search for the first variable
locEnd1 = loc;
for m = loc-1:-1:1
    if isLegalName(txt(m))
        locEnd1 = m;
        break
    end
end

locStart1 = 1;
for m = locEnd1-1:-1:1
    if ~isLegalName(txt(m))
        locStart1 = m + 1;
        break
    end
end

varName1 = txt(locStart1:locEnd1);

% Forward search for the second variable
nstr = length(txt);
locStart2 = loc;
for m = loc+1:nstr
    if isLegalName(txt(m))
        locStart2 = m;
        break
    end
end

locEnd2 = nstr;
for m = locStart2+1:nstr
    if ~isLegalName(txt(m))
        locEnd2 = m - 1;
        break
    end    
end

varName2 = txt(locStart2:locEnd2);

end

%-------------------------------------------------------------------------
% Obtain the variable size by its name in the varList
function [nrow,ncol] = getVarSize(varName,varList)
varName = strtrim(varName);
mask = strcmp(varList(:,1),varName);
if sum(mask) == 1
    seq = 1:size(varList,1);
    ind = seq(mask);
    sizeInfo = varList{ind,2};
    nrow = sizeInfo(1);
    ncol = sizeInfo(2);    
else
    nrow = 1;
    ncol = 1;
end
end


%-------------------------------------------------------------------------
% Count FLOPs in a math expression
function [flopCounter,flopDetails] = countMath(flopCounter,flopDetails,varList,MATfileName,mathExpress,NewRule)

% Negation, such as A = -B, does not count towards FLOPS
if ~isempty(strfind(mathExpress,'-'))
    mathExpress = strrep(mathExpress,'=-','= ');
    mathExpress = strrep(mathExpress,'= -','=  ');
    mathExpress = strrep(mathExpress,'>-','> ');
    mathExpress = strrep(mathExpress,'> -','>  ');
    mathExpress = strrep(mathExpress,'<-','< ');
    mathExpress = strrep(mathExpress,'< -','<  ');
end

for r = 1:4
    
    % Hierarchy of arithmetic operations
    switch r
        case 1
            operatorPool = {'.^', '^'};
        case 2
            operatorPool = {'.*', '*', './', '.\', '/', '\'};
        case 3
            operatorPool = {'+', '-'};
        case 4
            operatorPool = {'>', '>=','<', '<=', '==','~='};
    end
    
    % Work on one operator at a time
    for justRepeat = 1:100
        
        % Search for operators
        loc = [];
        for m = 1:length(operatorPool)
            loc = [loc, strfind(mathExpress,operatorPool{m})]; %#ok<AGROW>
        end
        
        if isempty(loc)
            break
        end
        
        % Only work on the first one
        % Count FLOPS from left to right
        % Replace the math operation with an intermediate variable
        loc = min(loc);
        operator = mathExpress(loc);
        if strcmp(operator,'.')
            operator = mathExpress(loc:loc+1);
        end
        if (strcmp(operator,'>') || strcmp(operator,'<') || strcmp(operator,'~'))...
                && length(mathExpress) > loc && strcmp(mathExpress(loc+1),'=')
            operator = mathExpress(loc:loc+1);
        end        
        [varName1,varName2,locStart1,locEnd2] = getMathVarName(mathExpress,loc);
        [nrow1,ncol1] = getVarSize(varName1,varList);
        [nrow2,ncol2] = getVarSize(varName2,varList);
        flopAdd = Rules(operator,nrow1,ncol1,nrow2,ncol2,NewRule);
        flopCounter = flopCounter + flopAdd;
        flopDetails(size(flopDetails,1)+1,:) = {operator,flopAdd};
        [mathExpress,varList] = virtualEvalPlus(mathExpress,varList,MATfileName,locStart1,locEnd2);
    end
    
end

end


%-------------------------------------------------------------------------
% Count flops for functions such as sum, sin and chol
function [flopCounter,flopDetails] = funCount(flopCounter,flopDetails,varList,MATfileName,funName,funInputArg,NewRule)


% User supplied new rules for functions in ExtendedRules.xlsx
if ~isempty(NewRule)    
    for m = 1:size(NewRule,1)
        if strcmp(funName,NewRule{m,1})
            % Identify the sizes of input arguments
            locComma = strfind(funInputArg,',');
            switch length(locComma)
                case 0
                    InputMatrix1 = virtualEval(MATfileName,funInputArg,varList);
                    [nrow1,ncol1] = size(InputMatrix1);
                case 1
                    InputMatrix1 = virtualEval(MATfileName,funInputArg(1:locComma(1)-1),varList);
                    [nrow1,ncol1] = size(InputMatrix1);
                    InputMatrix2 = virtualEval(MATfileName,funInputArg(locComma(1)+1:end),varList);
                    [nrow2,ncol2] = size(InputMatrix2); %#ok<ASGLU>
                case 2
                    InputMatrix1 = virtualEval(MATfileName,funInputArg(1:locComma(1)-1),varList);
                    [nrow1,ncol1] = size(InputMatrix1);
                    InputMatrix2 = virtualEval(MATfileName,funInputArg(locComma(1)+1:locComma(2)-1),varList);
                    [nrow2,ncol2] = size(InputMatrix2); %#ok<ASGLU>
                    InputMatrix3 = virtualEval(MATfileName,funInputArg(locComma(2)+1:end),varList);
                    [nrow3,ncol3] = size(InputMatrix3);                     %#ok<ASGLU>                
            end
            nrow = nrow1; %#ok<NASGU>
            ncol = ncol1; %#ok<NASGU>
            
            % FLOPS count
            try
                flopAdd = round(eval(NewRule{m,2}));
            catch
                flopAdd = 0;
                warning('User supplied rules %s cannot be evaluated.',NewRule{m,1})
            end            
            flopCounter = flopCounter + flopAdd;
            flopDetails(size(flopDetails,1)+1,:) = {funName,flopAdd};
            return            
        end
    end    
end


% Handle an important function bsxfun
if strcmp(funName,'bsxfun')
    if ~isempty([strfind(funInputArg,'@plus'),strfind(funInputArg,'@minus'),...
            strfind(funInputArg,'@times'),strfind(funInputArg,'@rdivide'),...
            strfind(funInputArg,'@ldivide')])
        locComma = strfind(funInputArg,',');
        if length(locComma)<2
            return
        end        
        Expression1 = funInputArg(locComma(1)+1:locComma(2)-1);
        Expression2 = funInputArg(locComma(2)+1:end);
        InputMatrix1 = virtualEval(MATfileName,Expression1,varList);
        InputMatrix2 = virtualEval(MATfileName,Expression2,varList);
        [nrow1,ncol1] = size(InputMatrix1);
        [nrow2,ncol2] = size(InputMatrix2);
        flopAdd = max(nrow1*ncol1,nrow2*ncol2);
        flopCounter = flopCounter + flopAdd;
        flopDetails(size(flopDetails,1)+1,:) = {funName,flopAdd};
        return
    end    
end

% Mainly support functions with only one input arguments
% Limited support to sum(A,2), etc., but the algorithm is not perfect
secondDim = ~isempty(strfind(funInputArg,',2')) || ~isempty(strfind(funInputArg,', 2'));
locComma = strfind(funInputArg,',');
if ~isempty(locComma)
    funInputArgCut = funInputArg(1:locComma(1)-1);
else
    funInputArgCut = funInputArg;
end
    
% Evaluate the input argument, get the size the input matrix
% because the inputs could still be a math expression
InputMatrix = virtualEval(MATfileName,funInputArgCut,varList);
if ~secondDim
    [nrow,ncol] = size(InputMatrix);
else
    [nrow,ncol] = size(InputMatrix');
end

flopAdd = Rules(funName,nrow,ncol);
if flopAdd > 0
    flopCounter = flopCounter + flopAdd;
    flopDetails(size(flopDetails,1)+1,:) = {funName,flopAdd};    
end

end


%-------------------------------------------------------------------------
% Replace the transposed variable by an intermediate variable
function [txt,varList] = transpose2var(txt,varList,MATfileName)

tranSign = strfind(txt,'''');
if isempty(tranSign)
    return
end

nvarExist = size(varList,1);
addNewVar = 1;

for m = length(tranSign):-1:1
    locEnd = tranSign(m)-1;
    if ~isLegalName(txt(locEnd))
        continue
    end
    locStart = tranSign(m)-1;
    for n = locEnd:-1:1
        if ~isLegalName(txt(n))
            locStart = n + 1;
            break
        end        
    end
    varName = txt(locStart:locEnd);
    mask = strcmp(varList(:,1),varName);   
    
    if ~isempty(varName) && any(mask)
        GenVarName = ['GenVar',num2str(nvarExist+addNewVar)];
        txt = [txt(1:locStart-1),GenVarName,txt(locEnd+2:end)];
        varList{nvarExist+addNewVar,1} = GenVarName;        
        [nrow,ncol] = getVarSize(varName,varList);
        varList{nvarExist+addNewVar,2} = [ncol,nrow];
        varList{nvarExist+addNewVar,3} = virtualEval(MATfileName,varName,varList)';        
        addNewVar = addNewVar + 1;
    end    
end

end


%-------------------------------------------------------------------------
function [marker,numPairs,LeftLoc,RightLoc] = bracket(txt,bracketType)
%bracket Identify the level and location of brackets in a string
%
% Syntax:
%
%   marker = bracket(txt)
%   [marker,numPairs,LeftLoc,RightLoc] = bracket(txt,bracketType)
%
% Description:
%
%   Identify the level and location of small brackets in a string
%
% Input Arguments:
%
%   txt - A 1-by-n string of a math expression
%
%   bracketType - Type of the bracket
%                 1 = parentheses ()
%                 2 = bracket []
%                 3 = curly bracket {}
%                 4 = angle bracket <>
%                 The default is 1
%
% Output Arguments:
%
%   marker - A 1-by-n marker.
%            0 = not a bracket
%            1 = first level (outer) left bracket
%           -1 = first level (outer) right bracket
%            2 = second level (inner) left bracket
%           -2 = second level (inner) right bracket
%           Similarly, 3,-3, 4,-4 are higher level brackets
%
%  numPairs - A numLevel-by-1 vector summarizing the number of bracket pairs
%            for each level.
%
%  LeftLoc - A matrix summarizing the level and location of left brackets
%            The ith row is for the ith level brackets
%            The jth element of the ith row shows the location of the jth left bracket of the ith level
%            LeftLoc is a numLevel-by-numPairsMax matrix with padding NaNs
%
%  RightLoc - A matrix summarizing the level and location of right brackets
%            The ith row is for the ith level brackets
%            The jth element of the ith row shows the location of the jth right bracket of the ith level
%            RightLoc is a numLevel-by-numPairsMax matrix with padding NaNs
%
% Notes:
%
% An example:
% txt  =   '3+(rand(5,5)+7*zeros(5))*8'
%            3 + ( r a n d ( 5 , 5  ) + 7 * z e r o s ( 5  )  ) * 8
% marker  = [0 0 1 0 0 0 0 2 0 0 0 -2 0 0 0 0 0 0 0 0 2 0 -2 -1 0 0]
% numPairs = [1;2], which indicates that
%         There is 1 outer bracket pair
%         There are 2 inner bracket pairs
% LeftLoc = [3 NaN; 8 21], which indicates that
%         The location of the outer left bracket is 3
%         The location of the first inner left bracket is 8, the second 21
% RightLoc = [24 NaN; 12 23], which indicates that
%         The location of the outer right bracket is 24
%         The location of the first inner right bracket is 12, the second 23
%
% Written by Hang Qian
% Contact: matlabist@gmail.com

if ~ischar(txt)
    error('The input must be a string.')
end

if nargin < 2
    bracketType = 1;
end

switch bracketType
    case 1
        leftBracket = '(';
        rightBracket = ')';
    case 2
        leftBracket = '[';
        rightBracket = ']';
    case 3
        leftBracket = '{';
        rightBracket = '}';
    case 4
        leftBracket = '<';
        rightBracket = '>';
    otherwise
        error('Not a supported bracket type.')
end

% Identify the bracket marker
nword = length(txt);
marker = zeros(1,nword);
count = 0;
for m = 1:nword
    if strcmp(txt(m),leftBracket)
        count = count + 1;
        marker(m) = count;
    elseif strcmp(txt(m),rightBracket)
        marker(m) = -count;
        count = count - 1;        
    end
end

if count ~= 0
    error('Unbalanced bracket.')
end

% Summarize the location of the marker
if nargout > 1
    nlevel = max(marker);
    numPairs = zeros(nlevel,1);
    for m = 1:nlevel
        numPairs(m) = sum(marker == m);
    end
    npairMax = max(numPairs);
    LeftLoc = NaN(nlevel,npairMax);
    RightLoc = NaN(nlevel,npairMax);
    for m = 1:nlevel
        LeftLoc(m,1:numPairs(m)) = find(marker == m);
        RightLoc(m,1:numPairs(m)) = find(marker == -m);
    end
end
end

%------------------------------------------------------------------------
% Count FLOPS for arithmetic operation, statistics and elementary functions
function count = Rules(name,nrow1,ncol1,nrow2,ncol2,NewRule)

%Rules: Rules of counting FLOPS
%
% Syntax:
%   Rules(name,nrow,ncol)
%   Rules(name,nrow1,ncol1,nrow2,ncol2)
%   Rules(name,nrow1,ncol1,nrow2,ncol2,NewRule)
%
% Input Arguments
%
% name: Name of an arithmetic operator or function
% nrow1: Number of rows of the first input argument
% ncol1: Number of columns of the first input argument
% nrow2: Number of rows of the second input argument (arithmetic operator only)
% ncol2: Number of columns of the second input argument (arithmetic operator only)
% NewRule: User supplied rules for counting FLOPS
%
% Output Arguments
%
% count: FLOPS count
%

if nargin < 3
    error('An arithmetic operator or math function name, and sizes of the inputs must be provided.')
end

if nargin < 6
    NewRule = cell(0,2);
end

% Support both nrow, ncol and nrow1, ncol1
nrow = nrow1;
ncol = ncol1;

% User supplied new rules for arithmetic operations in ExtendedRules.xlsx
if ~isempty(NewRule)    
    for m = 1:size(NewRule,1)
        if strcmp(name,NewRule{m,1})            
            try
                count = round(eval(NewRule{m,2}));
            catch
                count = 0;
                warning('User supplied rules %s cannot be evaluated.',NewRule{m,1})
            end
            return
        end
    end    
end
    
% Internal rules for arithmetic operations
switch name
    
    case '+'
        
        count = max(nrow1*ncol1, nrow2*ncol2);
        
    case '-'
        
        count = max(nrow1*ncol1, nrow2*ncol2);
        
    case '*'
        
        if (nrow1==1 && ncol1==1) || (nrow2==1 && ncol2==1)
            count = max(nrow1*ncol1, nrow2*ncol2);
        else
            count = 2*nrow1*ncol1*ncol2;
        end
        
    case '.*'
        
        count = max(nrow1*ncol1, nrow2*ncol2);
        
    case '/'
        
        if nrow2==1 && ncol2==1            
            count = max(nrow1*ncol1, nrow2*ncol2);
        else            
            count = round(2/3*nrow2^3) + 2*nrow2^2*nrow1;
        end
        
    case './'
        
        count = max(nrow1*ncol1, nrow2*ncol2);
        
    case '\'
        
        if nrow1==1 && ncol1==1
            % Scalar right division
            count = max(nrow1*ncol1, nrow2*ncol2);
        elseif nrow1 == ncol1
            % Solving equations Ax=b
            count = round(2/3*ncol1^3) + 2*ncol1^2*ncol2;
        else
            % OLS (X'*X)\(X'*Y)
            count = 2*ncol1*nrow1*ncol1 + 2*ncol1*nrow1*ncol2 + round(2/3*ncol1^3) + 2*ncol1^2*ncol2;
        end        
        
    case '.\'
        
        count = max(nrow1*ncol1, nrow2*ncol2);
        
    case '^'
        
        temp1 = dec2bin(ncol2);
        temp2 = length(temp1) + sum(temp1=='1') - 1;
        count = 2*nrow1^3*temp2; 
        
    case '.^'
        
        count = 2 .* max(nrow1*ncol1, nrow2*ncol2);
        
    case {'>', '>=','<', '<=', '==','~='}
        
        count = 0;
        
    case {'sum','prod','cumsum','cumprod'}
        
        count = nrow*ncol;    
        
    case 'mean'
        
        count = (nrow+1)*ncol;
        
    case 'var'
        
        count = 4*nrow*ncol;
        
    case 'std'
        
        count = 4*nrow*ncol;
        
    case 'cov'
        
        count = 2*nrow*ncol*(ncol+1);
        
    case 'corr'
        
        count = 2*nrow*ncol*(ncol+1);
        
    case 'diff'
        
        count = (nrow-1)*ncol;
        
    case {'log','log10','log2','reallog','exp','sqrt','sin','cos','tan','asin','acos','atan'}
        
        count = nrow*ncol;
        
    case 'chol'
        
        count = round(nrow^3/3 + nrow^2/2 + nrow/6); 
        
    case 'lu'
        
        count = round(2/3*nrow^3); 
        
    case 'qr'
        
        count = round(2*nrow*ncol^2);
        
    case 'svd'
        
        count = round(2*nrow*ncol^2 + 2*ncol^3);
        
    case 'inv'
        
        % It is believed that inversion takes round(2/3*nrow^3) FLOPS; 
        count = round(2*nrow^3); 
        
    case 'det'
        
        count = round(2/3*nrow^3); 
        
    otherwise
        
        count = 0;
        
end   

end

