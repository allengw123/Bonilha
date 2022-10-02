function textprogressbar(order,c,msg)
% This function creates a text progress bar. It should be called with a 
% STRING argument to initialize and terminate. Otherwise the number correspoding 
% to progress in % should be supplied.
% INPUTS:   
%           order       0 = start bar, 1 = progress bar, 2 = terminate bar
%           C   Either: Text string to initialize or terminate 
%                       Percentage number to show progress 
% OUTPUTS:  N/A
% Example:  Please refer to demo_textprogressbar.m

% Author: Paul Proteus (e-mail: proteus.paul (at) yahoo (dot) com)
% Version: 1.0
% Changes tracker:  29.06.2010  - First version

% Inspired by: http://blogs.mathworks.com/loren/2007/08/01/monitoring-progress-of-a-calculation/

%% Initialization

persistent strCR;           %   Carriage return pesistent variable
persistent count;
% Vizualization parameters
strPercentageLength = 10;   %   Length of percentage string (must be >5)
strDotsMaximum      = 10;   %   The total number of dots in a progress bar

if ~exist('msg','var')
    msg = [];
end

%% Main 

if order == 0
    % Progress bar - initialization
    fprintf('%s\n',c);
    strCR = -1;
    count = 0;
elseif order == 1
    % Progress bar - normal progress
    if rem(count, 2) == 0
        switchchar = 'X';
    else
        switchchar = '+';
    end
    c = floor(c);
    percentageOut = [num2str(c) '%%'];
    percentageOut = [percentageOut repmat(' ',1,strPercentageLength-length(percentageOut)-1)];
    nDots = floor(c/100*strDotsMaximum);
    dotOut = ['[' repmat('*',1,nDots) switchchar repmat(' ',1,strDotsMaximum-nDots-1) '] --> '];
    strOut = [percentageOut dotOut msg];
    
    % Print it on the screen
    if strCR == -1
        % Don't do carriage return during first run
        fprintf(strOut);
    else
        % Do it during all the other runs
        fprintf([strCR strOut]);
    end
    
    % Update carriage return
    strCR = repmat('\b',1,length(strOut)-1);
    count = count+1;
    
elseif order == 2
    % Progress bar  - termination
    strCR = [];
    count = 0;
    fprintf(['\n' c '\n']);
else
    % Any other unexpected input
    error('Order variable must either 0,1,2');
end
