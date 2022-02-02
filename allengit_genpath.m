function allengit_genpath(gitpath,type)

addpath(genpath(fullfile(gitpath,'Scripts')))

if nargin<2
    type='none';
    disp('No specific analysis argument detected. No additional toolboxes will be added')
end

addpath(genpath(fullfile(gitpath,'Functions')));

if strcmp(type,'imaging')
    addpath(genpath(fullfile(gitpath,'Toolbox','imaging')))
    
    % Don't add all spm subfloders
    spm_rmpath
    addpath(fullfile(gitpath,'Toolbox','imaging','spm12'))
    spm
    close all
    clc
elseif strcmp(type,'EEG')
    disp('Adding EEG Toolboxes')
    addpath(genpath(fullfile(gitpath,'Toolbox','EEG')))   

    % EEGlab
    rmpath(genpath(fullfile(gitpath,'Toolbox','EEG','eeglab2021.1')))
    addpath(fullfile(gitpath,'Toolbox','EEG','eeglab2021.1'))
    eeglab
    close all
    
    %%% Manually add biosig plugin, automatic addition from eeglab DOES NOT
    %%% WORK?!
    cd(dir(which('biosig_installer')).folder);
    biosig_installer
    cd(gitpath)
    
    % FieldTrip
    rmpath(genpath(fullfile(gitpath,'Toolbox','EEG','fieldtrip-20200607')))
    addpath(fullfile(gitpath,'Toolbox','EEG','fieldtrip-20200607'))
    ft_defaults
    addpath(fullfile(gitpath,'Toolbox','EEG','fieldtrip-20200607','external','spm12'))
    addpath(fullfile(gitpath,'Toolbox','EEG','fieldtrip-20200607','external','bsmart'))
end

end