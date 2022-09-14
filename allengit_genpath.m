function allengit_genpath(gitpath,type)

restoredefaultpath

addpath(genpath(fullfile(gitpath,'Scripts')))
addpath(genpath(fullfile(gitpath,'Functions')));

if nargin<2
    type='none';
end

switch type
    case 'imaging'
        addpath(genpath(fullfile(gitpath,'Toolbox','imaging')))
    
        % Remove brainageR spm
        rmpath(genpath(fullfile(gitpath,'Toolbox','imaging','brainageR_SPM')))
        
        % Don't add all spm subfloders
        spm_rmpath
        addpath(fullfile(gitpath,'Toolbox','imaging','spm12'))
        spm
        close all
        clc
    
    case 'EEG'
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
    case 'brainageR'
        addpath(genpath(fullfile(gitpath,'Toolbox','imaging')))
    
        % Remove all spm12 paths
        rmpath(genpath(fullfile(gitpath,'Toolbox','imaging','spm12')))
        rmpath(genpath(fullfile(gitpath,'Toolbox','imaging','brainageR_SPM')))
        
        % Don't add all brainageR spm subfloders
        addpath(fullfile(gitpath,'Toolbox','imaging','brainageR_SPM'))
        spm
        close all
        clc
        
    case 'none'
        disp('No specific analysis argument detected. No additional toolboxes will be added')
end

end