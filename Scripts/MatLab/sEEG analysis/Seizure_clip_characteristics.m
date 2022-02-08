%% Add correct paths
clear all
close all
clc

gitpath='C:\Users\allen\Documents\GitHub\Bonilha';
cd(gitpath)
allengit_genpath(gitpath,'EEG')
%% Subject/Trial info

datadir='C:\Users\allen\Box Sync\Desktop\Bonilha\Projects\sEEG project\PatientData';
cd(datadir)
subjID = {dir(fullfile(datadir,'Patient *')).name};
subjnum = regexp(subjID,'\d*','Match');

%% Calculate # of seizure and length
info=[];
for p=1:numel(subjID)
    
    % Find seizure clip
    sClips_dir=dir(fullfile(datadir,subjID{p},'sEEG','*.edf'));
    sClips={sClips_dir.name};
    sClips=sClips(cellfun(@isempty,regexp(sClips,'.*Baseline.edf','match')));
    
    % Load seizure clips
    avgSTime=[];
    for c=1:numel(sClips)
        cfg = [];
        cfg.dataset     = fullfile(sClips_dir(1).folder,sClips{c});
        cfg.continuous  = 'yes';
        
        % Load EDF
        data_eeg        = ft_preprocessing(cfg);
        
        % Save time
        avgSTime=[avgSTime data_eeg.time{1, 1}(end)];
    end
    
    % Save info
    info{p,1}=subjID{p};
    info{p,2}=numel(sClips);
    info{p,3}=mean(avgSTime);
end
    
