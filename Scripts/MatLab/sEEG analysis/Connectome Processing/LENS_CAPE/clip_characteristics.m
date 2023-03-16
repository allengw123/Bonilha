%% Add correct paths
clear all
close all
clc

gitpath='/home/bonilha/Documents/GitHub/Bonilha';
cd(gitpath)
allengit_genpath(gitpath,'EEG')

%% Subject/Trial info
% Define info
trialnames={'Baseline','Pre-transition','Post-transition','Mid Sezuire','Late Sezuire','Early Post','Late Post'};

datadir='/media/bonilha/AllenProj/sEEG_project/PatientData/CAPES_LEN/';


master_electrode={'LA','LAH','LAI','LLF','LMF','LPH','LPI','RA','RAH','RAI','RLF','RMF','RPH','RPI'};

%% Connectivity

% Reference sheet
ref_sheet = load(fullfile('/media/bonilha/AllenProj/sEEG_project/PatientData','outcome.mat')).ref_sheet;

% Find subjects
subjID = [dir(fullfile('/media/bonilha/AllenProj/sEEG_project/PatientData','*','Patient*'));dir(fullfile('/media/bonilha/AllenProj/sEEG_project/PatientData','*','3T*'))];
response = [];
tab = [];
for s = 1:numel(subjID)
    wk_sbj = subjID(s).name;
    wk_res = regexp(ref_sheet.ILAElatest{contains(ref_sheet.PreviousIDs,wk_sbj)},'\d*','Match');
    tab = [tab;ref_sheet(contains(ref_sheet.PreviousIDs,wk_sbj),:)];
    if isempty(wk_res)
        wk_res = -1;
    else
        wk_res = str2double(wk_res{1});
    end
    response = [response;wk_res];
end
responsive_idx = response==1;
nonresponsive_idx = response>1;
tab = tab(nonresponsive_idx|responsive_idx,:);


subjID = {dir(fullfile(datadir,'3T*')).name}';
sez_length = [];
targets = [];

for subj=[2 3 6 8]

    % Dedicate working subject paths
    wk_subject_dir = fullfile(datadir,subjID{subj});
    wk_ieeg_dir = fullfile(wk_subject_dir,'sEEG');
    save_folder = fullfile(wk_ieeg_dir,'matdata');
    mkdir(save_folder)

    % Find seizure files
    seizureEDF= dir(fullfile(wk_ieeg_dir,'*seizure*.set'));

    wk_sez_length = [];
    for sez=1:size(seizureEDF,1)

        % Seizure name
        wk_sez_name = extractBetween(seizureEDF(sez).name,'task-','_ieeg');
        wk_sez_name = wk_sez_name{:};

        %%%%%%%%%%%%%%%%%%%%%%%%%%%% Sezuire CUSTOM EPOCH  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Find event times
        eeglab = load('-mat',fullfile(seizureEDF(sez).folder,seizureEDF(sez).name));
        on_idx = find(contains({eeglab.event.type},'ONSET'));
        off_idx = find(contains({eeglab.event.type},'END'));
        fs = eeglab.srate;
        
        if isempty(on_idx) | isempty(off_idx)
            continue
        end

        onset = eeglab.event(on_idx(end)).latency;
        offset = eeglab.event(off_idx(end)).latency;

        
        prev_offset = [];
        if numel(off_idx)>1
            prev_offset = eeglab.event(off_idx(end-1)).latency;
        end


        % check to see if there was overlap with previous seizure
        if ~isempty(prev_offset)
            if (onset-(30*fs)) < prev_offset
                continue
            end
        end
       
        wk_sez_length = [wk_sez_length;offset - onset];
    end
    
    sez_length = [sez_length;mean(wk_sez_length)/fs];
    
    wk_targets = regexp({eeglab.chanlocs.labels},'\D','match');
    wk_targets = cellfun(@(x) [x{:}],wk_targets,'UniformOutput',false);
    wk_targets = unique(wk_targets);
    
    wk_targets = wk_targets(cellfun(@(x) any(strcmp(x,master_electrode)),wk_targets));

    targets = [targets; {wk_targets}];
end
