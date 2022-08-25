clear all
clc

% This code is used to process calculate brainage
%%% Used to process Rebecca's MASTER_EPILEPSY DATABASE


%%%%%%%%%%%%%%%%%%%%%%%%%% Requirments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clone of Allen Chang's Github (https://github.com/allengw123/Bonilha)
% BrainAgeR --> Requires BrainAgeR setup (https://github.com/james-cole/brainageR)
% Developed using MATLAB vR2022 (likely need that version or higher)
% Input folder with the following structure:one volume
%%% INPUT_PATH
%%%% DISEASE_TAG
%%%%% SUBJECT NAME
%%%%%% CORRECTLY FORMATED FILES (SUBJECTNAME_SESSION_AQUISITIONTYPE.nii+.json)

%%%%%%%%%%%%%%%%%%%%%%% ALTER VARIABLES BELOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
INPUT_PATH = '/media/bonilha/Elements/Master_Epilepsy_Database_SYNC';
OUTPUT_PATH = '/media/bonilha/Elements/MasterSet';
DISEASE_TAG = 'Patients';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% DON'T CHANGE CODE BELOW (unless you know what you are doing) %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Find Database paths
raw_database = fullfile(INPUT_PATH,DISEASE_TAG);
if ~exist(raw_database,'dir')
    error([DISEASE_TAG,' Folder Not Found. Format is : path_to_master_set_folder/<patients variable name>'])
end
nii_preproc_database = fullfile(OUTPUT_PATH,'nii_proc_format',DISEASE_TAG);
harvest_output = fullfile(OUTPUT_PATH,'harvestOutput',DISEASE_TAG);
processed_output = fullfile(OUTPUT_PATH,'processed',DISEASE_TAG);

%% Brain Age

disp('Step 3 - Running Brain Age')

% Assign brainageR Path
brainage_path = fileparts(which('spm_preprocess_brainageR.m'));

% Assign brainageR-specific SPM path
spm_path = fullfile(GITHUB_PATH,'Toolbox/imaging/brainageR_SPM/');

% Prep Brain Age Files
setup_brainagedir(brainage_path,spm_path)

% Detect Nii_Harvested Subjects
processed_matfiles = dir(fullfile(processed_output,'**','*.mat'));

% Run BrainAge
for a = 1:numel(processed_matfiles)

    % Create temp brainage folder
    tempbrain_folder = fullfile(OUTPUT_PATH,'brainageTEMP');
    mkdir(tempbrain_folder)

    % Load processed matfile
    mat_folder = processed_matfiles.folder;
    wk_mat = load(fullfile(mat_folder,processed_matfiles(a).name));
    
    % Find T1 scans/sessions
    sbj_name = extractBefore(processed_matfiles(a).name,'.mat');
    input = dir(fullfile(OUTPUT_PATH,'nii_proc_format',DISEASE_TAG,sbj_name,'**','T1*'));

    for s = 1:numel(input)

        % Define session name
        [~,ses] = fileparts(input(s).folder);

        % Skip if brainage already completed
        if isfield(wk_mat.(ses),'brainage')
            continue
        end

        % Copy T1 into brainage temp folder
        s_input = fullfile(tempbrain_folder,input(s).name);
        copyfile(fullfile(input(s).folder,input(s).name),s_input);

        % Run brainageR
        cmd = sprintf('%s -f %s -o %s',fullfile(brainage_path,'brainageR'),s_input,fullfile(tempbrain_folder,'brainage.csv'));
        system(cmd)

        % Save brainageR output
        brain_age_pred = readtable(fullfile(tempbrain_folder,'brainage.csv'));
        brain_age_pred_tissue_vols = readtable(fullfile(tempbrain_folder,[brain_age_pred.File{:},'_tissue_volumes.csv']));
        wk_mat.(ses).brainage.agePred = brain_age_pred;
        wk_mat.(ses).brainage.TissueVol = brain_age_pred_tissue_vols;

        slice_dir = dir(fullfile(tempbrain_folder,['slicesdir_T1_',extractBefore(processed_matfiles(a).name,'.mat'),'.nii'],'*.png'));
        for p = 1:numel(slice_dir)
            png = imread(fullfile(slice_dir(p).folder,slice_dir(p).name));
            t_name = extractBetween(slice_dir(p).name,'TEMP__','.png');
            wk_mat.(ses).brainage.slicedir.(t_name{:}) = png;
        end
    end

    % Save completed matfile with brainageR field
    save(fullfile(mat_folder,processed_matfiles(a).name),'-struct','wk_mat')

    % Remove tempbrainage folder
    rmdir(tempbrain_folder,'s')
end
disp('Brain Age Complete')

%% Functions

function setup_brainagedir(brainage_path,spm_path)

% Detect brainage SH file
brainage_file_path = fullfile(brainage_path,'brainageR');

% Read brainage SH file
brainage_file = fopen(brainage_file_path,'r');
eof = false;
fileContents = [];
while ~eof
    str = fgetl(brainage_file);
    if str==-1
        eof = true;
    else
        fileContents = [fileContents; {str}];
    end
end
fclose(brainage_file);

% Replace bin/bash interpreter
fileContents{1} = '#!/usr/bin/bash';

% Replace brainageR_dir path
idx = find(~cellfun(@isempty,(regexp(fileContents,'brainageR_dir'))));
brainageR_dir_line = sprintf('brainageR_dir=%s',[fileparts(brainage_path),filesep]);
fileContents{idx(1)} = brainageR_dir_line;

% Replace spm_dir path
idx = find(~cellfun(@isempty,(regexp(fileContents,'spm_dir'))));
spm_path_line = sprintf('spm_dir=%s',[spm_path,filesep]);
fileContents{idx(1)} = spm_path_line;

% Replace matlab_path path
idx = find(~cellfun(@isempty,(regexp(fileContents,'matlab_path'))));
matlab_path_line = sprintf('matlab_path=%s',fullfile(matlabroot,'bin','matlab'));
fileContents{idx(1)} = matlab_path_line;

% write/replace brainageR sh file
brainage_file = fopen(brainage_file_path,'w');
fprintf(brainage_file,'%s\n',string(fileContents));
fclose(brainage_file);
end