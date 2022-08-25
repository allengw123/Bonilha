clear all
clc

% This code is used to process DTI
%%% Used to process Rebecca's MASTER_EPILEPSY DATABASE


%%%%%%%%%%%%%%%%%%%%%%%%%% Requirments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clone of Allen Chang's Github (https://github.com/allengw123/Bonilha)
% BrainAgeR --> Requires BrainAgeR setup (https://github.com/james-cole/brainageR)
% MATLAB's parallel processing toolbox 
% Developed using MATLAB vR2022 (likely need that version or higher)
% Nii_stat preprocessing configued computer
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


%%%%%%%%%%%%%%%%%%%%%% ADVANCE OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt.SKIP_PROBLEM_TAG = true; %true = don't preprocess subjects with a "problem" string attached at the end
opt.PROBLEM_TAGS = {'problem','MissingT1','MissingLesion'}; % ends with tags that SKIP_PROBLEM_TAG will look for
opt.RECHECK_ALREADY_FORMATED = false; % Rechecks formated subjects
opt.HYPER_THREAD = true; % MAY CAUSE OVERHEATING. Since moving a lot of files, CPUs will benefit from hyperthreading

opt.setOrigin = true; %attempt to crop and set anterior commissure for images
opt.isExitAfterTable = false; % <- if true, only generates table, does not process data
opt.isPreprocess = true; % <- if true full processing, otherwise just cropping
opt.isReportDims = true; %if true, report dimensions of raw data
opt.reprocessRest = false;
opt.reprocessfMRI = false;
opt.reprocessASL = false;
opt.reprocessDTI = false;
opt.reprocessVBM = false;
opt.explicitProcess = false; % <- if true, will only process if the reprocess flag is true
opt.interweave = true; % Subjects will get dedicated to which parallel worker in an interweave fashion (helps efficiently process new subjects to database)

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



%% Preprocess DTI
% Create Harvest Output Folder
mkdir(harvest_output)

% Run nii_harvest (DTI)
errors = nii_harvest(nii_preproc_database,harvest_output,opt);
