clear all
close all


GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
FREESURFER_HOME = '/usr/local/freesurfer/7.3.2';
IMAGE_DATABASE = '/media/bonilha/Elements/Image_database';
DATABASE_NAME = 'MasterSet_TLE';
%DATABASE_NAME = 'UCSD_TLE';
%DATABASE_NAME = 'ADNI_AD';
DISEASE_TAG = 'Patients';
%DISEASE_TAG = 'Controls';

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

%% Run FreeSurfer

% Define paths
database_dir = fullfile(IMAGE_DATABASE,DATABASE_NAME);
harvest_output = fullfile(database_dir,'harvestOutput',DISEASE_TAG);

% Find subjects
subjects = dir(harvest_output);
subjects(contains({subjects.name},'.'),:) = [];

% Create Freesurfer folder
fs_output = fullfile(database_dir,'freesurfer_output',DISEASE_TAG);
mkdir(fs_output)

% Run Freesurfer
err = cell(numel(subjects),1);
pool = setpool(4);
parfor sbj = 1:numel(subjects)

    % Find subject
    wk_subject_name = subjects(sbj).name;
    wk_subject_folder = fullfile(subjects(sbj).folder,wk_subject_name);

    
    % Find sesions
    sessions = dir(wk_subject_folder);
    sessions(contains({sessions.name},'.')) = [];

    for sess = 1:numel(sessions)
        
        % Find session
        session_name = sessions(sess).name;
        session_folder = sessions(sess).folder;

        % Create Freesurfer output
        session_fs_output = fullfile(fs_output,wk_subject_name,session_name);
        mkdir(session_fs_output)

        % Transfer T1
        t1 = dir(fullfile(wk_subject_folder,session_name,'eT1*.nii'));
        if isempty(t1)
            t1 = dir(fullfile(wk_subject_folder,session_name,'T1*.nii'));
        end
        %fs_t1 = fullfile(session_fs_output,strrep(t1.name,['_',session_name],''));
        fs_t1 = fullfile(session_fs_output,t1.name);
        copyfile(fullfile(t1.folder,t1.name),fs_t1)

        % Run recon-all
        command = sprintf('recon-all -all -i %s -s %s -sd %s',fs_t1,wk_subject_name,session_fs_output);
        status = system(command);
        if status ~= 0
            err{sbj}{sess} = sprintf('recon failed on %s',session_name)
            continue
        end
        
        % Run Hippocampal Subfield
        % (https://surfer.nmr.mgh.harvard.edu/fswiki/HippocampalSubfieldsAndNucleiOfAmygdala)
        segment_subregions_path = fullfile(FREESURFER_HOME,'bin','');
        command = sprintf('segmentHA_T1.sh %s %s',wk_subject_name,session_fs_output);
        status = system(command);
        if status ~=0
            err{sbj}{sess} = sprintf('hippocampal subfield failed on %s',session_name)
        end
    end
end

