clear all
clc

% This code is used as a Quality Check for MASTER_EPILEPSY pipeline
%%% Used to process Rebecca's MASTER_EPILEPSY DATABASE


%%%%%%%%%%%%%%%%%%%%%%%%%% Requirments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clone of Allen Chang's Github (https://github.com/allengw123/Bonilha)
% Developed using MATLAB vR2022 (likely need that version or higher)

% Input folder with the following structure:
%%% INPUT_PATH
%%%% DISEASE_TAG
%%%%% SUBJECT NAME
%%%%%% CORRECTLY FORMATED FILES (SUBJECTNAME_SESSION_AQUISITIONTYPE.nii+.json)

%%%%%%%%%%%%%%%%%%%%%%% ALTER VARIABLES BELOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
INPUT_PATH = '/media/bonilha/Elements/Master_Epilepsy_Database_SYNC';
OUTPUT_PATH = '/media/bonilha/Elements/MasterSet';
DISEASE_TAG = 'Patients';

%%%%%%%%%%%%%%%%%%%%%%%% ADVANCE OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CHECK_SESSIONMATCH = true;
CHECK_AQ = true;
CHECK_BRAINAGER = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% DON'T CHANGE CODE BELOW (unless you know what you are doing) %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Perform QC


% Set paths
nii_proc_formated = fullfile(OUTPUT_PATH,'nii_proc_format');
processed_dir = fullfile(OUTPUT_PATH,'processed');

% Find processed matfiles
subjects = dir(fullfile(processed_dir,DISEASE_TAG,'**','*.mat'));
subject_folder = subjects.folder;
subjects = {subjects.name};

% Find Input directory to check against
checkdir = fullfile(nii_proc_formated,DISEASE_TAG);

% Post QC folder
post_qc = fullfile(OUTPUT_PATH,'post_qc',DISEASE_TAG);
mkdir(post_qc)

% Pipeline Info
pipelineinfo.author = 'Allen Chang';
pipelineinfo.toolboxes = [{'Nii_preproc (Allen edited) adapted from https://github.com/neurolabusc/nii_preprocess'};
    {'brainageR (software version 2.1 Oct 2020; https://github.com/james-cole/brainageR)'};
    {'brainageR Specific SPM'}];
pipelineinfo.date = date;

parfor s = 1:numel(subjects)
    warning('off','all')

    % Define subject name
    wk_sbj = extractBefore(subjects{s},'.mat');

    % Load processed matfile
    wk_mat = load(fullfile(subject_folder,subjects{s}));

    % Allocated QC variable
    qc = [];

    % Check session matches
    input_ses = dir(fullfile(nii_proc_formated,DISEASE_TAG,wk_sbj));
    input_ses(contains({input_ses.name},'.')) = [];
    wk_sesions = input_ses([input_ses.isdir]);
    if CHECK_SESSIONMATCH
        if numel(fieldnames(wk_mat)) == numel(wk_sesions)
            qc.ses_match = 'pass';
        else
            qc.ses_match = 'fail';
            disp([wk_sbj,' SESSION MATCH FAIL'])
            continue
        end
    end

    % Check each session
    cont = true;
    for ses = 1:numel(input_ses)

        % Define working session
        wk_ses = input_ses(ses);
        in_ses_folder = fullfile(wk_ses.folder,wk_ses.name);
        ses_mat = wk_mat.(wk_ses.name);

        % Find Aquistions
        aq = dir(fullfile(in_ses_folder,'*.nii'));

        for a = 1:numel(aq)
            
            % Define working aquisition
            wk_aq = aq(a);
            aq_name = extractBefore(wk_aq.name,'_');

            fn = fieldnames(ses_mat);

            if CHECK_AQ
                % Check output of each aquisition
                switch aq_name
                    case 'Lesion'
                        if sum(contains(fn,'lesion')) > 1
                            qc.lesion = 'pass';
                        else
                            qc.lesions = 'fail';
                            disp([wk_sbj,' ',wk_ses.name,' Lesion FAIL'])
                            cont = false;
                        end
                    case 'T1'
                        if any(contains(fn,'T1')) && any(contains(fn,'vbm')) && any(contains(fn,'VBM_'))
                            qc.T1 = 'pass';
                        else
                            qc.T1 = 'fail';
                            disp([wk_sbj,' ',wk_ses.name,' T1 FAIL'])
                            cont = false;
                        end
                    case 'T2'
                        continue
                    case 'Rest'
                        if any(contains(fn,'RestAve')) && any(contains(fn,'rest_'))
                            qc.Rest = 'pass';
                        else
                            qc.Rest = 'fail';
                            disp([wk_sbj,' ',wk_ses.name,' Rest FAIL'])
                            cont = false;
                        end
                    case 'DTI'
                        if any(contains(fn,'dti')) && any(contains(fn,'md')) && any(contains(fn,'fa'))
                            qc.Rest = 'pass';
                        else
                            qc.Rest = 'fail';
                            disp([wk_sbj,' ',wk_ses.name,' DTI FAIL'])
                            cont = false;
                        end
                end
            end
        end

        % Check brainage
        if CHECK_BRAINAGER
            if any(contains(fn,'brainage'))
                b_fn = fieldnames(wk_mat.(wk_ses.name).brainage);
                if any(contains(b_fn,'agePred')) && any(contains(b_fn,'TissueVol')) && any(contains(b_fn,'slicedir'))
                    qc.brainageR = 'pass';
                else
                    qc.brainageR = 'fail';
                    disp([wk_sbj,' ',wk_ses.name,' brainage FAIL'])
                    cont = false;
                end
            else
                qc.brainageR = 'fail';
                disp([wk_sbj,' ',wk_ses.name,' brainage FAIL'])
                cont = false;
            end
        end

        % Save QC structure
        wk_mat.(wk_ses.name).QC = qc;
        qc = [];
    end
    
    % Save QC structure back to matfile
    if cont
        % Add pipeline info
        wk_mat.pipelineinfo = pipelineinfo;

        saveparfor(fullfile(post_qc,subjects{s}),'-struct',wk_mat)
    end
end

%% Functions
function saveparfor(outname,opt1,save_mfile)
save(outname,opt1,'save_mfile')
end
%end saveparfor()
