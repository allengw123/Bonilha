clear all
clc

% This code is used to process T1, Lesion, and RestingState MRIs, DTI, and
% BrainAGE
%%% Unable to do DTI in parallel --> use nii_harvest to do DTI serially
%%% Used to process Rebecca's MASTER_EPILEPSY DATABASE


%%%%%%%%%%%%%%%%%%%%%%%%%% Requirments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clone of Allen Chang's Github (https://github.com/allengw123/Bonilha)
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
IMAGE_DATABASE = '/media/bonilha/Elements/Image_database';
DATABASE_NAME = 'MasterSet_TLE';
%DATABASE_NAME = 'UCSD_TLE';
%DATABASE_NAME = 'ADNI_AD';
DISEASE_TAG = 'Patients';
%DISEASE_TAG = 'Controls';

    
%%%%%%%%%%%%%%%%%%%%%%%%% ADVANCE OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Database options
opt.SESSION_TAG = {'pre','pos','session'};

% Format Options
opt.SKIP_PROBLEM_TAG = true; %true = don't preprocess subjects with a "problem" string attached at the end
opt.PROBLEM_TAGS = {'problem','MissingT1','MissingLesion','IncorrectLesion'}; % ends with tags that SKIP_PROBLEM_TAG will look for
opt.SKIP_LESION = true; % Set true to skip Missinglesion and IncorrectLesion based ond PROBLEM_TAGS_LOGIC
opt.PROBLEM_TAGS_LOGIC = {false,false,true,true}; % Logic for SKIP_LESION option
opt.RECHECK_ALREADY_FORMATED = false; % Rechecks formated subjects
opt.HYPER_THREAD = true; % Enable hyperthreading
opt.MATCH_INPUT = true; % Remove subjects found in output and not input

% Preprocess Options
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
opt.clearpsfile = true;
opt.sync_with_formated = true; % Removes any harvest ouput detected that isn't in the input database
opt.isMakeModalityTable = false;

% Organize Preprocess Data Options
opt.forcedPull = false;
opt.syncPreprocessed = true;

% BrainageR Options
opt.deleteBrainageRTemp = true;
opt.syncBrainageR = true;

% Quality Check Options
opt.recheckOutput = true;
opt.CHECK_SESSIONMATCH = true;
opt.CHECK_AQ = true;
opt.CHECK_BRAINAGER = false;
opt.DELETEBRAINAGEIFFAIL = false;

% Autoremove options
opt.AR.nii_proc = true;
opt.AR.harvest_output = true;
opt.AT.wk_matfile = true;
opt.AR.brainageR = true;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% DON'T CHANGE CODE BELOW (unless you know what you are doing) %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%DISEASE_TAG = 'Patients';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%status%%%%%%%%%%%

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Find Database paths
opt.paths.database_path = fullfile(IMAGE_DATABASE,DATABASE_NAME);
cd(opt.paths.database_path)

% Define paths
opt.paths.image_database = IMAGE_DATABASE;
opt.paths.github = GITHUB_PATH;
opt.paths.raw_database= fullfile(opt.paths.database_path,'raw',DISEASE_TAG);
opt.paths.nii_preproc_database = fullfile(opt.paths.database_path,'nii_proc_format',DISEASE_TAG);
opt.paths.harvest_output = fullfile(opt.paths.database_path,'harvestOutput',DISEASE_TAG);
opt.paths.processed_output = fullfile(opt.paths.database_path,'processed',DISEASE_TAG);
opt.paths.brainage_folder = fullfile(opt.paths.database_path,'brainageOutput',DISEASE_TAG);
opt.paths.post_qc = fullfile(opt.paths.database_path,'post_qc',DISEASE_TAG);
opt.paths.checkdir = fullfile(opt.paths.nii_preproc_database,DISEASE_TAG);

opt.paths.brainage_path = fileparts(which('spm_preprocess_brainageR.m'));
opt.paths.spm_path = fullfile(GITHUB_PATH,'Toolbox','imaging','brainageR_SPM');
opt.paths.tempbrain_folder = fullfile(opt.paths.database_path,'brainageTEMP');

% Check if raw database exits
if ~exist(opt.paths.raw_database,'dir')
    error([DISEASE_TAG,' Folder Not Found. Format is : path_to_master_set_folder/<patients variable name>'])
end

%% Prepare Files for nii_harvest_parallel

% Prepare file for nii_harvest_parallel
format_errors = prep_niiharvest(opt);

% Display Step 1 completion
display_complete(1,'Preprocess Format',format_errors)
%% Harvest Paralllel

% Run nii_harvest_parallel
errors_parallel = nii_harvest_parallel(opt.paths.nii_preproc_database,opt.paths.harvest_output,opt);
% nii_harvest_parallel(opt.paths.nii_preproc_database,opt.paths.harvest_output,opt,{'BONPL0149'});

% Display Step 2 completion
display_complete(2,'Harvest Parallel',errors_parallel)

%% Preprocess Parallel DTI

% Run nii_harvest (DTI)
DTI_errors = nii_harvest_parallel_DTI(opt.paths.nii_preproc_database,opt.paths.harvest_output,opt);
% nii_harvest_parallel_DTI(opt.paths.nii_preproc_database,opt.paths.harvest_output,opt,{'EMOPL0012'});

% Display Step 3 completion
display_complete(3,'DTI processing',DTI_errors)
%% Organize Preprocessed Data

% Run extraction script
nii_preprocess_subfolders(opt)

% Display Step 4 completion
display_complete(4,'wk_matfile Extraction')
%% Brain Age

% Prep Brain Age Files
setup_brainagedir(opt)

% Run BrainAge
brainageR_errors = run_brainage_parallel(opt);
%%
brainageR_errors = run_brainage_parallel(opt,'BONPL0120');

% Display Step 5 completion
display_complete(5,'Brain Age Calculation',brainageR_errors)
%% Perform QC

% Pipeline Info
pipelineinfo.author = 'Allen Chang';
pipelineinfo.toolboxes = [{'Nii_preproc (Allen edited) adapted from https://github.com/neurolabusc/nii_preprocess'};
    {'brainageR (software version 2.1 Oct 2020; https://github.com/james-cole/brainageR)'};
    {'brainageR Specific SPM'}];
pipelineinfo.date = date;

% Run Quality Check
QC_failed = run_QC(opt,pipelineinfo);

% Display Step 6 completion
display_complete(6,'Quality Check',QC_failed)

%% Auto Remove Failed Subjects

%autoremove(QC_failed,opt)

%np_clear('EMOPR0036',opt,false)

%% Functions
function [errors] = prep_niiharvest(opt)

% Define paths
opt.paths.database_path = opt.paths.raw_database;
output_database = opt.paths.nii_preproc_database;

% Define options
SKIP_PROBLEM_TAG = opt.SKIP_PROBLEM_TAG;
HYPER_THREAD = opt.HYPER_THREAD;
RECHECK_ALREADY_FORMATED = opt.RECHECK_ALREADY_FORMATED;
PROBLEM_TAGS = opt.PROBLEM_TAGS;
MATCH_INPUT = opt.MATCH_INPUT;

% Create formated folder
mkdir(opt.paths.nii_preproc_database)

% Obtain Directories
sbj_dir = dir(opt.paths.raw_database);
sbj_dir = sbj_dir(~contains({sbj_dir.name},'.') & [sbj_dir.isdir]);
output_dir = dir(output_database);
output_dir = output_dir(~contains({output_dir.name},'.') & [output_dir.isdir]);

% Remove subject if not found in inputdatabase
if MATCH_INPUT
    rm_idx = find(~cellfun(@(x) any(strcmp(x,cellfun(@(x) strrep(x,'_',''),{sbj_dir.name},'UniformOutput',false))),{output_dir.name}));
    for i = 1:numel(rm_idx)
        disp(['Subject [',output_dir(rm_idx(i)).name,'] found in output but not in input ..  Removing output subject folder'])
        rmdir(fullfile(output_dir(rm_idx(i)).folder,output_dir(rm_idx(i)).name),'s')
    end
end


% Throw Error if No subjects detected
if isempty(sbj_dir)
    clc
    disp('ERROR: NO SUBJECTS DETECTED IN INPUT FOLDER')
    return
end

% Display # of detected subjects
nns = num2str(numel(sbj_dir)-numel(output_dir));
disp(['New Subjects Detect: ',nns])
if nns == '0'
    errors = [];
    return
end

% Dedicate Error Vars
errors = cell(1,numel(sbj_dir));
sbj_error = cell(1,numel(sbj_dir));

% Enable Hyperthreading
if HYPER_THREAD
    setpool(3);
else
    setpool(2);
end

% Count how many with problem tags
problem_tag_count = cell(size(sbj_dir));

% Parallel Processing REQUIRED
parfor sbj = 1:numel(sbj_dir)
    removal_dir = [];

    % Turn off unnessary warnings
    warning('off','all')

    % Skip subject if has a 'problem' tag
    skip_post = false;
    if SKIP_PROBLEM_TAG
        skip_tag = cellfun(@(x) endsWith(sbj_dir(sbj).name,x),PROBLEM_TAGS);
        if any(skip_tag)
            if opt.SKIP_LESION
                if opt.PROBLEM_TAGS_LOGIC{skip_tag}
                    skip_post = true;
                else
                    problem_tag_count{sbj} = 1;
                    continue
                end
            else
                problem_tag_count{sbj} = 1;
                continue
            end
        end
    end

    % Find Subject Name/Folder
    subject_name = sbj_dir(sbj).name;
    subject_input_folder = sbj_dir(sbj).folder;

    % Define New Subject name (no '_' allowed)
    new_subject_name = strrep(subject_name,'_','');

    % Define removal output folder if any errors occur
    removal_dir = fullfile(output_database,new_subject_name);

    % Define pipeline Status file
    statusFile = fullfile(output_database,new_subject_name,'PipelineStatus.txt');

    % Skip if pipeline Status file exists (i.e., reformated already)
    if ~RECHECK_ALREADY_FORMATED
        if any(exist(statusFile,'file'))
            continue
        end
    end

    % Attempt to format images
    try
        % Find Session Type based on T1 labels
        s_dir = dir(fullfile(subject_input_folder,subject_name,'*T1*'));
        t1_names = {s_dir.name};

        if isempty(t1_names) % Make sure there is T1 present
            errors{sbj} = 'Missing T1 Scan';
            sbj_error{sbj} = sbj_dir(sbj).name;
            continue
        end

        aq = [];
        if any(contains(t1_names,'pre'))
            aq = [aq,{'pre'}];
        end
        if any(contains(t1_names,'pos'))
            aq = [aq,{'pos'}];
        end
        if isempty(aq)
            aq = [aq,{'session'}];
        end

        % Check to see all sessions are accounted for
        secondary = extractBetween(t1_names,'T1','.nii');
        secondary(cellfun(@isempty,secondary)) = [];
        secondary = unique(secondary);
        if ~isempty(secondary)
            for i = 1:numel(secondary)
                t1_names(cellfun(@(x) contains(x,secondary{i}),t1_names)) = [];
            end
        end
        if ~(numel(t1_names) == numel(aq))
            new_t1_names = [];
            for a = 1:numel(aq)
                aq_t1_names = t1_names(contains(t1_names,aq{a}));
                if numel(aq_t1_names) > 1
                    % Remove .gz files to see if that fixes the problem
                    new_t1_names = [new_t1_names aq_t1_names(~cellfun(@(x) contains(x,'.gz'),aq_t1_names))];
                else
                    new_t1_names = [new_t1_names aq_t1_names{:}];
                end
            end
            t1_names = new_t1_names;

            if ~(numel(t1_names) == numel(aq))
                errors{sbj} = 'Number of unique Sessions do not match number of unique T1 (T1 file naming error)';
                sbj_error{sbj} = sbj_dir(sbj).name;
                continue
            end
        end

        cont = true;
        while cont

            % Transfer Modalities of each Aquisition Session
            modality = [];
            status = [];
            for a = 1:numel(aq)
                aq_type = aq{a};

                T1 = [];
                T2 = [];
                RS = [];
                dti = [];
                lesion = [];
                les_dim =[];
                fn = [];
                % Detect T1/RS/LS files
                if strcmp(aq_type,'session')
                    T1 = dir(fullfile(opt.paths.database_path,subject_name,t1_names{1}));
                    T2 = dir(fullfile(opt.paths.database_path,subject_name,'*T2*.nii*'));
                    RS = dir(fullfile(opt.paths.database_path,subject_name,'*rs*.nii*'));
                    dti = dir(fullfile(opt.paths.database_path,subject_name,'*diff*.nii*'));
                    lesion = []; %THERE IS NO LESION FOR SESSION
                elseif strcmp(aq_type,'pre')
                    T1 = dir(fullfile(opt.paths.database_path,subject_name,t1_names{contains(t1_names,aq_type)}));
                    T2 = dir(fullfile(opt.paths.database_path,subject_name,['*',aq_type,'*T2*.nii*']));
                    RS = dir(fullfile(opt.paths.database_path,subject_name,['*',aq_type,'*rs*.nii*']));
                    lesion = []; %THERE IS NO LESION FOR PRE
                    dti = dir(fullfile(opt.paths.database_path,subject_name,['*',aq_type,'*diff*.nii*']));
                elseif strcmp(aq_type,'pos')
                    if skip_post
                        continue
                    end
                    T1 = dir(fullfile(opt.paths.database_path,subject_name,t1_names{contains(t1_names,aq_type)}));
                    T2 = []; % Post Surgical Lesions drawn on T1s, don't find T2
                    RS = dir(fullfile(opt.paths.database_path,subject_name,['*',aq_type,'*rs*.nii*']));
                    lesion = dir(fullfile(opt.paths.database_path,subject_name,'*les.nii*')); % ONLY LESION FOR POS
                    dti = dir(fullfile(opt.paths.database_path,subject_name,['*',aq_type,'*diff*.nii*']));
                    if isempty(lesion)
                        errors{sbj} = 'Post Nifti found but missing LESION';
                        sbj_error{sbj} = sbj_dir(sbj).name;
                        cont = false;
                        break
                    end
                end

                % Define Subject Image Folders
                subject_output_folder = fullfile(output_database,new_subject_name,aq_type);

                % Recheck already formated folders to see if any need
                % updating
                if any(exist(statusFile,'file'))
                    fid=fopen(statusFile);
                    lines = cell(0,1);
                    while true
                        tline = fgetl(fid);
                        if ~ischar(tline)
                            break
                        end
                        lines{end+1,1} = tline;
                    end
                    fclose(fid);
                    if any(cellfun(@(x) contains(x,aq_type),lines))
                        num_files = numel(dir(fullfile(output_database,new_subject_name,aq_type,['*',new_subject_name,'*'])));
                        num_files_detected = sum([~isempty(T1)
                            ~isempty(T2)
                            ~isempty(RS)*2
                            ~isempty(lesion)
                            ~isempty(dti)*4]);
                        if num_files_detected == num_files
                            cont = false;
                            break
                        else
                            rmdir(removal_dir,'s');
                        end
                    end
                end

                % Create Subject Image Folder
                mkdir(subject_output_folder)

                %%%%%%%%%%%%%%%% Transfer Lesion
                if isempty(lesion)
                    % Record Status
                    modality = [modality;{'Lesion'}];
                    status = [status;{'Lesion not found'}];
                else
                    % Record Status
                    modality = [modality;{'Lesion'}];
                    status = [status;{fullfile(lesion.folder,lesion.name)}];

                    % Transfer Lesion image
                    lesion_new = ['Lesion_',new_subject_name,'.nii'];
                    if contains(lesion.name,'.gz')
                        filenames = gunzip(fullfile(lesion.folder,lesion.name));
                        movefile(filenames{:},fullfile(subject_output_folder,lesion_new));
                    else
                        copyfile(fullfile(lesion.folder,lesion.name),fullfile(subject_output_folder,lesion_new))
                    end

                    % Store lesion dimension
                    V = spm_vol(fullfile(subject_output_folder,lesion_new));
                    les_dim = V.dim;
                end


                %%%%%%%%%%%%%%%%%%%%% Transfer T1

                % Check for matching T1 to lesion
                for t = 1:numel(T1)

                    % Transfer T1 image
                    T1_new = ['T1_',new_subject_name,'.nii'];
                    if contains(T1(t).name,'.gz')
                        filenames = gunzip(fullfile(T1(t).folder,T1(t).name));
                        movefile(filenames{:},fullfile(subject_output_folder,T1_new));
                    else
                        copyfile(fullfile(T1(t).folder,T1(t).name),fullfile(subject_output_folder,T1_new))
                    end

                    % Load T1 volume
                    V = spm_vol(fullfile(subject_output_folder,T1_new));

                    % Check to see T1 is only a single volume
                    if numel(V)>1
                        delete(fullfile(subject_output_folder,T1_new))
                        if t == numel(T1)
                            errors{sbj} = [aq_type,' T1 files have more that 1 volume'];
                            sbj_error{sbj} = sbj_dir(sbj).name;
                            rmdir(removal_dir,'s');
                            cont = false;
                            break
                        end
                        continue
                    end
                    if ~isempty(lesion)

                        % Check if T1 and lesion match
                        t1_dim = V.dim;
                        if all(t1_dim == les_dim)
                            break
                        else
                            delete(fullfile(subject_output_folder,T1_new))
                            if t == numel(T1)
                                errors{sbj} = [aq_type,' T1 files dont match lesion'];
                                sbj_error{sbj} = sbj_dir(sbj).name;
                                rmdir(removal_dir,'s');
                                cont = false;
                                break
                            end
                            continue
                        end
                    else
                        break
                    end
                end

                % Break out of the aquisition forloop if error is found
                if cont == false
                    break
                end


                %%%%%%%%%%%%%%%%%%% Transfer T2

                if isempty(T2)
                    % Record Status
                    modality = [modality;{'T2'}];
                    status = [status;{'T2 not found'}];
                else
                    % Remove Secondary
                    T2 = T2(cellfun(@isempty,extractBetween({T2.name},'T2','.nii')));
                    if isempty(T2)
                        errors{sbj} = [aq_type,' secondary scan removal failed for T2'];
                        sbj_error{sbj} = sbj_dir(sbj).name;
                        rmdir(removal_dir,'s');
                        cont = false;
                        break
                    end

                    % Remove extra .gz if present
                    if numel(T2) > 1
                        T2(contains({T2.name},'.gz')) = [];
                    end
                    % Record Status
                    modality = [modality;{'T2'}];
                    status = [status;{fullfile(T2.folder,T2.name)}];

                    % Transfer T2 image
                    T2_new = ['T2_',new_subject_name,'.nii'];
                    if contains(T2.name,'.gz')
                        filenames = gunzip(fullfile(T2.folder,T2.name));
                        movefile(filenames{:},fullfile(subject_output_folder,T2_new));
                    else
                        copyfile(fullfile(T2.folder,T2.name),fullfile(subject_output_folder,T2_new))
                    end
                end

                %%%%%%%%%%%%%%%%%%% Transfer Resting State
                if isempty(RS)
                    % Record Status
                    modality = [modality;{'RS'}];
                    status = [status;{'RS not found'}];
                else
                    for r = 1:numel(RS)

                        % Assign new RS file name
                        RS_new = ['Rest_',new_subject_name,'.nii'];

                        % Check json file
                        rs_json = fullfile(RS(r).folder,[extractBefore(RS(r).name,'.nii'),'.json']);
                        if ~any(exist(rs_json,'file'))
                            if r == numel(RS)
                                errors{sbj} = [aq_type,'_rs.json file missing'];
                                sbj_error{sbj} = sbj_dir(sbj).name;
                                rmdir(removal_dir,'s');
                                cont = false;
                                break
                            end
                            continue
                        else
                            slice_options = {[1 2 3 4],[4 3 2 1],[1 3 2 4],[4 2 3 1],[2 4 1 3],[3 1 4 2]};
                            try
                                [~,so] = sort(jsondecode(fileread(rs_json)).SliceTiming(1:4));
                            catch
                                continue
                            end
                            slice_order = find(cellfun(@(x) all(so' == x),slice_options), 1);
                            if isempty(slice_order)
                                if r == numel(RS)
                                    errors{sbj} = [aq_type,' - Slice Order for fMRI is not accetable'];
                                    sbj_error{sbj} = sbj_dir(sbj).name;
                                    rmdir(removal_dir,'s');
                                    cont = false;
                                    break
                                end
                                continue
                            end
                        end

                        % Check Number of Volumes
                        if contains(RS(r).name,'.gz')
                            filenames = gunzip(fullfile(RS(r).folder,RS(r).name));
                            [pth,nam,ext] = spm_fileparts(deblank(filenames));
                            sesname = fullfile(pth,[nam, ext]);
                            hdr = spm_vol(sesname);
                            delete(filenames{:})
                        else
                            [pth,nam,ext] = spm_fileparts(deblank(fullfile(RS(r).folder,RS(r).name)));
                            sesname = fullfile(pth,[nam, ext]);
                            hdr = spm_vol(sesname);
                        end
                        nvol = length(hdr);
                        if nvol < 12
                            if r == numel(RS)
                                errors{sbj} = [aq_type,' RS files contains <12 volumes'];
                                sbj_error{sbj} = sbj_dir(sbj).name;
                                rmdir(removal_dir,'s');
                                cont = false;
                                break
                            end
                            continue
                        end

                        % Transfer RS image
                        if contains(RS(r).name,'.gz')
                            filenames = gunzip(fullfile(RS(r).folder,RS(r).name));
                            movefile(filenames{:},fullfile(subject_output_folder,RS_new));
                        else
                            copyfile(fullfile(RS(r).folder,RS(r).name),fullfile(subject_output_folder,RS_new))
                        end


                        % Copy RS json
                        copyfile(rs_json,fullfile(subject_output_folder,strrep(RS_new,'.nii','.json')))

                        % Update RS Status
                        modality = [modality;{'RS'}];
                        status = [status;{fullfile(RS(r).folder,RS(r).name)}];
                        break
                    end
                end

                % Break out of the aquisition forloop if error is found
                if cont == false
                    break
                end

                %%%%%%%%%%%%%%%%%%%%%% Transfer DTI
                if isempty(dti)
                    modality = [modality;{'DTI'}];
                    status = [status;{'DTI not found'}];
                else
                    for d = 1:numel(dti)

                        % Define bval/bvec/json files
                        bval = fullfile(dti(d).folder,[extractBefore(dti(d).name,'.nii'),'.bval']);
                        bvec = fullfile(dti(d).folder,[extractBefore(dti(d).name,'.nii'),'.bvec']);
                        dti_json = fullfile(dti(d).folder,[extractBefore(dti(d).name,'.nii'),'.json']);

                        % Check bval files
                        if ~any(exist(bval,'file'))
                            if d == numel(dti)
                                errors{sbj} = [aq_type,'_DTI.bval files missing'];
                                sbj_error{sbj} = sbj_dir(sbj).name;
                                rmdir(removal_dir,'s');
                                cont = false;
                                break
                            end
                            continue
                        end

                        % Check bvec files
                        if ~any(exist(bvec,'file'))
                            if d == numel(dti)
                                errors{sbj} = [aq_type,'_DTI.bvec file missing'];
                                sbj_error{sbj} = sbj_dir(sbj).name;
                                rmdir(removal_dir,'s');
                                cont = false;
                                break
                            end
                            continue
                        end

                        % Check json files
                        if ~any(exist(dti_json,'file'))
                            if d == numel(dti)
                                errors{sbj} = [aq_type,'_DTI.json file missing'];
                                sbj_error{sbj} = sbj_dir(sbj).name;
                                rmdir(removal_dir,'s');
                                cont = false;
                                break
                            end
                            continue
                        end

                        % Check to see >12 # of bval values
                        fileID = fopen(bval,'r');
                        [~, n] = fscanf(fileID,'%g');
                        fclose(fileID);
                        if n < 12
                            % If all scans fail to have >11 then throw error
                            if d == numel(dti)
                                errors{sbj} = [aq_type,'_DTI.bval values <12'];
                                sbj_error{sbj} = sbj_dir(sbj).name;
                                rmdir(removal_dir,'s');
                                cont = false;
                            end
                            continue
                        else

                            % Copy DTI scan
                            dti_new = ['DTI_',new_subject_name,'.nii'];
                            if contains(dti(d).name,'.gz')
                                filenames = gunzip(fullfile(dti(d).folder,dti(d).name));
                                movefile(filenames{:},fullfile(subject_output_folder,dti_new));
                            else
                                copyfile(fullfile(dti(d).folder,dti(d).name),fullfile(subject_output_folder,dti_new))
                            end

                            % Check if DTI volume matches bval
                            copied_dtifile = fullfile(subject_output_folder,dti_new);
                            h = spm_vol(copied_dtifile);
                            if numel(h) ~= n
                                if d == numel(dti)
                                    errors{sbj} = [aq_type,'_DTI.bval values doesnt match DTI'];
                                    sbj_error{sbj} = sbj_dir(sbj).name;
                                    rmdir(removal_dir,'s');
                                    cont = false;
                                end
                                delete(copied_dtifile)
                                continue
                            end

                            % Copy DTI bval/bvec/json
                            copyfile(bval,fullfile(subject_output_folder,['DTI_',new_subject_name,'.bval']))
                            copyfile(bvec,fullfile(subject_output_folder,['DTI_',new_subject_name,'.bvec']))
                            copyfile(dti_json,fullfile(subject_output_folder,['DTI_',new_subject_name,'.json']))

                            % Update DTI Status
                            modality = [modality;{'DTI'}];
                            status = [status;{fullfile(dti(d).folder,dti(d).name)}];

                            break
                        end
                    end
                end

                % Break out of the aquisition forloop if error is found
                if cont == false
                    break
                end

                % Create file note
                fid = fopen(statusFile, 'wt' );
                fprintf(fid, '%s\n',aq_type);
                for i = 1:numel(modality)
                    fprintf(fid, '-%s ... %s\n',modality{i}, status{i});
                end
                fclose(fid);
            end

            % Break out of the while forloop if error is found
            if cont == false
                break
            end

            % Display Complete Update
            if exist(removal_dir,'dir')
                disp([sbj_dir(sbj).name,' formatting completed'])
            else
                problem_tag_count{sbj} = 1;
            end
            cont = false;
        end
    catch e
        errors{sbj} = e.message;
        sbj_error{sbj} = sbj_dir(sbj).name;
        if exist(removal_dir,'dir')
            rmdir(removal_dir,'s');
        end
    end
end

% Display problem tag count
disp([num2str(sum(~cellfun(@isempty,problem_tag_count))),' Subjects skiped with problem tag'])

% Check if there were any errors
errors(cellfun(@isempty,errors)) = [];
sbj_error(cellfun(@isempty,sbj_error)) = [];
if ~isempty(errors)
    errors =[sbj_error' errors'];
end


end

function setup_brainagedir(opt)
disp('Preping brainageR')

% Set up brainageR specific paths
GITHUB_PATH = opt.paths.github;
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'brainageR')

% Check to see if any large files missing that cant be cloned via github
pca_rotation = fullfile(opt.paths.brainage_path,'pca_rotation.rds');
if ~exist(pca_rotation,"file")
    disp('brainageR setup file missing... attempting to auto-download missing file... may take awhile')
    cmd = 'cd ~/Downloads; wget -nv https://github.com/james-cole/brainageR/releases/download/2.1/pca_rotation.rds';
    status = system(cmd,'-echo');
    if status ~=0
        error('FAILED attempt to download pca_rotation.rds file from https://github.com/james-cole/brainageR/releases/download/2.1/pca_rotation.rds to Downloads Folder');
    else
        system(['mv ~/Downloads/pca_rotation.rds ',opt.paths.brainage_path]);
    end
end

% Detect brainage SH file
brainage_file_path = fullfile(opt.paths.brainage_path,'brainageR');

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
brainageR_dir_line = sprintf('brainageR_dir=%s',[fileparts(opt.paths.brainage_path),filesep]);
fileContents{idx(1)} = brainageR_dir_line;

% Replace spm_dir path
idx = find(~cellfun(@isempty,(regexp(fileContents,'spm_dir'))));
opt.paths.spm_path_line = sprintf('spm_dir=%s',[opt.paths.spm_path,filesep]);
fileContents{idx(1)} = opt.paths.spm_path_line;

% Replace matlab_path path
idx = find(~cellfun(@isempty,(regexp(fileContents,'matlab_path'))));
matlab_path_line = sprintf('matlab_path=%s',fullfile(matlabroot,'bin','matlab'));
fileContents{idx(1)} = matlab_path_line;

% write/replace brainageR sh file
brainage_file = fopen(brainage_file_path,'w');
fprintf(brainage_file,'%s\n',string(fileContents));
fclose(brainage_file);

disp('brainageR prep steps complete')
end

function brainageR_errors = run_brainage_parallel(opt,debug_sub)

disp('Running brainageR')

% Create temp brainage folder
if exist(opt.paths.tempbrain_folder,'dir') && opt.deleteBrainageRTemp
    rmdir(opt.paths.tempbrain_folder,'s')
end
mkdir(opt.paths.tempbrain_folder)

% Create brianage output folder
mkdir(opt.paths.brainage_folder)

% Sync brainage output with processed
if opt.syncBrainageR
    processed_brainageR = dir(fullfile(opt.paths.brainage_folder,'*.mat'));
    processed_matfile = dir(fullfile(opt.paths.processed_output,'*','*.mat'));

    rm_idx = find(~cellfun(@(x) any(contains({processed_matfile.name},x)),{processed_brainageR.name}));
    if ~isempty(rm_idx)
        disp(['Removing ',num2str(numel(rm_idx)),' subject found in brainageR but not processed'])
        for r = 1:numel(rm_idx)
            disp(['Removed ',processed_brainageR(rm_idx(r)).name])
            delete(fullfile(processed_brainageR(rm_idx(r)).folder,processed_brainageR(rm_idx(r)).name))
        end
    end
end

% Start brainageR or debug mode
if exist('debug_sub','var')
    brainageR_errors{1} = start_brainageR_parallel( ...
        dir(fullfile(opt.paths.processed_output,'**',[debug_sub,'.mat'])), ...
        opt);
else

    % Reset parallel pool
    setpool(1,true);

    % Detect Nii_Harvested Subjects
    processed_wk_matfiles = dir(fullfile(opt.paths.processed_output,'**','*.mat'));
    
    % Run brainageR
    brainageR_errors = cell(size(processed_wk_matfiles));
    parfor a = 1:numel(processed_wk_matfiles)
        brainageR_errors{a} = start_brainageR_parallel(processed_wk_matfiles(a),opt);
    end
end
brainageR_errors(cellfun(@isempty,brainageR_errors)) = [];
end

function brainageR_error = start_brainageR_parallel(wk_matfile,opt)
brainageR_error = [];
er = [];

% Define paths
brainage_folder = opt.paths.brainage_folder;
nii_preproc_database = opt.paths.nii_preproc_database;
tempbrain_folder = opt.paths.tempbrain_folder;
brainage_path = opt.paths.brainage_path;
harvest_output = opt.paths.harvest_output;

% Supress warnings
warning('off','all')

% Define brainageR output wk_matfile
wk_mat_output = fullfile(brainage_folder,wk_matfile.name);

% Mat sure inputs are not empty
if isempty(wk_matfile)
    error('Subjects cannot be found')
end

% Skip if brainageR completed
if exist(wk_mat_output,'file')
    rerun = false;
    matObj = matfile(wk_mat_output);
    fn = fieldnames(matObj);
    for f = 1:numel(fn)
        if strcmp(fn{f},'Properties')
            continue
        end
        if any(strcmp(fieldnames(matObj.(fn{f})),'brainage'))
            continue
        else
            rerun = true;
        end
    end
    if ~rerun
        return
    end
end

% Find T1 scans/sessions
sbj_name = extractBefore(wk_matfile.name,'.mat');
input = dir(fullfile(nii_preproc_database,sbj_name,'**','T1*'));

% Load processed wk_matfile
mat_folder = wk_matfile.folder;
wk_mat = load(fullfile(mat_folder,wk_matfile.name));

for s = 1:numel(input)
    try
        % Define session name
        [~,ses] = fileparts(input(s).folder);
    
        % Make subject folder
        wk_sbj = extractBefore(input(s).name,'.nii');
        wk_sbj_folder = fullfile(tempbrain_folder,wk_sbj);
        mkdir(wk_sbj_folder);
    
        % Copy origin corrected T1 into brainage temp folder
        if strcmp(ses,'pos')
            oc_T1 = dir(fullfile(harvest_output,sbj_name,ses,'eT1*.nii'));
            if isempty(oc_T1)
                er='Enantiomorphic pos-T1 not found';
                error(er)
            end
        else
            oc_T1 = dir(fullfile(harvest_output,sbj_name,ses,'T1*.nii'));
        end
        if numel(oc_T1) ~= 1; er='more or no oc_T1 found';error(er);end
        s_input = fullfile(wk_sbj_folder,input(s).name);
        copyfile(fullfile(oc_T1.folder,oc_T1.name),s_input);
        
    
        % Define output
        brainage_output = fullfile(wk_sbj_folder,[wk_sbj,'_',ses,'_brainage.csv']);
        if ~exist(brainage_output,'file')
            % Run brainageR
            cmd = sprintf('%s -f %s -o %s ',fullfile(brainage_path,'brainageR'),s_input,brainage_output);
            system(cmd,'-echo');
        end
    
        if exist(brainage_output,'file')
    
            % Save brainageR output
            brain_age_pred = readtable(brainage_output);
            brain_age_pred_tissue_vols = readtable(fullfile(wk_sbj_folder,[brain_age_pred.File{:},'_tissue_volumes.csv']));
            wk_mat.(ses).brainage.agePred = brain_age_pred;
            wk_mat.(ses).brainage.TissueVol = brain_age_pred_tissue_vols;
    
            slice_dir = dir(fullfile(wk_sbj_folder,['slicesdir_T1_',extractBefore(wk_matfile.name,'.mat'),'.nii'],'*.png'));
            for p = 1:numel(slice_dir)
                png = imread(fullfile(slice_dir(p).folder,slice_dir(p).name));
                t_name = extractBetween(slice_dir(p).name,'__','.png');
                wk_mat.(ses).brainage.slicedir.(t_name{:}) = png;
            end
    
            % Save completed wk_matfile with brainageR field
            saveparfor(wk_mat_output,'-struct',wk_mat)

            % Delete files
            rmdir(wk_sbj_folder,'s')
    
        else
            brainageR_error = [brainageR_error;{[wk_matfile.name,' brainageR failed']}];
        end
    catch
        brainageR_error = [brainageR_error;{[wk_matfile.name,' ',er]}];
    end
end


end

function errors = run_QC(opt,pipelineinfo)

% Make Post QC folder
mkdir(opt.paths.post_qc)
cd(opt.paths.post_qc)

% Set options
CHECK_SESSIONMATCH = opt.CHECK_SESSIONMATCH;
CHECK_AQ = opt.CHECK_AQ;
CHECK_BRAINAGER = opt.CHECK_BRAINAGER;
DELETEBRAINAGEIFFAIL = opt.DELETEBRAINAGEIFFAIL;

% Delete pool
setpool(0)

% Find processed wk_matfiles
subjects = dir(fullfile(opt.paths.brainage_folder,'*.mat'));
subject_folder = subjects.folder;
subjects = {subjects.name};

errors = cell(size(subjects));
textprogressbar(0,'Running Quality Check');
for s = 1:numel(subjects)
    fn = [];

    % Define Save_mat
    save_mat = fullfile(opt.paths.post_qc,subjects{s});
    if exist(save_mat,'file') && ~opt.recheckOutput
        continue
    end

    % Define subject name
    wk_sbj = extractBefore(subjects{s},'.mat');

    % Load processed wk_matfile
    processed_wk_matfile = fullfile(subject_folder,subjects{s});
    wk_mat = load(processed_wk_matfile);

    % Allocated variables
    qc = [];

    % Check session matches
    input_ses = dir(fullfile(opt.paths.nii_preproc_database,wk_sbj));
    input_ses(contains({input_ses.name},'.')) = [];
    wk_sesions = input_ses([input_ses.isdir]);
    if CHECK_SESSIONMATCH
        if numel(fieldnames(wk_mat)) == numel(wk_sesions)
            qc.ses_match = 'pass';
        else
            qc.ses_match = 'fail';
            msg = [wk_sbj,' SESSION MATCH FAIL'];
            textprogressbar(1,s/numel(subjects)*100,msg)
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
                            msg = [wk_sbj,' ',wk_ses.name,' Lesion FAIL'];
                            textprogressbar(1,s/numel(subjects)*100,msg)
                            cont = false;
                        end
                    case 'T1'
                        if any(contains(fn,'T1')) && any(contains(fn,'vbm')) && any(contains(fn,'VBM_'))
                            qc.T1 = 'pass';
                        else
                            qc.T1 = 'fail';
                            msg = [wk_sbj,' ',wk_ses.name,' T1 FAIL'];
                            textprogressbar(1,s/numel(subjects)*100,msg)
                            cont = false;
                        end
                    case 'T2'
                        continue
                    case 'Rest'
                        if any(contains(fn,'RestAve')) && any(contains(fn,'rest_'))
                            qc.Rest = 'pass';
                        else
                            qc.Rest = 'fail';
                            msg = [wk_sbj,' ',wk_ses.name,' Rest FAIL'];
                            textprogressbar(1,s/numel(subjects)*100,msg)
                            cont = false;
                        end
                    case 'DTI'
                        if any(contains(fn,'dti')) && any(contains(fn,'md')) && any(contains(fn,'fa'))
                            qc.DTI = 'pass';
                        elseif contains(wk_sbj,'DTIProblem')
                            qc.DTI = 'Skipped Due to problem tag';
                        else
                            qc.DTI = 'fail';
                            msg = [wk_sbj,' ',wk_ses.name,' DTI FAIL'];
                            textprogressbar(1,s/numel(subjects)*100,msg)
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
                    msg = [wk_sbj,' ',wk_ses.name,' brainage FAIL'];
                    textprogressbar(1,s/numel(subjects)*100,msg)
                    cont = false;
                end
            else
                qc.brainageR = 'fail';
                msg = [wk_sbj,' ',wk_ses.name,' brainage FAIL'];
                textprogressbar(1,s/numel(subjects)*100,msg)
                cont = false;
                if DELETEBRAINAGEIFFAIL
                    delete(processed_wk_matfile)
                end
            end
        end

        % Save QC structure
        wk_mat.(wk_ses.name).QC = qc;
        qc = [];
    end

    % Save QC structure back to wk_matfile
    if cont
        % Add pipeline info
        wk_mat.pipelineinfo = pipelineinfo;
        saveparfor(save_mat,'-struct',wk_mat)
        textprogressbar(1,s/numel(subjects)*100,[wk_sbj,' passed QC'])
    else
        errors{s} = [{wk_sbj} {wk_mat} {extractAfter(msg,[wk_sbj,' '])}];
    end
end
errors = errors(~cellfun(@isempty,errors));
errors = cat(1,errors{:});
textprogressbar(2,'QC Complete')


end

function saveparfor(outname,opt1,save_mfile)
save(outname,opt1,'save_mfile')
end

function display_complete(stepnum,step,error_num)

if ~exist('error_num','var')
elseif isempty(error_num)
    error_num = 0;
else
    error_num = size(error_num,1);
end

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp(' ')
disp(['Step ',num2str(stepnum),'---',step,' Complete'])
if nargin>2
    disp([num2str(error_num),' Errors Detected'])
end
disp(' ')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

end


function autoremove(QC_failed,opt)

disp(['Auto-removing ',num2str(length(QC_failed)),' subjects'])

for sbj = 1:size(QC_failed,1)
    wk_sbj = QC_failed{sbj,1};
    wk_mat = QC_failed{sbj,2};

    sessions = fieldnames(wk_mat);

    for s = 1:numel(sessions)
        wk_qc = wk_mat.(sessions{s}).QC;
        checks = fieldnames(wk_qc);

        for c = 1:numel(checks)
            if strcmp(wk_qc.(checks{c}),'pass')
                continue
            end
            switch checks{c}
                case 'ses_match'
                    np_clear(wk_sbj,opt,false)
                case 'lesion'
                    np_clear(wk_sbj,opt,false)
                case 'T1'
                    np_clear(wk_sbj,opt,false)
                case 'DTI'
                    np_clear(wk_sbj,opt,false)
                case 'Rest'
                    np_clear(wk_sbj,opt,false)
                case 'brainageR'
                    np_clear(wk_sbj,opt,true)
            end
        end

    end
end
end

function np_clear(wk_sbj,opt,o)
if o
    try
        if opt.AR.brainageR
            % Delete brainageR .wk_matfile
            mat = dir(fullfile(opt.paths.brainage_folder,['*',wk_sbj,'*.mat']));
            delete(fullfile(mat.folder,mat.name))
            disp(['Removed ',fullfile(mat.folder,mat.name)])
        end
    end
else
    try
        if opt.AR.nii_proc
            % Delete nii_preproc formated
            rmdir(fullfile(opt.paths.nii_preproc_database,wk_sbj),'s')
            disp(['Removed ',fullfile(opt.paths.nii_preproc_database,wk_sbj)])
        end
    end

    try
        if opt.AR.harvest_output
            % Delete harvest output
            rmdir(fullfile(opt.paths.harvest_output,wk_sbj),'s')
            disp(['Removed ',fullfile(opt.paths.harvest_output,wk_sbj)])
        end
    end

    try
        if opt.AT.wk_matfile
            % Delete extracted .wk_matfile
            mat = dir(fullfile(opt.paths.processed_output,'**',['*',wk_sbj,'*.mat']));
            delete(fullfile(mat.folder,mat.name))
            disp(['Removed ',fullfile(mat.folder,mat.name)])
        end
    end

    try
        if opt.AR.brainageR
            % Delete brainageR .wk_matfile
            mat = dir(fullfile(opt.paths.brainage_folder,'**',['*',wk_sbj,'*.mat']));
            delete(fullfile(mat.folder,mat.name))
            disp(['Removed ',fullfile(mat.folder,mat.name)])
        end
    end
end

end


