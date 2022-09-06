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
INPUT_PATH = '/media/bonilha/Elements/Master_Epilepsy_Database_SYNC';
OUTPUT_PATH = '/media/bonilha/Elements/MasterSet';
DISEASE_TAG = 'Patients';


%%%%%%%%%%%%%%%%%%%%%%%%% ADVANCE OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Format Options
opt.SKIP_PROBLEM_TAG = true; %true = don't preprocess subjects with a "problem" string attached at the end
opt.PROBLEM_TAGS = {'problem','MissingT1','MissingLesion','IncorrectLesion'}; % ends with tags that SKIP_PROBLEM_TAG will look for
opt.RECHECK_ALREADY_FORMATED = false; % Rechecks formated subjects
opt.HYPER_THREAD = true; % MAY CAUSE OVERHEATING. Since moving a lot of files, CPUs will benefit from hyperthreading
opt.MATCH_INPUT = true;

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
opt.sync_with_formated = true;

% Quality Check Options
opt.CHECK_SESSIONMATCH = true;
opt.CHECK_AQ = true;
opt.CHECK_BRAINAGER = true;

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
brainage_folder = fullfile(OUTPUT_PATH,'brainageOutput',DISEASE_TAG);
post_qc = fullfile(OUTPUT_PATH,'post_qc',DISEASE_TAG);
checkdir = fullfile(nii_preproc_database,DISEASE_TAG);

brainage_path = fileparts(which('spm_preprocess_brainageR.m'));
spm_path = fullfile(GITHUB_PATH,'Toolbox','imaging','brainageR_SPM');
tempbrain_folder = fullfile(OUTPUT_PATH,'brainageTEMP');


%% Prepare Files for nii_harvest_parallel

% Create formated folder
mkdir(nii_preproc_database)
cd(nii_preproc_database)

% Prepare file for nii_harvest_parallel
[format_errors] = prep_niiharvest(raw_database,nii_preproc_database,opt);

% Display Step 1 completion
display_complete(1,'Preprocess Format',format_errors)


%% Harvest Paralllel
% Create Harvest Output Folder
mkdir(harvest_output)

% Run nii_harvest_parallel
errors_parallel = nii_harvest_parallel(nii_preproc_database,harvest_output,opt);
errors_parallel = errors_parallel(~cellfun(@isempty,errors_parallel));
errors_parallel_sbjs = cellfun(@(x) x{1},errors_parallel,'UniformOutput',false);

% % DEBUG
% nii_harvest_parallel(nii_preproc_database,harvest_output,opt,{'BONPL0103'});

% Display Step 2 completion
display_complete(2,'Harvest Parallel',errors_parallel_sbjs)

%% Preprocess DTI

% Create Harvest Output Folder
mkdir(harvest_output)

% Run nii_harvest (DTI)
DTI_errors = nii_harvest_DTI(nii_preproc_database,harvest_output,opt);

% Display Step 3 completion
display_complete(3,'DTI processing',DTI_errors)
%% Organize Preprocessed Data

% Create Processed Output Folder
mkdir(processed_output)

nii_preprocess_subfolders
nii_preprocess_subfolders(harvest_output,processed_output)

% Display Step 4 completion
display_complete(4,'Matfile Extraction')
%% Brain Age

% Prep Brain Age Files
setup_brainagedir(brainage_path,spm_path)

% Create temp brainage folder
mkdir(tempbrain_folder)

% Create brianage output folder
mkdir(brainage_folder)

% Run BrainAge
run_brainage_parallel(processed_output,nii_preproc_database,tempbrain_folder,brainage_path,brainage_folder)

% Display Step 5 completion
display_complete(5,'Brain Age Calculation')
%% Perform QC

% Make Post QC folder
mkdir(post_qc)
cd(post_qc)

% Pipeline Info
pipelineinfo.author = 'Allen Chang';
pipelineinfo.toolboxes = [{'Nii_preproc (Allen edited) adapted from https://github.com/neurolabusc/nii_preprocess'};
    {'brainageR (software version 2.1 Oct 2020; https://github.com/james-cole/brainageR)'};
    {'brainageR Specific SPM'}];
pipelineinfo.date = date;

% Run Quality Check
run_QC(post_qc,brainage_folder,pipelineinfo,nii_preproc_database,opt)

% Display Step 6 completion
display_complete(6,'Quality Check')

%% Functions

function [errors] = prep_niiharvest(database_path,output_database,opt)

% Define options/media/bonilha/AllenProj/CNN_project/PatientData/smallSet/Cat12_segmented

SKIP_PROBLEM_TAG = opt.SKIP_PROBLEM_TAG;
HYPER_THREAD = opt.HYPER_THREAD;
RECHECK_ALREADY_FORMATED = opt.RECHECK_ALREADY_FORMATED;
PROBLEM_TAGS = opt.PROBLEM_TAGS;
MATCH_INPUT = opt.MATCH_INPUT;

% Obtain Directories
sbj_dir = dir(database_path);
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

% Dedicate Error Vars
errors = cell(1,numel(sbj_dir));
sbj_error = cell(1,numel(sbj_dir));

% Enable Hyperthreading
if HYPER_THREAD
    core_info = evalc('feature(''numcores'')');
    l_cores = regexp(core_info,'MATLAB was assigned: ','split');
    l_cores = str2double(extractBefore(l_cores{2},' logical cores'));
    if ~isempty(gcp('nocreate'))
        pool = gcp('nocreate');
        delete(pool)
    end
    disp('WARNING....')
    disp('HYPER THREADING ENABLE')
    disp('MAY CAUSE OVERHEATING')
    disp('WARNING....')

    c = parcluster;
    c.NumWorkers = l_cores;
    pool = c.parpool(l_cores);
end

% Count how many with problem tags
problem_tag_count = cell(size(sbj_dir));

% Parallel Processing REQUIRED
parfor sbj = 1:numel(sbj_dir)

    % Turn off unnessary warnings
    warning('off','all')

    % Skip subject if has a 'problem' tag
    if SKIP_PROBLEM_TAG
        if any(cellfun(@(x) endsWith(sbj_dir(sbj).name,x),PROBLEM_TAGS))
            problem_tag_count{sbj} = 1;
            continue
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
                    T1 = dir(fullfile(database_path,subject_name,t1_names{1}));
                    T2 = dir(fullfile(database_path,subject_name,'*T2*.nii*'));
                    RS = dir(fullfile(database_path,subject_name,'*rs*.nii*'));
                    dti = dir(fullfile(database_path,subject_name,'*diff*.nii*'));
                    lesion = []; %THERE IS NO LESION FOR SESSION
                elseif strcmp(aq_type,'pre')
                    T1 = dir(fullfile(database_path,subject_name,t1_names{contains(t1_names,aq_type)}));
                    T2 = dir(fullfile(database_path,subject_name,['*',aq_type,'*T2*.nii*']));
                    RS = dir(fullfile(database_path,subject_name,['*',aq_type,'*rs*.nii*']));
                    lesion = []; %THERE IS NO LESION FOR PRE
                    dti = dir(fullfile(database_path,subject_name,['*',aq_type,'*diff*.nii*']));
                elseif strcmp(aq_type,'pos')
                    T1 = dir(fullfile(database_path,subject_name,t1_names{contains(t1_names,aq_type)}));
                    T2 = []; % Post Surgical Lesions drawn on T1s, don't find T2
                    RS = dir(fullfile(database_path,subject_name,['*',aq_type,'*rs*.nii*']));
                    lesion = dir(fullfile(database_path,subject_name,'*les.nii*')); % ONLY LESION FOR POS
                    dti = dir(fullfile(database_path,subject_name,['*',aq_type,'*diff*.nii*']));
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

                    % Store lesion dimesnsion
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

                    if ~isempty(lesion)

                        % Load T1 volume
                        V = spm_vol(fullfile(subject_output_folder,T1_new));
                        t1_dim = V.dim;

                        % Check if T1 and lesion match
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
            disp([sbj_dir(sbj).name,' formatting completed'])
            cont = false;
        end
    catch e
        errors{sbj} = e.message;
        sbj_error{sbj} = sbj_dir(sbj).name;
        rmdir(removal_dir,'s');
    end
end

% Display problem tag count
disp([num2str(sum(~cellfun(@isempty,problem_tag_count))),' Subjects skiped with problem tag'])

% Check if there were any errors
errors(cellfun(@isempty,errors)) = [];
sbj_error(cellfun(@isempty,sbj_error)) = [];
if ~isempty(errors)
    errors =[sbj_error' errors'];

    incorrect_input = true;
    while incorrect_input
        disp([num2str(size(errors,1)),' Error(s) detected'])
        disp('  Errors included the following tags. Please fix by checking errors variable or continue')
        unique_errors = unique(errors(:,2));
        for u = 1:numel(unique_errors)
            disp(['     ',unique_errors{u}])
        end
        cont = input('Do you still want to continue? (y/n)','s');
        if cont == 'y'
            incorrect_input = false;
        elseif cont == 'n'
            disp('Exiting out of pipeline. Check errors variable in workspace')
            return
        else
            clc
            disp(['You entered ... ',cont])
            disp('Input unknown. Please answer by entering y for yes or n for no')
        end
    end
end


end

function setup_brainagedir(brainage_path,spm_path)

% Check to see if any large files missing that cant be cloned via github
pca_rotation = fullfile(brainage_path,'pca_rotation.rds');

if ~exist(pca_rotation,"file")
    disp('brainageR setup file missing... attempting to auto-download missing file... may take awhile')
    cmd = 'cd ~/Downloads; wget -nv https://github.com/james-cole/brainageR/releases/download/2.1/pca_rotation.rds';
    status = system(cmd);
    if status ~=0
        error('FAILED attempt to download pca_rotation.rds file from https://github.com/james-cole/brainageR/releases/download/2.1/pca_rotation.rds to Downloads Folder');
    else
        system(['mv ~/Downloads/pca_rotation.rds ',brainage_path]);
    end
end

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

function run_brainage_parallel(processed_output,nii_preproc_database,tempbrain_folder,brainage_path,brainage_folder)

% Detect Nii_Harvested Subjects
processed_matfiles = dir(fullfile(processed_output,'**','*.mat'));

parfor a = 1:numel(processed_matfiles)
    skip = [];

    % Load processed matfile
    mat_folder = processed_matfiles(a).folder;
    wk_mat = load(fullfile(mat_folder,processed_matfiles(a).name));

    % Find T1 scans/sessions
    sbj_name = extractBefore(processed_matfiles(a).name,'.mat');
    input = dir(fullfile(nii_preproc_database,sbj_name,'**','T1*'));

    for s = 1:numel(input)
        skip = false;

        % Define session name
        [~,ses] = fileparts(input(s).folder);

        % Skip if brainage already completed
        if isfield(wk_mat.(ses),'brainage')
            skip = true;
            continue
        end

        % Make subject folder
        wk_sbj = extractBefore(input(s).name,'.nii');
        wk_sbj_folder = fullfile(tempbrain_folder,wk_sbj);
        mkdir(wk_sbj_folder);

        % Copy T1 into brainage temp folder
        s_input = fullfile(wk_sbj_folder,input(s).name);
        copyfile(fullfile(input(s).folder,input(s).name),s_input);

        % Define output
        brainage_output = fullfile(wk_sbj_folder,[wk_sbj,'_brainage.csv']);

        % Run brainageR
        cmd = sprintf('%s -f %s -o %s',fullfile(brainage_path,'brainageR'),s_input,brainage_output);
        system(cmd)

        % Save brainageR output
        brain_age_pred = readtable(brainage_output);
        brain_age_pred_tissue_vols = readtable(fullfile(wk_sbj_folder,[brain_age_pred.File{:},'_tissue_volumes.csv']));
        wk_mat.(ses).brainage.agePred = brain_age_pred;
        wk_mat.(ses).brainage.TissueVol = brain_age_pred_tissue_vols;

        slice_dir = dir(fullfile(wk_sbj_folder,['slicesdir_T1_',extractBefore(processed_matfiles(a).name,'.mat'),'.nii'],'*.png'));
        for p = 1:numel(slice_dir)
            png = imread(fullfile(slice_dir(p).folder,slice_dir(p).name));
            t_name = extractBetween(slice_dir(p).name,'__','.png');
            wk_mat.(ses).brainage.slicedir.(t_name{:}) = png;
        end
    end

    % Skip if brainage already done
    if skip
        continue
    end

    % Save completed matfile with brainageR field
    saveparfor(fullfile(brainage_folder,processed_matfiles(a).name),'-struct',wk_mat)
end
end

function run_QC(post_qc,processed_matfolder,pipelineinfo,nii_proc_formated,opt)

CHECK_SESSIONMATCH = opt.CHECK_SESSIONMATCH;
CHECK_AQ = opt.CHECK_AQ;
CHECK_BRAINAGER = opt.CHECK_BRAINAGER;

% Find processed matfiles
subjects = dir(fullfile(processed_matfolder,'*.mat'));
subject_folder = subjects.folder;
subjects = {subjects.name};
parfor s = 1:numel(subjects)
    fn = [];

    % Define subject name
    wk_sbj = extractBefore(subjects{s},'.mat');

    % Load processed matfile
    wk_mat = load(fullfile(subject_folder,subjects{s}));

    % Allocated variables
    qc = [];

    % Check session matches
    input_ses = dir(fullfile(nii_proc_formated,wk_sbj));
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
                            qc.DTI = 'pass';
                        else
                            qc.DTI = 'fail';
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
end

function saveparfor(outname,opt1,save_mfile)
save(outname,opt1,'save_mfile')
end

function display_complete(stepnum,step,error_num)

if isempty(error_num)
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