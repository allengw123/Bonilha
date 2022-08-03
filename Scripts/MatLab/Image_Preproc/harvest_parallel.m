clear all
clc

% This code is used to process T1, Lesion, and RestingState MRIs
%%% Unable to do DTI in parallel --> use nii_harvest to do DTI serially
%%% Used to process Rebecca's MASTER_EPILEPSY DATABASE


%%%%%%%%%%%%%%%%%%%%%%%%%% Requirments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clone of Allen Chang's Github (https://github.com/allengw123/Bonilha)
% MATLAB's parallel processing toolbox 
% Developed using MATLAB vR2022 (likely need that version or higher)
% Nii_stat preprocessing configued computer
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% DON'T CHANGE CODE BELOW (unless you know what you are doing) %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Prepare Files for nii_harvest_parallel

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Find Database paths
raw_database = fullfile(INPUT_PATH,DISEASE_TAG);
if ~exist(raw_database,'dir')
    error([DISEASE_TAG,' Folder Not Found. Format is : path_to_master_set_folder/<patients variable name>'])
end
nii_preproc_database = fullfile(OUTPUT_PATH,'nii_proc_format',DISEASE_TAG);
mkdir(nii_preproc_database)
cd(nii_preproc_database)

% Prepare file for nii_harvest_parallel
[errors] = prep_niiharvest(raw_database,nii_preproc_database);
if ~isempty(errors)
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
else
    disp('Format Completed')
end


%% Run nii_harvest_parallel
harvest_output = '/media/bonilha/AllenProj/MasterSet/harvestOutput/TLE';
mkdir(harvest_output)

nii_harvest_parallel(nii_preproc_database,harvest_output)
%% nii_preprocess_subfolders 
nii_preprocess_subfolders(harvest_output)


%% Brain Age
disp('Step 3 - Running Brain Age')
cd(fullfile(brainage_path,'software'))

% Prep Brain Age Files
setup_brainagedir(brainage_path,spm_path)

% Make brain age folder
brain_age_folder = fullfile(subjectdir,'BrainAge');
mkdir(brain_age_folder)

% Detect T1 acquisitions
t1_acq = dir(fullfile(rawdir,'*T1*'));
disp([num2str(numel(t1_acq)),' ... T1 Acquisitions Detected'])

% Copy T1 acquisitions
for a = 1%:numel(t1_acq)
    
    % Copy Raw files over
    current_aq_name = char(extractBetween(t1_acq(a).name,[subjectname,'_'],'.nii'));
    aq_folder = fullfile(brain_age_folder,current_aq_name);
    mkdir(aq_folder)
    niftifile = fullfile(t1_acq(a).folder,t1_acq(a).name);
    copyfile(niftifile,aq_folder)

    input = niftifile;
    output = fullfile(aq_folder,[subjectname,'_T1_brain_predicted.csv']);
    
    cmd = sprintf('brainageR.sh -f %s -o %s',input,output);
    system(cmd)
    
end
            
%% Functions

function [error_output] = prep_niiharvest(database_path,output_database)
error_output = [];
% Obtain Directories
sbj_dir = dir(database_path);
output_dir = dir(output_database);

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

% Parallel Processing REQUIRED
parfor sbj = 1:numel(sbj_dir)
    warning('off','all')
    try
        if startsWith(sbj_dir(sbj).name,'.')
            continue
        end
    
        % Find Subject Name/Folder
        subject_name = sbj_dir(sbj).name;
        subject_folder = sbj_dir(sbj).folder;
    
        % Define New Subject name (no '_' allowed)
        new_subject_name = strrep(subject_name,'_','');

        % Find Session Type based on T1 labels 
        s_dir = dir(fullfile(subject_folder,subject_name,'*T1*'));
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
        if numel(t1_names) == numel(aq)
        else
            errors{sbj} = 'Number of unique Sessions do not match number of unique T1 (T1 file naming error)';
            sbj_error{sbj} = sbj_dir(sbj).name;
            continue
        end
    
        cont = true;
        while cont

            % Transfer Modalities of each Aquisition Session
            modality = [];
            status = [];
            for a = 1:numel(aq)
                aq_type = aq{a};
        
                % Detect T1/RS/LS files
                if strcmp(aq_type,'session')
                    T1 = dir(fullfile(database_path,subject_name,'*T1*.nii*'));
                    T2 = dir(fullfile(database_path,subject_name,'*T2*.nii*'));
                    RS = dir(fullfile(database_path,subject_name,'*rs*.nii*'));
                    dti = dir(fullfile(database_path,subject_name,'*diff*.nii*'));
                    lesion = []; %THERE IS NO LESION FOR SESSION
                elseif strcmp(aq_type,'pre')
                    T1 = dir(fullfile(database_path,subject_name,['*',aq_type,'*T1*.nii*']));
                    T2 = dir(fullfile(database_path,subject_name,['*',aq_type,'*T2*.nii*']));
                    RS = dir(fullfile(database_path,subject_name,['*',aq_type,'*rs*.nii*']));
                    lesion = []; %THERE IS NO LESION FOR PRE
                    dti = dir(fullfile(database_path,subject_name,['*',aq_type,'*diff*.nii*']));
                elseif strcmp(aq_type,'pos')
                    T1 = dir(fullfile(database_path,subject_name,['*',aq_type,'*T1*.nii*']));
                    T2 = dir(fullfile(database_path,subject_name,['*',aq_type,'*T2*.nii*']));
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
    
                % Define pipeline Status file and check if it has been completed
                statusFile = fullfile(output_database,new_subject_name,'PipelineStatus.txt');
                if any(exist(statusFile,'file'))
                    num_files = numel(dir(fullfile(output_database,new_subject_name,aq_type,['*',new_subject_name,'*'])));
                    num_files_detected = sum([~isempty(T1)
                        ~isempty(T2)
                        ~isempty(RS)*2
                        ~isempty(lesion)
                        ~isempty(dti)*4]);
                    if num_files_detected == num_files
                        cont = false;
                        break
                    end
                end
    
                %%%%%%%%%%%%%%%%%%%%% Transfer T1
                % Remove Secondary
                T1 = T1(cellfun(@isempty,extractBetween({T1.name},'T1','.nii')));
                if isempty(T1)
                    errors{sbj} = [aq_type,' secondary scan removal failed for T1'];
                    sbj_error{sbj} = sbj_dir(sbj).name;
                    rmdir(subject_folder,'s');
                    cont = false;
                    break
                end
            
                % Record Status
                modality = [modality;{'T1'}];
                status = [status;{fullfile(T1.folder,T1.name)}];
        
                % Make Subject Image Folders
                subject_folder = fullfile(output_database,new_subject_name,aq_type);
                mkdir(subject_folder)
            
                % Transfer T1 image
                T1_new = ['T1_',new_subject_name,'.nii'];
                if contains(T1.name,'.gz')
                    filenames = gunzip(fullfile(T1.folder,T1.name));
                    movefile(filenames{:},fullfile(subject_folder,T1_new));
                else
                    copyfile(fullfile(T1.folder,T1.name),fullfile(subject_folder,T1_new))
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
                        rmdir(subject_folder,'s');
                        cont = false;
                        break
                    end
                    % Record Status
                    modality = [modality;{'T2'}];
                    status = [status;{fullfile(T2.folder,T2.name)}];
                
                    % Transfer T2 image
                    T2_new = ['T2_',new_subject_name,'.nii'];
                    if contains(T2.name,'.gz')
                        filenames = gunzip(fullfile(T2.folder,T2.name));
                        movefile(filenames{:},fullfile(subject_folder,T2_new));
                    else
                        copyfile(fullfile(T2.folder,T2.name),fullfile(subject_folder,T2_new))
                    end
                end

                %%%%%%%%%%%%%%%%%%% Transfer Resting State
                if isempty(RS)
                    % Record Status
                    modality = [modality;{'RS'}];
                    status = [status;{'RS not found'}];
                else
                    for r = 1:numel(RS)
    
                        % Check json file
                        rs_json = fullfile(RS(r).folder,[extractBefore(RS(r).name,'.nii'),'.json']);
                        if ~any(exist(rs_json,'file'))
                            if r == numel(rs)
                                errors{sbj} = [aq_type,'_rs.json file missing'];
                                sbj_error{sbj} = sbj_dir(sbj).name;
                                rmdir(subject_folder,'s');
                                cont = false;
                                break
                            end
                            continue
                        end
                        
                        % Transfer RS image
                        RS_new = ['Rest_',new_subject_name,'.nii'];
                        if contains(RS(r).name,'.gz')
                            filenames = gunzip(fullfile(RS(r).folder,RS(r).name));
                            movefile(filenames{:},fullfile(subject_folder,RS_new));
                        else
                            copyfile(fullfile(RS(r).folder,RS(r).name),fullfile(subject_folder,RS_new))
                        end
    
                        % Copy RS json
                        copyfile(rs_json,fullfile(subject_folder,strrep(RS_new,'.nii','.json')))
                        
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
                        movefile(filenames{:},fullfile(subject_folder,lesion_new));
                    else
                        copyfile(fullfile(lesion.folder,lesion.name),fullfile(subject_folder,lesion_new))
                    end
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
                                rmdir(subject_folder,'s');
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
                                rmdir(subject_folder,'s');
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
                            % If <12 try second scan
                            modality = [modality;{'DTI'}];
                            status = [status;{[aq_type,'_DTI.bval values less than 12(',num2str(n),')']}];
                            
                            % If all scans fail to have >11 then throw error
                            if d == numel(dti)
                                errors{sbj} = aq_type,'_DTI.bval values <12';
                                sbj_error{sbj} = sbj_dir(sbj).name;
                                rmdir(subject_folder,'s');
                                cont = false;
                            end
                        else
    
                            % Copy DTI scan
                            dti_new = ['DTI_',new_subject_name,'.nii'];
                            if contains(dti(d).name,'.gz')
                                filenames = gunzip(fullfile(dti(d).folder,dti(d).name));
                                movefile(filenames{:},fullfile(subject_folder,dti_new));
                            else
                                copyfile(fullfile(dti(d).folder,dti(d).name),fullfile(subject_folder,dti_new))
                            end
    
                            % Copy DTI bval/bvec/json
                            copyfile(bval,fullfile(subject_folder,['DTI_',new_subject_name,'.bval']))
                            copyfile(bvec,fullfile(subject_folder,['DTI_',new_subject_name,'.bvec']))
                            copyfile(dti_json,fullfile(subject_folder,['DTI_',new_subject_name,'.json']))
    
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
        rmdir(subject_folder,'s');
    end
end

% Check if there were any errors
errors(cellfun(@isempty,errors)) = [];
sbj_error(cellfun(@isempty,sbj_error)) = [];
clc
if ~isempty(errors)
    error_output =[sbj_error' errors'];
end

end

function setup_brainagedir(brainage_path,spm_path)
    
% Detect brainage SH file
brainage_file_path = fullfile(brainage_path,'software','brainageR');

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

% Replace brainageR_dir path
idx = find(~cellfun(@isempty,(regexp(fileContents,'brainageR_dir'))));
brainageR_dir_line = sprintf('brainageR_dir=%s',brainage_path);
fileContents{idx(1)} = brainageR_dir_line;

% Replace spm_dir path
idx = find(~cellfun(@isempty,(regexp(fileContents,'spm_dir'))));
spm_path_line = sprintf('spm_dir=%s',spm_path);
fileContents{idx(1)} = spm_path_line;

% Replace matlab_path path
idx = find(~cellfun(@isempty,(regexp(fileContents,'matlab_path'))));
matlab_path_line = sprintf('matlab_path=%s',matlabroot);
fileContents{idx(1)} = matlab_path_line;

% write/replace brainageR sh file
brainage_file = fopen(brainage_file_path,'w');
fprintf(brainage_file,'%s\n',string(fileContents));
fclose(brainage_file);
end