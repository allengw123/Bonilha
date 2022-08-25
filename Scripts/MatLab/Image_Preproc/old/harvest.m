clear all
clc

% Gen path
% gitpath='C:\Users\allen\Google Drive\GitHub\Bonilha';
% gitpath = '/home/allenchang/Documents/GitHub/Bonilha';
gitpath = '/home/bonilha/Documents/GitHub/Bonilha';
cd(gitpath)
allengit_genpath(gitpath,'imaging')


%%%% DON'T CHANGE (unless paths are incorrect; paths should be correct if
%%%% cloning git)

% SPM
spm_path=fullfile(gitpath,'Toolbox','imaging','spm12');

% DSI studio
dsipath=fullfile(gitpath,'Toolbox','imaging','dsi_studio_64');

% Brain age
brainage_path = fullfile(gitpath,'Toolbox','imaging','brainage');

%%
raw = '/media/bonilha/AllenProj/MasterSet/raw/controls';
database_dir = dir(raw);

nii_preproc_database = '/media/bonilha/AllenProj/MasterSet/nii_proc_format/controls';
mkdir(nii_preproc_database)

harvest_output = '/media/bonilha/AllenProj/MasterSet/harvestOutput/controls';
mkdir(harvest_output)
cd(harvest_output)

error_sbj = [];
error_msg = [];
% Modify folder to niistat structure
for s = 3%1:numel(database_dir)
    if strcmp(database_dir(s).name,'.') || strcmp(database_dir(s).name,'..') || strcmp(database_dir(s).name,'.DS_Store')
        continue
    end

    formated_sbj_name = strrep(database_dir(s).name,'_','');
    % Detect if "New" or "Complete/Incomplete"
    if any(exist(fullfile(nii_preproc_database,formated_sbj_name,'PipelineStatus.txt'),'file'))
        fileID = fopen(fullfile(nii_preproc_database,formated_sbj_name,'PipelineStatus.txt'),'r');
        lines = textscan(fileID,'%s', 'Delimiter', '\n');
        fclose(fileID);
        if strcmp(lines{:}{end},'COMPLETE')
            continue
        else
            incomplete_dir = fullfile(harvest_output,'incomplete');
            mkdir(incomplete_dir)
            id = strrep(strrep(strrep(char(datetime),'-','_'),':','_'),' ','');
            try
                movefile(fullfile(harvest_output,formated_sbj_name),fullfile(incomplete_dir,[formated_sbj_name,'_',id]))
                warning(['Moving INCOMPLETE HARVEST OUTPUT to ',incomplete_dir])
            catch
            end
        end
    end
    
    try

        % Prepare file and run nii_harvest
        start_niiharvest(database_dir(s).folder, ...
            database_dir(s).name, ...
            nii_preproc_database, ...
            harvest_output);
    catch e
        warning(['Error in subject ',database_dir(s).name,' Please check error_msg variable'])
        error_msg = [error_msg {e}];
        error_sbj = [error_sbj {database_dir(s).name}];
    end
end

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
%%

% Modify folder to niistat structure
parfor s = 1:numel(database_dir)
    if strcmp(database_dir(s).name,'.') || strcmp(database_dir(s).name,'..') || strcmp(database_dir(s).name,'.DS_Store')
        continue
    end
    
    s_dir = dir(fullfile(database_dir(s).folder,database_dir(s).name));
    s_dir(strcmp({s_dir.name},'.')) = [];
    s_dir(strcmp({s_dir.name},'..')) = [];

    if any(contains({s_dir.name},'Processed'))
        idx = strcmp({s_dir.name},'Processed');
        rmdir(fullfile(s_dir(idx).folder,s_dir(idx).name),"s");
    end
    
    if  any(contains({s_dir.name},'Raw'))
        idx = strcmp({s_dir.name},'Raw');
        r_dir = dir(fullfile(s_dir(idx).folder,s_dir(idx).name));
        r_dir(strcmp({r_dir.name},'.')) = [];
        r_dir(strcmp({r_dir.name},'..')) = [];
        for i = 1:numel(r_dir)
            [path,name,~] = fileparts(r_dir(i).folder);
            movefile(fullfile(r_dir(i).folder,r_dir(i).name),fullfile(path,r_dir(i).name))
        end
        rmdir(fullfile(path,name),"s")
    end
end
            
%% Functions

function new_subject_name = start_niiharvest(database_path,subject_name,output_database,harvest_output)

new_subject_name = strrep(subject_name,'_','');
statusFile = fullfile(output_database,new_subject_name,'PipelineStatus.txt');

s_dir = dir(fullfile(database_path,subject_name,'*T1*'));
t1_names = {s_dir.name};
aq = [];
if any(contains(t1_names,'pre'))
    aq = [aq,{'pre'}];
end
if any(contains(t1_names,'post'))
    aq = [aq,{'post'}];
end
if isempty(aq)
    aq = [aq,{'session'}];
end



modality = [];
status = [];
for a = 1:numel(aq)

    aq_type = aq{a};

    % Transfer T1
    if strcmp(aq_type,'session')
        T1 = dir(fullfile(database_path,subject_name,'*T1*'));
        fMRI = dir(fullfile(database_path,subject_name,'*rs*'));
        dti = dir(fullfile(database_path,subject_name,'*diff.nii*'));
    else
        T1 = dir(fullfile(database_path,subject_name,['*',aq_type,'*T1*']));
        fMRI = dir(fullfile(database_path,subject_name,['*',aq_type,'*rs*']));
        dti = dir(fullfile(database_path,subject_name,['*',aq_type,'*diff.nii*']));
    end
    T1(contains({T1.name},'_b.')) = [];
    if isempty(T1)
        modality = [modality;{'T1'}];
        status = [status;{'T1 not found'}];
        continue
    else
        modality = [modality;{'T1'}];
        status = [status;{fullfile(T1.folder,T1.name)}];

        subject_folder = fullfile(output_database,new_subject_name,aq_type);
        mkdir(subject_folder)
        T1_new = ['T1_',new_subject_name,'.nii'];
    
        if contains(T1.name,'.gz')
            filenames = gunzip(fullfile(T1.folder,T1.name));
            movefile(filenames{:},fullfile(subject_folder,T1_new));
        else
            copyfile(fullfile(T1.folder,T1.name),fullfile(subject_folder,T1_new))
        end
    end
    
    % Transfer fMRI
    if isempty(fMRI)
        modality = [modality;{'fMRI'}];
        status = [status;{'fMRI not found'}];
    else
        fMRI_new = ['Rest_',new_subject_name,'.nii'];
        status = [status;{fullfile(fMRI.folder,fMRI.name)}];

        if contains(fMRI.name,'.gz')
            filenames = gunzip(fullfile(fMRI.folder,fMRI.name));
            movefile(filenames{:},fullfile(subject_folder,fMRI_new));
        else
            copyfile(fullfile(fMRI.folder,fMRI.name),fullfile(subject_folder,fMRI_new))
        end
    end
    
    % Transfer DTI
    if isempty(dti)
        modality = [modality;{'DTI'}];
        status = [status;{'DTI not found'}];
    else
        % Check bval tables to make sure >12
        bval = fullfile(dti.folder,[extractBefore(dti.name,'.nii'),'.bval']);
        bvec = fullfile(dti.folder,[extractBefore(dti.name,'.nii'),'.bvec']);
    
        fileID = fopen(bval,'r');
        [~, n] = fscanf(fileID,'%g');
        fclose(fileID);
        if n < 12
            modality = [modality;{'DTI'}];
            status = [status;{['bval values less than 12(',num2str(n),')']}];
        else
            modality = [modality;{'DTI'}];
            status = [status;{fullfile(dti.folder,dti.name)}];
            dti_new = ['DTI_',new_subject_name,'.nii'];
            if contains(dti.name,'.gz')
                filenames = gunzip(fullfile(dti.folder,dti.name));
                movefile(filenames{:},fullfile(subject_folder,dti_new));
            else
                copyfile(fullfile(dti.folder,dti.name),fullfile(subject_folder,dti_new))
            end
            
            copyfile(bval,fullfile(subject_folder,['DTI_',new_subject_name,'.bval']))
            copyfile(bvec,fullfile(subject_folder,['DTI_',new_subject_name,'.bvec']))
        end
    end
        
    % Create file note
    fid = fopen(statusFile, 'wt' );
    fprintf(fid, '%s\n',aq_type);
    for i = 1:numel(modality)
        fprintf(fid, '-%s ... %s\n',modality{i}, status{i});
    end
    fclose(fid);

    try
        nii_harvest(output_database,harvest_output)
        writematrix('COMPLTETE',statusFile,'WriteMode','append')
    catch e
        writematrix('INCOMPLETE/FAILURE',statusFile,'WriteMode','append')
        writematrix(e,statusFile,'WriteMode','append')
    end

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