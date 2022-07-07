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
raw = '/home/bonilha/Documents/Nii_preproc/raw';
cd(raw)
database_dir = dir(raw);

BIDS_database = '/home/bonilha/Documents/Nii_preproc/BIDS';
mkdir(BIDS_database)

% Modify folder to niistat structure
for s = 1:numel(database_dir)
    if strcmp(database_dir(s).name,'.') || strcmp(database_dir(s).name,'..') || strcmp(database_dir(s).name,'.DS_Store')
        continue
    end
    
    % Move files
    subject_name = prep_niiharvest(database_dir(s).folder,database_dir(s).name,'pre',BIDS_database);
    subject_name = prep_niiharvest(database_dir(s).folder,database_dir(s).name,'post',BIDS_database);
end
%%
cd(BIDS_database)

harvest_output = '/home/bonilha/Documents/Nii_preproc/harvestOutput';
mkdir(harvest_output)
cd(harvest_output)
nii_harvest(BIDS_database,harvest_output)



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

function new_subject_name = prep_niiharvest(database_path,subject_name,aq_type,output_database)

new_subject_name = strrep(subject_name,'_','');

% Transfer T1
T1 = dir(fullfile(database_path,subject_name,['*',aq_type,'*T1*']));
T1(contains({T1.name},'_b.')) = [];
if isempty(T1)
    return
else
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
fMRI = dir(fullfile(database_path,subject_name,['*',aq_type,'*rs*']));
if ~isempty(fMRI)
    fMRI_new = ['Rest_',new_subject_name,'.nii'];
    if contains(fMRI.name,'.gz')
        filenames = gunzip(fullfile(fMRI.folder,fMRI.name));
        movefile(filenames{:},fullfile(subject_folder,fMRI_new));
    else
        copyfile(fullfile(fMRI.folder,fMRI.name),fullfile(subject_folder,fMRI_new))
    end
end

% Transfer DTI
dti = dir(fullfile(database_path,subject_name,['*',aq_type,'*diff.nii*']));
if ~isempty(dti)
    
    % Check bval tables to make sure >12
    bval = fullfile(dti.folder,[extractBefore(dti.name,'.nii'),'.bval']);
    bvec = fullfile(dti.folder,[extractBefore(dti.name,'.nii'),'.bvec']);

    fileID = fopen(bval,'r');
    [~, n] = fscanf(fileID,'%g');
    fclose(fileID);
    if n < 12
        return
    else
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