clear all
clc

% Gen path
%gitpath='C:\Users\allen\Google Drive\GitHub\Bonilha';
gitpath = '/home/allenchang/Documents/GitHub/Bonilha';
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
input_database = '/home/allenchang/Downloads/Sample_Participants';
cd(input_database)
database_dir = dir(input_database);

output_database = '/home/allenchang/Downloads/TestSub';
mkdir(output_database)

% Modify folder to niistat structure
for s = 1:numel(database_dir)
    if strcmp(database_dir(s).name,'.') || strcmp(database_dir(s).name,'..') || strcmp(database_dir(s).name,'.DS_Store')
        continue
    end
    
    % Move files
    subject_name = prep_niiharvest(database_dir(s).folder,database_dir(s).name,'pre',output_database);
    subject_name = prep_niiharvest(database_dir(s).folder,database_dir(s).name,'post',output_database);
end
%%
cd(output_database)

harvest_output = '/home/allenchang/Downloads/harvestOutput';;
mkdir(harvest_output)
cd(harvest_output)
nii_harvest(output_database,harvest_output)



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
end
T1 = fullfile(T1.folder,T1.name);
T1_new = ['T1_',new_subject_name,'.nii'];
copyfile(T1,fullfile(subject_folder,T1_new))

% Transfer fMRI
fMRI = strrep(T1,'T1','rs');
if any(exist(fMRI,'file'))
    copyfile(fMRI,fullfile(subject_folder,strrep(T1_new,'T1','Rest')))
end

% Transfer DTI
dti = strrep(T1,'T1','diff');
if any(exist(dti,'file'))
    copyfile(dti,fullfile(subject_folder,strrep(T1_new,'T1','DTI')))
    copyfile(strrep(dti,'nii','bval'),fullfile(subject_folder,strrep(strrep(T1_new,'T1','DTI'),'nii','bval')))
    copyfile(strrep(dti,'nii','bvec'),fullfile(subject_folder,strrep(strrep(T1_new,'T1','DTI'),'nii','bvec')))
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