clear all
clc

% Gen path
%gitpath='C:\Users\allen\Google Drive\GitHub\Bonilha';
gitpath = '/home/allenchang/Documents/GitHub/Bonilha';
cd(gitpath)
allengit_genpath(gitpath,'imaging')
dcm2niix_dir = fullfile(gitpath,'Toolbox','imaging','dcm2niix-master');


DICOM_dir = '/home/allenchang/Downloads/DICOM_database';
NIFTI_dir = '/home/allenchang/Downloads/Nifti_database';
BIDS_database = '/home/allenchang/Downloads/BIDS_database';

%% DICOM to NIFTI

mkdir(NIFTI_dir)

cd(DICOM_dir)
database_dir = dir(DICOM_dir);

for s = 1:numel(database_dir)
    if strcmp(database_dir(s).name,'.') || strcmp(database_dir(s).name,'..') || strcmp(database_dir(s).name,'.DS_Store')
        continue
    end

    dicom_input = fullfile(database_dir(s).folder,database_dir(s).name);
    nifti_output = fullfile(NIFTI_dir,database_dir(s).name);
    mkdir(nifti_output)
    cmd = ['dcm2niix -z y -f %s_%p_%q_%e_%d -o ',nifti_output,' ',dicom_input];
    system(cmd)
end

%% NIFTI to BIDS format

mkdir(BIDS_database)
cd(NIFTI_dir)

nifti_input = dir(NIFTI_dir);
for s = 1:numel(nifti_input)
    if strcmp(database_dir(s).name,'.') || strcmp(database_dir(s).name,'..') || strcmp(database_dir(s).name,'.DS_Store')
        continue
    end

end
