
clear
clc

githubpath='/home/bonilha/Documents/GitHub/Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='/media/bonilha/Elements/Image_database';
save_path = '/media/bonilha/AllenProj/Thesis/niftifiles';
matter = 'gm';

%%

mkdir(save_path)
cd(save_path)

patients = dir(fullfile(PatientData,'*TLE*','post_qc','Patients','*.mat'));
controls = dir(fullfile(PatientData,'*TLE*','post_qc','Controls','*.mat'));

% Patients img
parfor pat=1:numel(patients)
    matoutput2nifti(fullfile(patients(pat).folder,patients(pat).name),fullfile(save_path,'Patients'),false,true);
end

% Control img
parfor con=1:numel(controls)
    matoutput2nifti(fullfile(controls(con).folder,controls(con).name),fullfile(save_path,'Controls'),false,true);
end
%%