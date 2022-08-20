%% This script is used to smooth cat12 segmented images 
% located under mri folder for the purpose of running an Independent t-test
% (Two-sample t-test) using the "Specify 2nd-level" of spm12. 
%--------------------------------------------------------------------------
% Notes: 
% The codes takes into acount that under the patients' folder there are
% seperate folders for left and right patients.
%
% "location" is the location of a test folder
% "location1" is the location of the folder containing the controls
% "location2" is the location of the folder containing the patients 
% 
%--------------------------------------------------------------------------

clear
clc

% Add github path
githubpath = '/home/bonilha/Documents/GitHub/Bonilha';
% githubpath = 'C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs: 
PatientData_folder = '/media/bonilha/AllenProj/PatientData/disease_dur/Cat12';
Smooth_folder='/media/bonilha/AllenProj/PatientData/disease_dur/smooth';mkdir(Smooth_folder)
Thres_folder='/media/bonilha/AllenProj/PatientData/disease_dur/thres_smooth';mkdir(Thres_folder)

Threshold=0.2;
%--------------------------------------------------------------------------

%% Detect folder
subfolder_path = dir(PatientData_folder).folder;
sub_list={dir(PatientData_folder).name};
sub_list=sub_list(~startsWith(sub_list,'.'));
for m=1:2
    if m==1
        matter='GM';
    elseif m==2
        matter='WM';
    end
    parfor sbj = 1:numel(sub_list)
        
        % Spm Smoothing
        tempsub=sub_list{sbj};
        tempsub_path=dir(fullfile(subfolder_path,tempsub,'mri',['mwp',num2str(m),'*']));
        tempsub_path = fullfile(tempsub_path.folder,tempsub_path.name);
        
        disp(['Running ',matter,' Subject ',tempsub])
        tempoutput_folder = fullfile(Smooth_folder,tempsub);
        smooth_output = fullfile(tempoutput_folder,['smooth10_',matter,'_',tempsub,'.nii']);
        mkdir(tempoutput_folder)
        spm_smooth(tempsub_path,smooth_output,[10 10 10]);
        
        % Thresholding
        smoothimg=load_nii(smooth_output);
        smoothimg.img(smoothimg.img<Threshold)=0;
        sbj_thres_folder = fullfile(Thres_folder,tempsub);
        mkdir(sbj_thres_folder)
        [~,file,ext] = fileparts(smooth_output);
        save_nii(smoothimg,fullfile(sbj_thres_folder,['thres0.2_',file,ext]));
    end
end
