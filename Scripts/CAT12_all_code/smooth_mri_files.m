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
% Inputs: 
PatientData_folder = 'F:\PatientData\Cat12_segmented';
Smooth_folder='F:\PatientData\smooth';mkdir(Smooth_folder)
Thres_folder='F:\PatientData\thres';

Threshold=0.2;
%--------------------------------------------------------------------------

%% Detect folder
subject_folders={dir(fullfile(PatientData_folder,'*nifti')).name};


for sf = 1:numel(subject_folders)
    tempsubfolder_path=fullfile(PatientData_folder,subject_folders{sf});
    tempsub_list={dir(tempsubfolder_path).name};
    tempsub_list=tempsub_list(~startsWith(tempsub_list,'.'));
    for m=1:2
        if m==1
            matter='GM';
        else
            matter='WM';
        end
        for sbj = 1:numel(tempsub_list)
            
            % Spm Smoothing
            tempsub=tempsub_list{sbj};
            tempsub_path=fullfile(tempsubfolder_path,tempsub,'mri',dir(fullfile(tempsubfolder_path,tempsub,'mri',['mwp',num2str(m),'*'])).name);
            
            disp(['Running ',matter,' Subject ',tempsub])
            mkdir(fullfile(Smooth_folder,subject_folders{sf},tempsub))
            spm_smooth(tempsub_path,...
                fullfile(Smooth_folder,subject_folders{sf},tempsub,['smooth10_',matter,'_',tempsub,'.nii']),...
                [10 10 10])
            
            % Thresholding
            tempimg=load_nii(fullfile(Smooth_folder,subject_folders{sf},tempsub,['smooth10_',matter,'_',tempsub,'.nii']));
            tempimg.img(tempimg.img<Threshold)=0;
            mkdir(fullfile(Thres_folder,subject_folders{sf},tempsub))
            save_nii(tempimg,fullfile(Thres_folder,subject_folders{sf},tempsub,['smooth10_',matter,'_',tempsub,'.nii']));
        end
    end
end
