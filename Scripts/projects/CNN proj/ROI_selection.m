patientfolder='C:\Users\allen\Box Sync\Eleni\Smoothed_Files_thr_0.2';
resliced_ROI_path='C:\Users\allen\Box Sync\Desktop\Allen_Bonilha_EEG\Atlas\mni152NLin2009casy\JHU\r75__Hippo_L.nii';

control_path=fullfile(patientfolder,'mod_0.2_smooth10_controls_gm');
patient_path=fullfile(patientfolder,'mod_0.2_smooth10_patients_left_gm');

%%
control_data={dir(fullfile(control_path,'*.nii')).name}';
patient_data={dir(fullfile(patient_path,'*.nii')).name}';

%%

% Load ROI
ROI_nii=load_nii(resliced_ROI_path);
ROI_nii_log=ROI_nii.img~=0;

% Control
for c=1:numel(control_data)
    tempnii=load_nii(control_data{c});
    temproinii=tempnii.img(ROI_nii_log);
    
    control_roi_data(c,:)=temproinii';
end


% Patients
for p=1:numel(patient_data)
    tempnii=load_nii(patient_data{p});
    temproinii=tempnii.img(ROI_nii_log);
    
    patient_roi_data(p,:)=temproinii';
end

% Concat All subjects (control=1, patient=2)
ROI_data.label=resliced_ROI_path;
ROI_data.all=[control_roi_data;patient_roi_data];
ROI_data.ident=[ones(numel(control_data),1);ones(numel(patient_data),1)*2];