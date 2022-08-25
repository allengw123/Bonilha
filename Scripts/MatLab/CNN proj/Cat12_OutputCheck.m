
clear
clc

% Add github path
githubpath = '/home/bonilha/Documents/GitHub/Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

data_dir = '/media/bonilha/AllenProj/CNN_project/PatientData/smallSet/Cat12_segmented';

%%

patients_dir = dir(fullfile(data_dir,'*TLE*','*'));
patients_dir(contains({patients_dir.name},'.')) = [];

control_dir = dir(fullfile(data_dir,'EP_CN_nifti/'));
control_dir(contains({control_dir.name},'.')) = [];

mri_p = cell(size(patients_dir));
gm_mri_p = cell(size(patients_dir));
gm_v_p_abs = cell(size(patients_dir));
gm_v_p_rel = cell(size(patients_dir));
parfor p = 1:numel(patients_dir)
    mri_file = dir(fullfile(patients_dir(p).folder,patients_dir(p).name,'*.nii'));
    mri_file_load = load_nii(fullfile(mri_file.folder,mri_file.name));
    mri_p{p} = mri_file_load.img;

    gm_file = dir(fullfile(patients_dir(p).folder,patients_dir(p).name,'mri','*p1*.nii'));
    gm_file(contains({gm_file.name},'smooth')) = [];
    gm_file_load = load_nii(fullfile(gm_file.folder,gm_file.name));
    gm_mri_p{p} = gm_file_load.img;

    gm_v_p = dir(fullfile(patients_dir(p).folder,patients_dir(p).name,'report','*.mat'));
    gm_v_p_load = load(fullfile(gm_v_p.folder,gm_v_p.name)).S.subjectmeasures;
    gm_v_p_abs{p} = gm_v_p_load.vol_abs_CGW;
    gm_v_p_rel{p} = gm_v_p_load.vol_rel_CGW;
end


mri_c = cell(size(control_dir));
gm_mri_c = cell(size(control_dir));
gm_v_c_abs = cell(size(control_dir));
gm_v_c_rel = cell(size(control_dir));
parfor p = 1:numel(control_dir)
    try
        mri_file = dir(fullfile(control_dir(p).folder,control_dir(p).name,'*.nii'));
        mri_file_load = load_nii(fullfile(mri_file.folder,mri_file.name));
        mri_c{p} = mri_file_load.img;

        gm_file = dir(fullfile(control_dir(p).folder,control_dir(p).name,'mri','*p1*.nii'));
        gm_file(contains({gm_file.name},'smooth')) = [];
        gm_file_load = load_nii(fullfile(gm_file.folder,gm_file.name));
        gm_mri_c{p} = gm_file_load.img;

        gm_v_c = dir(fullfile(control_dir(p).folder,control_dir(p).name,'report','*.mat'));
        gm_v_c_load = load(fullfile(gm_v_c.folder,gm_v_c.name)).S.subjectmeasures;
        gm_v_c_abs{p} = gm_v_c_load.vol_abs_CGW;
        gm_v_c_rel{p} = gm_v_c_load.vol_rel_CGW;
    catch
        disp(['Error in ',control_dir(p).name])
    end
end

error_idx = cellfun(@isempty,mri_c);
mri_c(error_idx) = [];
gm_mri_c(error_idx) = [];
gm_v_c_abs(error_idx) = [];
gm_v_c_rel(error_idx) = [];
%%

% raw mri intensity difference
figure
mri_mean_intensity_p =cellfun(@(x) mean(x,"all"),mri_p);
mri_mean_intensity_c =cellfun(@(x) mean(x,"all"),mri_c);
bar([mean(mri_mean_intensity_p,'omitnan');mean(mri_mean_intensity_c,'omitnan')])
hold on
scatter([ones(size(mri_mean_intensity_p));ones(size(mri_mean_intensity_c))*2],[mri_mean_intensity_p;mri_mean_intensity_c])
[~,p,~,~] = ttest2(mri_mean_intensity_p,mri_mean_intensity_c);
title(['Raw Mri Intensity Diff (',num2str(p),')'])
xticklabels({'Patients','Controls'})


% GM mri intensity difference
figure
mri_mean_intensity_p =cellfun(@(x) mean(x,"all"),gm_mri_p);
mri_mean_intensity_c =cellfun(@(x) mean(x,"all"),gm_mri_c);
bar([mean(mri_mean_intensity_p,'omitnan');mean(mri_mean_intensity_c,'omitnan')])
hold on
scatter([ones(size(mri_mean_intensity_p));ones(size(mri_mean_intensity_c))*2],[mri_mean_intensity_p;mri_mean_intensity_c])
[h,p,~,~] = ttest2(mri_mean_intensity_p,mri_mean_intensity_c);
title(['GM Mri Intensity Diff (',num2str(p),')'])
xticklabels({'Patients','Controls'})


% GM volume abs difference
figure
mri_mean_intensity_p =cellfun(@(x) x(2),gm_v_p_abs);
mri_mean_intensity_c =cellfun(@(x) x(2),gm_v_c_abs);
bar([mean(mri_mean_intensity_p,'omitnan');mean(mri_mean_intensity_c,'omitnan')])
hold on
scatter([ones(size(mri_mean_intensity_p));ones(size(mri_mean_intensity_c))*2],[mri_mean_intensity_p;mri_mean_intensity_c])
[~,p,~,~] = ttest2(mri_mean_intensity_p,mri_mean_intensity_c);
title(['GM vol abs Diff (',num2str(p),')'])
xticklabels({'Patients','Controls'})


% GM volume rel difference
figure
mri_mean_intensity_p =cellfun(@(x) x(2),gm_v_p_rel);
mri_mean_intensity_c =cellfun(@(x) x(2),gm_v_c_rel);
bar([mean(mri_mean_intensity_p,'omitnan');mean(mri_mean_intensity_c,'omitnan')])
hold on
scatter([ones(size(mri_mean_intensity_p));ones(size(mri_mean_intensity_c))*2],[mri_mean_intensity_p;mri_mean_intensity_c])
[~,p,~,~] = ttest2(mri_mean_intensity_p,mri_mean_intensity_c);
title(['GM vol rel Diff (',num2str(p),')'])
xticklabels({'Patients','Controls'})


% GM  mri intensity difference
figure
mri_mean_intensity_p =cellfun(@(x) mean(x,"all"),gm_mri_p);
mri_mean_intensity_p = mri_mean_intensity_p./cellfun(@(x) x(2),gm_v_p_abs);
mri_mean_intensity_c =cellfun(@(x) mean(x,"all"),gm_mri_c);
mri_mean_intensity_c = mri_mean_intensity_c./cellfun(@(x) x(2),gm_v_c_abs);
bar([mean(mri_mean_intensity_p,'omitnan');mean(mri_mean_intensity_c,'omitnan')])
hold on
scatter([ones(size(mri_mean_intensity_p));ones(size(mri_mean_intensity_c))*2],[mri_mean_intensity_p;mri_mean_intensity_c])
[~,p,~,~] = ttest2(mri_mean_intensity_p,mri_mean_intensity_c);
title(['GM Mri Intensity Adjusted Diff (',num2str(p),')'])
xticklabels({'Patients','Controls'})
