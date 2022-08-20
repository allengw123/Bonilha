%% CNN MODEL (Confounding Factor)
clear
clc

githubpath = '/home/bonilha/Documents/GitHub/Bonilha';
% githubpath = 'C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
SmoothThres='/media/bonilha/AllenProj/PatientData/disease_dur/thres_smooth';
cd(SmoothThres)

save_path='/media/bonilha/AllenProj/PatientData/disease_dur/age_reg';
mkdir(save_path)
%% Calculate Residuals

% Read excel files
xml_info=readtable('/media/bonilha/AllenProj/PatientData/disease_dur/PatientInfo.xlsx');
directory = dir(SmoothThres);
sbj_files = {directory.name};
sbj_files = sbj_files(~startsWith(sbj_files,'.'));

g_img = [];
g_age = [];
sbj = [];
count = 1;
for con=1:numel(sbj_files)

    % Find Age
    temp_age = xml_info.Age(strcmp(sbj_files{con},xml_info.NewID));
    if isempty(temp_age)
        continue
    end

    % Load image
    gm_img = dir(fullfile(directory(1).folder,sbj_files{con},'*GM*'));
    temp=load_nii(fullfile(gm_img.folder,gm_img.name));

    % Save Image and Age
    g_img{count}=temp.img(:,:,28:85);
    g_age{count,1}=temp_age;
    sbj{count,1} = sbj_files{count};

    count = count+1;
end

warning('off','all')
lin_relationship = [];
template = [];
residual_imgs = cell(897898,1);
parfor vox = 1:897898
    
    intensities = [];
    age = [];
    for sub = 1:numel(g_img)
        intensities = [intensities;g_img{sub}(vox)];
        age = [age;g_age{sub}];
    end

    temp_residual = [];
    if any(intensities)
        mdl=LinearModel.fit(age,intensities);
        residuals = mdl.Residuals.('Raw');
        for s = 1:numel(residuals)
            temp_residual = [temp_residual,residuals(s)];
        end
    else
        for s = 1:numel(intensities)
            temp_residual= [temp_residual,0];
        end
    end
    
    residual_imgs{vox} = temp_residual';
end

residual_imgs = cat(2,residual_imgs{:});
reshaped_residuals = [];
for r = 1:size(residual_imgs,1)
    reshaped_residuals{r,1} = reshape(residual_imgs(r,:),113,137,58);
end

save(fullfile(save_path,'calculated_reg.mat'),"reshaped_residuals","sbj","g_age")