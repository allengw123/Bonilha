%% CNN MODEL (Confounding Factor)
clear
clc

githubpath='/home/bonilha/Documents/GitHub/Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='/media/bonilha/AllenProj/CNN_project/PatientData/smallSet';
cd(PatientData)


SmoothThres=fullfile(PatientData,'thres');
addpath(genpath(SmoothThres));

matter='GM';
%% Linear regress

% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Control','**',['*',matter,'*.nii'])).name}';

ep_controls_info=readtable(fullfile(PatientData,'ep_controls_info.xlsx'));
ADNI_CN_info=readtable(fullfile(PatientData,'ADNI_CN_info.csv'));

ADNI_log = contains(controlfiles,'ADNI');

[ADNI_Control_GM, ADNI_Control_GM_names] = get_volume_data(controlfiles(ADNI_log));
[TLE_Control_GM, TLE_Control_GM_names] = get_volume_data(controlfiles(~ADNI_log));

%% Calculate Residuals

ADNI_age = cellfun(@(x) ADNI_CN_info.Age(contains(ADNI_CN_info.ImageDataID,x)),extractBetween(ADNI_Control_GM_names,'_I','.nii'));
TLE_age =cellfun(@(x) ep_controls_info.Age(contains(ep_controls_info.Participant,x)),extractBetween(TLE_Control_GM_names,'_GM_','.nii'));

lin_map = zeros(113,137,113);
exp_map = zeros(113,137,113);
pwr_map = zeros(113,137,113);
for r = 1:size(ADNI_Control_GM,1)
    for c = 1:size(ADNI_Control_GM,2)
        for d = 1:size(ADNI_Control_GM,3)
            ADNI_vox = permute(ADNI_Control_GM(r,c,d,:),[4 3 2 1]);
            TLE_vox = permute(TLE_Control_GM(r,c,d,:),[4 3 2 1]);

            if median([TLE_vox ;ADNI_vox]) == 0
                continue
            end
            data = double([[ADNI_age;TLE_age] [ADNI_vox;TLE_vox]]);
            data = data(all(~isnan(data),2),:);

            [lin,lin_gof,~] = fit(data(:,1),data(:,2),'poly1');
            lin_map(r,c,d) = lin_gof.rsquare;

            [exp,exp_gof,~] = fit(data(:,1),data(:,2),'exp1');
            exp_map(r,c,d) = exp_gof.rsquare;

            [pow,pwr_gof,~] = fit(data(:,1),data(:,2),'power1');
            pwr_map(r,c,d) = pwr_gof.rsquare;

            if lin_gof.rsquare >0.5 && exp_gof.rsquare > 0.5 && pwr_gof.rsquare > 0.5
                figure;
                nexttile
                plot(lin,data(:,1),data(:,2))
                title('Lin')
                subtitle(num2str(lin_gof.rsquare))
                nexttile
                plot(exp,data(:,1),data(:,2))
                title('Exponential')
                subtitle(num2str(exp_gof.rsquare))
                nexttile
                plot(pow,data(:,1),data(:,2))
                title('Power')
                subtitle(num2str(pwr_gof.rsquare))
                error('')
            end
        end
    end
end

template = load_nii(controlfiles{1});

lin_nii = template;
exp_nii = template;
pwr_nii = template;

lin_nii.img = lin_map;
exp_nii.img = exp_map;
pwr_nii.img = pwr_map;

lin_ave = mean(lin_map(lin_map>0),'all');
exp_ave = mean(exp_map(exp_map>0),'all');
pwr_ave = mean(pwr_map(pwr_map>0),'all');

figure;
subplot(1,2,1)
histogram(lin_map(lin_map>0))
subplot(1,2,2)
histogram(exp_map(exp_map>0))

sum(lin_map>.2,'all')
sum(exp_map>.2,'all')
sum(pwr_map>.2,'all')

save_nii(lin_nii,'/home/bonilha/Downloads/lin_agereg.nii');
save_nii(exp_nii,'/home/bonilha/Downloads/exp_agereg.nii');
save_nii(pwr_nii,'/home/bonilha/Downloads/pwr_agereg.nii');
%% Functions

function [X, X_names,N] = get_volume_data(ff)
count = 1;
for i = 1:numel(ff)
    N = load_nii(ff{i});
    X(:,:,:,count) = N.img;
    X_names{count,1} =ff{i};
    count = count + 1;
end
end