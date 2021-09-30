 %% This script is used to run an Independent (two-sample) t-test using the 
% modified smoothed files from the VBM analysis of CAT12 (grey matter).
%--------------------------------------------------------------------------
% Notes: 
% It uses the smoothed files with a threshold of 0.5.  
% Analysis are run for: 
%       controls vs left patients 
%       controls vs right patients 
%       controls vs all patients 
%
%--------------------------------------------------------------------------
% Inputs: 
%   You have select the input files by looking at the functions at the
%   bottom of the script. 
%--------------------------------------------------------------------------
%%
clear
clc

githubpath='C:\Users\bonilha\Documents\GitHub\Bonilha';
% githubpath='C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='F:\PatientData';
cd(PatientData)


SmoothThres=fullfile(PatientData,'thres');
addpath(genpath(SmoothThres));
cnn_output = 'F:\CNN output';

matter='GM';

save_path='F:\VBM ouput';
mkdir(save_path)

%% Find Nii files

% look for Alz nifti files
Alzfiles={dir(fullfile(SmoothThres,'Alz\ADNI_Alz_nifti','*',['*',matter,'*.nii'])).name};

% look for TLE nifti files
tlefiles={dir(fullfile(SmoothThres,'TLE','*','*',['*',matter,'*.nii'])).name};

% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Control','*','*',['*',matter,'*.nii'])).name}';

% Read excel files
ep_tle_info=readtable(fullfile(PatientData,'ep_TLE_info.xlsx'));
ep_controls_info=readtable(fullfile(PatientData,'ep_controls_info.xlsx'));
ADNI_CN_info=readtable(fullfile(PatientData,'ADNI_CN_info.csv'));
ADNI_Alz_info=readtable(fullfile(PatientData,'ADNI_Alz_info.csv'));

%% get volume data

[Alz_GM, Alz_GM_names] = get_volume_data(Alzfiles);
[TLE_GM, TLE_GM_names] = get_volume_data(tlefiles);
[Control_GM, Control_GM_names] = get_volume_data(controlfiles);
 
%% compare volumes
[P, T] = compare_volumes(Alz_GM, TLE_GM,...
    controlfiles{1},save_path,'Alz vs TLE');

[P, T] = compare_volumes(Alz_GM, Control_GM,...
    controlfiles{1},save_path,'Alz vs Control');

[P, T] = compare_volumes(TLE_GM, Control_GM,...
    controlfiles{1},save_path,'TLE vs Control');

cd(save_path)
%% Fitlm

count=1;
for i=1:size(Alz_GM,4)
    tempname=extractBetween(Alz_GM_names(i),'_I','.nii');
    age=ADNI_Alz_info.Age(strcmp(extractAfter(ADNI_Alz_info.ImageDataID,'I'),tempname));
    if ~isnan(age)
        C = Alz_GM(:,:,:,i);
        Alz_GM_vec(count,:)=C(:)';
        Alz_GM_age(count,1)=age;
        count=count+1;
    else
        disp([tempname{:},' age is NAN'])
        continue
    end
end

count=1;
for i=1:size(TLE_GM,4)
    tempname=extractBetween(TLE_GM_names(i),'_GM_','.nii');
    age=ep_tle_info.Age(strcmp(ep_tle_info.ID,tempname));
    if ~isnan(age)
        C = TLE_GM(:,:,:,i);
        TLE_GM_vec(count,:)=C(:)';
        TLE_GM_age(count,1)=age;
        count=count+1;
    else
        disp([tempname{:},' age is NAN'])
        continue
    end
end

count=1;
for i=1:size(Control_GM,4)
    tempname=extractBetween(Control_GM_names(i),'GM_','.nii');
    if startsWith(tempname,'ADNI')
        tempname=extractAfter(tempname,'_I');
        age=ADNI_CN_info.Age(strcmp(extractAfter(ADNI_CN_info.ImageDataID,'I'),tempname));
    else
        age=ep_controls_info.Age(strcmp(ep_controls_info.Participant,tempname));
    end
    if ~isnan(age)
        C = Control_GM(:,:,:,i);
        Control_GM_vec(count,:)=C(:)';
        Control_GM_age(count,1)=age;
        count=count+1;
    else
        disp([tempname{:},' age is NAN'])
        continue
    end
end

per=nchoosek({'Alz','TLE','Control'},2);

for p=1:size(per,1)
    comp=[per{p,1} '(0) vs ' per{p,2},'(1)'];
    stacked=eval(['[',per{p,1},'_GM_vec ; ',per{p,2},'_GM_vec]']);
    y=[zeros(size(eval([per{p,1},'_GM_vec']),1),1) ;ones(size(eval([per{p,2},'_GM_vec']),1),1)];
    age_stacked=[eval([per{p,1},'_GM_age']);eval([per{p,2},'_GM_age'])];
    for vox=1:size(stacked,2)
        mdl{p,vox}=fitlm([age_stacked stacked(:,vox)], y);
    end
end

%% supporting functions

function [X, X_names] = get_volume_data(ff)
    count = 1;
    for i = 1:numel(ff)
        N = load_nii(ff{i});
        X(:,:,:,count) = N.img;
        X_names{count,1} =ff{i};
        count = count + 1;
    end
end
 
function [P, T] = compare_volumes(Cont, Pat, mask, save_place, savefile_name)

    [~,P,~,STATS] = ttest2(Cont,Pat,'dim',4);
 
    T = STATS.tstat;
    M = load_nii(mask);
    M.img = P;
    save_nii(M, fullfile(save_place,[savefile_name '_P.nii']));
    M.img = T;
    save_nii(M, fullfile(save_place,[savefile_name '_T.nii'])); 
    
    %% save bonferroni
    crit_p = 0.05/numel(find(mean(Cont,4)>0));
    PP = P; PP(P>crit_p) = NaN;
    TT = T; TT(P>crit_p) = NaN;
    
    TT=TT.*-1;
    
    
    M.img = PP;
    save_nii(M, fullfile(save_place,[savefile_name '_P_bonf.nii']));

    M.img = TT;
    save_nii(M, fullfile(save_place,[savefile_name '_T_bonf.nii']));     

end

