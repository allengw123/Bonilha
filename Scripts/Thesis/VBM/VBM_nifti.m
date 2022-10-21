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
%
% Positive t stat --> first input --> control higher
% Negative t stat --> second input -- patients higher
%--------------------------------------------------------------------------
% Inputs:
%   You have select the input files by looking at the functions at the
%   bottom of the script.
%--------------------------------------------------------------------------
%%
clear
clc

githubpath='/home/bonilha/Documents/GitHub/Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='/media/bonilha/Elements/Image_database';
save_path = '/media/bonilha/AllenProj/Thesis/VBM';

jhu_path = fullfile(githubpath,'Toolbox','imaging','Atlas','jhu','Resliced Atlas','rJHU','r75_Hippo_L.nii');
mask = jhu_path;
jhu = load_nii(jhu_path);
hipp_log = jhu.img~=0;
thres = 0.2;
%% Load nifti

patients = dir(fullfile(save_path,'niftifiles','Patients','*pre*gm*'));
patients_files = [];
for p = 1:length(patients)
    patients_files = [patients_files;{fullfile(patients(p).folder,patients(p).name)}];
end


controls = dir(fullfile(save_path,'niftifiles','Controls','*session*gm*'));
controls_files = [];
for p = 1:length(controls)
    controls_files = [controls_files;{fullfile(controls(p).folder,controls(p).name)}];
end

[TLE_GM, TLE_GM_names] = get_volume_data(patients_files);
[Control_GM, Control_GM_names] = get_volume_data(controls_files);
 
%% compare volumes (t-test/bonferroni)
ttest_savepath=fullfile(save_path,'ttest');
mkdir(ttest_savepath);

side = cellfun(@(x) {x(1)},cellfun(@(x) extractAfter(x,'P'), TLE_GM_names, 'UniformOutput', false));

right_log = cellfun(@(x) strcmp(x,'R'),side);
left_log = cellfun(@(x) strcmp(x,'L'),side);

[P, T] = compare_volumes(Control_GM, TLE_GM(:,:,:,right_log),...
    controls_files{1},ttest_savepath,'Control','R_TLE');

[P, T] = compare_volumes(Control_GM, TLE_GM(:,:,:,left_log),...
    controls_files{1},ttest_savepath,'Control','L_TLE');

cd(ttest_savepath)

%% supporting functions

function [X, X_names,N] = get_volume_data(ff)
    count = 1;
    for i = 1:numel(ff)
        N = load_nii(ff{i});
        X(:,:,:,count) = N.img;
        [~,sbjname,~] = fileparts(ff{i});
        X_names{count,1} =sbjname; 
        count = count + 1;
    end
end
 
function [P, T] = compare_volumes(Cont, Pat, mask, save_place, v1_savename, v2_savename)

    savefile_name = [v1_savename, ' vs ', v2_savename];
    [~,P,~,STATS] = ttest2(Cont,Pat,'dim',4);
 
    T = STATS.tstat;
    M = load_nii(mask);
    M.img = zeros(113,137,113);
    
    PM = M;
    TM = M;

    if size(P,3) == 58
        PM.img(:,:,28:85) = P;
        TM.img(:,:,28:85) = T;
    else
        PM.img = P;
        TM.img = T;
    end
    save_nii(PM, fullfile(save_place,[savefile_name '_P.nii']));
    save_nii(TM, fullfile(save_place,[savefile_name '_T.nii'])); 


    % save bonferroni
    crit_p = 0.05/numel(find(mean(Cont,4)>0));
    PP = P; PP(P>crit_p) = NaN;
    TT = T; TT(P>crit_p) = NaN;
        
    if size(P,3) == 58
        PM.img(:,:,28:85) = PP;
        TM.img(:,:,28:85) = TT;
    else
        PM.img = PP;
        TM.img = TT;
    end
    save_nii(TM, fullfile(save_place,[savefile_name '_T_bonf.nii'])); 
    save_nii(PM, fullfile(save_place,[savefile_name '_P_bonf.nii']));


end

