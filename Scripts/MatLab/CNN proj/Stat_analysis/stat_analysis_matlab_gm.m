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

save_path='F:\CNN output';

SmoothThres=fullfile(PatientData,'thres');
addpath(genpath(SmoothThres));
cnn_output = 'F:\CNN output';

matter='GM'
%% Find Nii files

% look for Alz nifti files
Alzfiles={dir(fullfile(SmoothThres,'Alz\ADNI_Alz_nifti','*',['*',matter,'*.nii'])).name};

% look for TLE nifti files
tlefiles={dir(fullfile(SmoothThres,'TLE','*','*',['*',matter,'*.nii'])).name};

% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Control','*','*',['*',matter,'*.nii'])).name}';


%% get volume data

[Alz_GM, Alz_GM_names] = get_volume_data(Alzfiles);
%%
[Patients_Left_GM, Patients_Left_GM_names] = get_volume_data('C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_mod\mod_smooth10_patients_right_gm');
 
%% compare volumes
[P, T] = compare_volumes(Controls_GM, Patients_Left_GM,...
    'C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_mod\mod_smooth10_controls_gm/smooth10mwp1ECK001.nii','C:\Users\bonilha\Documents\Project_Eleni\Results_matlab_wmp1_only_right');
 
%% supporting functions
 
function display_slices(F, slice_number)
    figure
    ff = dir(F);
    count = 1;
    for i = 1:numel(ff)
        if endsWith(ff(i).name, 'nii')
            subplot(9,25,count)
            N = load_nii(fullfile(F, ff(i).name));
            imagesc(N.img(:,:,slice_number))
            count = count + 1;
        end
    end
end
 
function [X, X_names] = get_volume_data(ff)
    count = 1;
    for i = 1:numel(ff)
        N = load_nii(ff{i});
        X(:,:,:,count) = N.img;
        X_names{count,1} =ff{i};
        count = count + 1;
    end
end
 
function [P, T] = compare_volumes(Cont, Pat, mask, save_place)
    tic
    Cont(Cont<0.5) = 0;
    Pat(Pat<0.5) = 0;
 
    [~,P,~,STATS] = ttest2(Cont,Pat,'dim',4);
 
    T = STATS.tstat;
    M = load_nii(mask);
    M.img = P;
    save_nii(M, fullfile(save_place, 'P.nii'));
    M.img = T;
    save_nii(M, fullfile(save_place, 'T.nii')); 
    
    %% save bonferroni
    crit_p = 0.05/numel(find(mean(Cont,4)>0));
    PP = P; PP(P>crit_p) = NaN;
    TT = T; TT(P>crit_p) = NaN;
    
    M.img = PP;
    save_nii(M, fullfile(save_place, 'P_bonf.nii'));
    M.img = TT;
    save_nii(M, fullfile(save_place, 'T_bonf.nii'));     
    
end

