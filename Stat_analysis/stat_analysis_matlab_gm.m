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

%% create folders for results 
Parentfolder = 'C:\Users\bonilha\Documents\Project_Eleni\SBA_output\';
%mkdir(fullfile(Parentfolder,'Results_matlab_wmp1_all_files'));
%mkdir(fullfile(Parentfolder,'Results_matlab_wmp1_only_left'));
%mkdir(fullfile(Parentfolder,'Results_matlab_wmp1_only_right'));
%mkdir(fullfile(Parentfolder,'Results_matlab_wmp2_all_files'));
%mkdir(fullfile(Parentfolder,'Results_matlab_wmp2_only_left'));
%mkdir(fullfile(Parentfolder,'Results_matlab_wmp2_only_right'));

%% quality control
% display_slices('C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_mod\mod_smooth10_controls_gm', 75)
% display_slices('C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_mod\mod_smooth10_patients_right_gm', 75)
 
%% get volume data
[Controls_GM, Controls_GM_names] = get_volume_data('C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_mod\mod_smooth10_controls_gm');
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
 
function [X, X_names] = get_volume_data(F)
    tic
    ff = dir(F);
    count = 1;
    for i = 1:numel(ff)
        if endsWith(ff(i).name, 'nii')
            subplot(9,25,count)
            N = load_nii(fullfile(F, ff(i).name));
            X(:,:,:,count) = N.img;
            X_names{count} = fullfile(F, ff(i).name);
            count = count + 1;
        end
    end
    toc
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

