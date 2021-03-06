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
PatientData='F:\PatientData\smallSet';
cd(PatientData)


SmoothThres=fullfile(PatientData,'thres');
addpath(genpath(SmoothThres));

matter='GM';

save_path='F:\VBM ouput';
mkdir(save_path)

%% Find Nii files

% look for Alz nifti files
Alzfiles={dir(fullfile(SmoothThres,'Alz\ADNI_Alz_nifti','*',['*',matter,'*.nii'])).name};

% look for TLE nifti files
tledir = dir(fullfile(SmoothThres,'TLE','*','*',['*',matter,'*.nii']));
tlefiles={tledir.name};
tleFolder = {tledir.folder};
flipIdx = cellfun(@(x) contains(x,'LTLE'),tleFolder);
% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Control','*','*',['*',matter,'*.nii'])).name}';

% Read excel files
ep_tle_info=readtable(fullfile(PatientData,'ep_TLE_info.xlsx'));
ep_controls_info=readtable(fullfile(PatientData,'ep_controls_info.xlsx'));
ADNI_CN_info=readtable(fullfile(PatientData,'ADNI_CN_info.csv'));
ADNI_Alz_info=readtable(fullfile(PatientData,'ADNI_Alz_info.csv'));

%% get volume data

[Alz_GM, Alz_GM_names,template_nii] = get_volume_data(Alzfiles);
[TLE_GM, TLE_GM_names] = get_volume_data(tlefiles);
[Control_GM, Control_GM_names] = get_volume_data(controlfiles);
 
%% compare volumes (t-test/bonferroni)
ttest_savepath=fullfile(save_path,'ttest');
mkdir(ttest_savepath);

[P, T] = compare_volumes(Alz_GM, TLE_GM,...
    controlfiles{1},ttest_savepath,'Alz vs TLE');

[P, T] = compare_volumes(Alz_GM, Control_GM,...
    controlfiles{1},ttest_savepath,'Alz vs Control');

[P, T] = compare_volumes(TLE_GM, Control_GM,...
    controlfiles{1},ttest_savepath,'TLE vs Control');

cd(ttest_savepath)

%% Load AgeRegress
save_path = 'F:\VBM ouput\Age_Regress';
ttest_savepath=fullfile(save_path,'flip_ttest');
mkdir(ttest_savepath);
cd(ttest_savepath)

imgs = load('F:\CNN output\2D_CNN\MATLAB\AgeRegress\residual_imgs.mat');
imgs = imgs.reshaped_residuals;

d_label = load('F:\CNN output\2D_CNN\MATLAB\AgeRegress\disease_label.mat');
d_label = d_label.disease;

s_label = load('F:\CNN output\2D_CNN\MATLAB\AgeRegress\side_label.mat');
s_label = s_label.side;

comp_label = [];
for i = 1:numel(d_label)
    switch d_label(i)
        case 1
            comp_label = [comp_label; 1];
        case 2
            if s_label(i) == 2
                comp_label = [comp_label; 3];
            else
                comp_label = [comp_label; 2];
            end
        case 3
            comp_label = [comp_label; 4];
    end
end
comps = {'Control','R_TLE','L_TLE','AD'};
c_label = [1,2,3,4];

p_comps = nchoosek(c_label,2);

for n = 1:length(p_comps)
    v1 = cellfun(@(x) reshape(x,113,137,[]),imgs(comp_label == p_comps(n,1)),'UniformOutput',false);
    v1 = cat(4,v1{:});
    disp([comps{p_comps(n,1)},'...',num2str(size(v1,4))])

    v2 = cellfun(@(x) reshape(x,113,137,[]),imgs(comp_label == p_comps(n,2)),'UniformOutput',false);
    v2 = cat(4,v2{:});
    disp([comps{p_comps(n,2)},'...',num2str(size(v2,4))])

   
    
    [P, T] = compare_volumes(v1, v2, 'F:\VBM ouput\Age_Regress\Example.nii', ttest_savepath,comps{p_comps(n,1)},comps{p_comps(n,2)});
end
%% Fitlm

fitlm_savepath=fullfile(save_path,'fitlm');
mkdir(fitlm_savepath)

% Organize and vectorize Alz data/age
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

% Organize and vectorize TLE data/age
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

% Organize and vectorize Control data/age
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

% Create local pool
n = parcluster('local');
nworkers = n.NumWorkers;


% Run linear model
per=nchoosek({'Alz','TLE','Control'},2);
for p=1:size(per,1)
    
    comp{p}=[per{p,1} '(0) vs ' per{p,2},'(1)'];
    disp(['Running comparison ',comp{p}])
    
    % Organize image/age data
    stacked=eval(['[',per{p,1},'_GM_vec ; ',per{p,2},'_GM_vec]']);
    age_stacked=[eval([per{p,1},'_GM_age']);eval([per{p,2},'_GM_age'])];
    additional=ones(size(stacked,1),(size(stacked,1)*nworkers)-(rem(numel(stacked),(size(stacked,1)*nworkers)))/size(stacked,1));
    stacked_reshape=reshape([stacked additional],size(stacked,1),[],12);
    
    % Label data
    y=[zeros(size(eval([per{p,1},'_GM_vec']),1),1) ;ones(size(eval([per{p,2},'_GM_vec']),1),1)];
    
    % Run parallel pool fitlm calculation
    tempx1=[];
    tempx2=[];
    tic
    spmd
        warning('off','all')
        for i=1:size(stacked_reshape,2)
            x=[age_stacked stacked_reshape(:,i,labindex)];
            mdl=fitlm(x,y);
            
            % save pval
            tempx1.pval(i,1)=mdl.Coefficients.pValue(2);
            tempx2.pval(i,1)=mdl.Coefficients.pValue(3);

            % save tStat
            tempx1.tStat(i,1)=mdl.Coefficients.tStat(2);
            tempx2.tStat(i,1)=mdl.Coefficients.tStat(3);
        end
    end
    toc
    
    % Rearrange data from parallel pool
    tempx1_pval=[];
    tempx1_tStat=[];
    tempx2_pval=[];
    tempx2_tStat=[];
    for i=1:nworkers
        % x1
        temp=tempx1{i};
        tempx1_pval=[tempx1_pval temp.pval'];
        tempx1_tStat=[tempx1_tStat temp.tStat'];
        
        %x2
        temp=tempx2{i};
        tempx2_pval=[tempx2_pval temp.pval'];
        tempx2_tStat=[tempx2_tStat temp.tStat'];
    end
    
    % Reshape vectorized data back into 3D-image
    x1{p}.pval=reshape(tempx1_pval(1:end-size(additional,2)),size(C));
    x1{p}.tStat=reshape(tempx1_tStat(1:end-size(additional,2)),size(C));
    
    x2{p}.pval=reshape(tempx2_pval(1:end-size(additional,2)),size(C));
    x2{p}.tStat=reshape(tempx2_tStat(1:end-size(additional,2)),size(C));
    
    % Save x1 (age) as nifti
    x1_nii_pval=template_nii;
    x1_nii_pval.img=x1{p}.pval;
    save_nii(x1_nii_pval,fullfile(fitlm_savepath,[comp{p},' --AGE-- pval.nii']));
    
    x1_nii_tStat=template_nii;
    x1_nii_tStat.img=x1{p}.tStat;
    save_nii(x1_nii_tStat,fullfile(fitlm_savepath,[comp{p},' --AGE-- tStat.nii']));
    
    % Save x2 (voxel intensity) as nifti
    x2_nii_pval=template_nii;
    x2_nii_pval.img=x2{p}.tStat;
    save_nii(x2_nii_pval,fullfile(fitlm_savepath,[comp{p},' --VOX-- pval.nii']));
    
    x2_nii_tStat=template_nii;
    x2_nii_tStat.img=x2{p}.tStat;
    save_nii(x2_nii_tStat,fullfile(fitlm_savepath,[comp{p},' --VOX-- tStat.nii']));
    
    % Threshold x1 at corrected pval (bonferroni)
    outsideIdx=(x1_nii_tStat.img==mode(x1_nii_tStat.img,'all'));
    
    x1_nii_tStat.img(outsideIdx)=NaN;
    x1_nii_pval.img(outsideIdx)=NaN;
    
    mc_pval=0.05/sum(outsideIdx,'all');
    
    ssIdx=(x1_nii_pval.img<=mc_pval);
    x1_nii_pval.img(~ssIdx)=NaN;
    x1_nii_tStat.img(~ssIdx)=NaN;
    
    save_nii(x1_nii_pval,fullfile(fitlm_savepath,[comp{p},' --AGE-- boncor_pval.nii']));
    save_nii(x1_nii_tStat,fullfile(fitlm_savepath,[comp{p},' --AGE-- boncor_tStat.nii']));

    % Threshold x2 at corrected pval (bonferroni)
    outsideIdx=(x2_nii_tStat.img==mode(x2_nii_tStat.img,'all'));

    x2_nii_tStat.img(outsideIdx)=NaN;
    x2_nii_pval.img(outsideIdx)=NaN;

    mc_pval=0.05/sum(outsideIdx,'all');

    ssIdx=(x2_nii_pval.img<=mc_pval);
    x2_nii_pval.img(~ssIdx)=NaN;
    x2_nii_tStat.img(~ssIdx)=NaN;
    
    save_nii(x2_nii_pval,fullfile(fitlm_savepath,[comp{p},' --VOX-- boncor_pval.nii']));
    save_nii(x2_nii_tStat,fullfile(fitlm_savepath,[comp{p},' --VOX-- boncor_tStat.nii']));

end

save(fullfile(fitlm_savepath,'fitlm_output.mat'),'x1','x2','comp')



%% supporting functions

function [X, X_names,N] = get_volume_data(ff)
    count = 1;
    for i = 1:numel(ff)
        N = load_nii(ff{i});
        X(:,:,:,count) = N.img;
        X_names{count,1} =ff{i}; 
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
    
    TT=TT.*-1;
    
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

