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
matter = 'gm';


jhu_path = fullfile(githubpath,'Toolbox','imaging','Atlas','jhu','Resliced Atlas','rJHU','r75_Hippo_L.nii');
mask = jhu_path;
jhu = load_nii(jhu_path);
hipp_log = jhu.img~=0;
thres = 0;
%% Load matfiles

mkdir(save_path)
cd(PatientData)

patients = dir(fullfile(PatientData,'*TLE*','post_qc','Patients','*.mat'));
controls = dir(fullfile(PatientData,'*TLE*','post_qc','Controls','*.mat'));

% Patients img
side = cell(numel(patients),1);
patient_img = cell(numel(patients),1);
patient_img_ns = cell(numel(patients),1);
patient_img_hipp = cell(numel(patients),1);
patient_img_hipp_ns = cell(numel(patients),1);
patient_name = cell(numel(patients),1);
parfor pat=1:numel(patients)
%     if contains(patients(pat).name,'Lesion')
%             continue
%     end
    matname = fullfile(patients(pat).folder,patients(pat).name);
    vars = whos('-file',matname);
    if ismember('pre', {vars.name}) %&& ismember('pos', {vars.name})
        temp=load(fullfile(patients(pat).folder,patients(pat).name));

        patient_name{pat} = patients(pat).name;

        % Non smoothed
        temp_img = temp.pre.vbm_gm.dat;
        temp_img(temp_img<thres) = 0;
        patient_img_ns{pat} = temp_img;
        patient_img_hipp_ns{pat} = temp_img(hipp_log);

        % Smoothed
        temp_img = temp.pre.smooth_vbm_gm.dat;
        temp_img(temp_img<thres) = 0;
        patient_img{pat} = temp_img;
        patient_img_hipp{pat} = temp_img(hipp_log);

        wk_side = extractAfter(patients(pat).name,'P');
        side{pat} = wk_side(1);
    end
end
patient_name = patient_name(~cellfun(@isempty,patient_name));
patient_img = cat(4,patient_img{:});
patient_img_ns = cat(4,patient_img_ns{:});
patient_img_hipp = cat(4,patient_img_hipp{:});
patient_img_hipp_ns = cat(4,patient_img_hipp_ns{:});

side = side(~cellfun(@isempty,side));
left_log = cellfun(@(x) strcmp(x,'L'),side);
right_log = cellfun(@(x) strcmp(x,'R'),side);

% Control img
control_img = cell(numel(controls),1);
control_img_ns = cell(numel(controls),1);
control_img_hipp = cell(numel(controls),1);
parfor con=1:numel(controls)
    temp=load(fullfile(controls(con).folder,controls(con).name));
    if isfield(temp,'session')

        % Non smoothed
        temp_img = temp.session.vbm_gm.dat;
        temp_img(temp_img<thres) = 0
        control_img_ns{con} = temp_img
        control_img_hipp_ns{con} = temp_img(hipp_log);

        % Smoothed
        temp_img = temp.session.smooth_vbm_gm.dat;
        temp_img(temp_img<thres) = 0;
        control_img{con} = temp_img
        control_img_hipp{con} = temp_img(hipp_log);
    end
end
control_img = cat(4,control_img{:});
control_img_ns = cat(4,control_img_ns{:});
control_img_hipp = cat(4,control_img_hipp{:});
control_img_hipp_ns = cat(4,control_img_hipp_ns{:});
%% display volume (raw)
comp_intensity(control_img_ns,patient_img_ns,'NonSmoothed-Cumultive intensity','controls','patients')
comp_intensity(control_img_hipp_ns,patient_img_hipp_ns(:,:,:,left_log),'NonSmoothed left hippocampal intensity','controls','left-patients')
comp_intensity(control_img_hipp_ns,patient_img_hipp_ns(:,:,:,right_log),'NonSmoothed left hippocampal intensity','controls','right-patients')
comp_intensity(patient_img_hipp_ns(:,:,:,left_log),patient_img_hipp_ns(:,:,:,right_log),'NonSmoothed-left hippocampal intensity','left-patients','right-patients')

%% display volume (smoothed)
comp_intensity(control_img,patient_img,'Smoothed-Cumultive intensity','controls','patients')
comp_intensity(control_img_hipp,patient_img_hipp(:,:,:,left_log),'Smoothed left hippocampal intensity','controls','left-patients')
comp_intensity(control_img_hipp,patient_img_hipp(:,:,:,right_log),'Smoothed left hippocampal intensity','controls','right-patients')
comp_intensity(patient_img_hipp(:,:,:,left_log),patient_img_hipp(:,:,:,right_log),'Smoothed-left hippocampal intensity','left-patients','right-patients')

%% compare volumes (t-test/bonferroni)
compare_volumes(control_img_ns, patient_img_ns(:,:,:,left_log), mask, save_path, 'control', 'left_tle');
compare_volumes(control_img_ns, patient_img_ns(:,:,:,right_log),mask, save_path, 'control', 'right_tle');

cd(save_path)
%% compare volumes (t-test/bonferroni)
compare_volumes(control_img, patient_img(:,:,:,left_log), mask, save_path, 'control', 'left_tle');
compare_volumes(control_img, patient_img(:,:,:,right_log),mask, save_path, 'control', 'right_tle');
cd(save_path)
%%
x =patient_img_ns(:,:,:,left_log);
for i = 1:size(x,4)
    temp = x(:,:,:,i);
    temp(hipp_log) = 0;
    x(:,:,:,i) = temp;
end
compare_volumes(control_img_ns, x, mask, save_path, 'test_control', 'test_left_tle');

%%
function [P, T] = compare_volumes(Cont, Pat, mask, save_place, v1_savename, v2_savename,hipp_log)

M = load_nii(mask);
M.img = zeros(size(M.img));
M.hdr.dime.datatype = 16;
M.hdr.dime.bitpix = 16;
%control = M;
%control.img = mean(Cont,4);
%     save_nii(control,fullfile(save_place,'control_mean.nii'))
%
%     patient = M;
%     patient.img = mean(Pat,4);
%     save_nii(patient,fullfile(save_place,'patient_mean.nii'))

savefile_name = [v1_savename, ' vs ', v2_savename];
[~,P,~,STATS] = ttest2(Cont,Pat,'dim',4);

T = STATS.tstat;

if exist('hipp_log','var')
    cont_sum = [];
    for i = 1:size(Cont,4)
        temp = Cont(:,:,:,i);
        cont_sum = [cont_sum;sum(temp(hipp_log))];
    end

    pat_sum = [];
    for i = 1:size(Pat,4)
        temp = Pat(:,:,:,i);
        pat_sum = [pat_sum;sum(temp(hipp_log))];
    end

    figure
    boxplot([cont_sum;pat_sum],[zeros(length(cont_sum),1); ones(length(pat_sum),1)])
    [~,P,~,stat] = ttest2(cont_sum,pat_sum);
    subtitle(['P = ',num2str(P),' T = ',num2str(stat.tstat)])
end


PM = M;
TM = M;

PM.img = P;
TM.img = T;
save_nii(PM, fullfile(save_place,[savefile_name '_P.nii']));
save_nii(TM, fullfile(save_place,[savefile_name '_T.nii']));


% save bonferroni
crit_p = 0.05/sum(P>0,"all");
PP = P; PP(P>crit_p) = nan;
TT = T; TT(P>crit_p) = nan;

PM.img = PP;
TM.img = TT;

save_nii(TM, fullfile(save_place,[savefile_name '_T_bonf.nii']));
save_nii(PM, fullfile(save_place,[savefile_name '_P_bonf.nii']));

end

function comp_intensity(control_img,patient_img,figure_title,control_title,patient_title)

figure;
control_sum = permute(sum(control_img,[1 2 3]),[4 1 2 3]);
patient_sum= permute(sum(patient_img,[1 2 3]),[4 1 2 3]);
boxplot([control_sum;patient_sum],[zeros(length(control_sum),1); ones(length(patient_sum),1)])
title(figure_title)
[~,P,~,stat] = ttest2(control_sum,patient_sum);
subtitle(['P = ',num2str(P),' T = ',num2str(stat.tstat)])
xticklabels({control_title,patient_title})
end