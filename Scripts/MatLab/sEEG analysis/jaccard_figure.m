clear all
close all
clc

gitpath = '/home/bonilha/Documents/GitHub/Bonilha';

cd(gitpath)

allengit_genpath(gitpath)

mat_dir = fullfile('/media/bonilha/AllenProj/sEEG_project/matfiles');
matfiles = dir(fullfile(mat_dir,'*.mat'));

for m = 1:numel(matfiles)
    load(fullfile(matfiles(m).folder,matfiles(m).name))
end

TOI = [1 3 4 5];

resp_dat = [resp_mean_base resp_mean_sez(:,TOI)];
resp_sem = std(resp_dat,[],1)/sqrt(size(resp_dat,1));

nonresp_dat = [nonresp_mean_base nonresp_mean_sez(:,TOI)];
nonresp_sem = std(nonresp_dat,[],1)/sqrt(size(nonresp_dat,1));

%% Create figure
figure

eb = errorbar([mean(resp_dat,1);mean(nonresp_dat,1)]',[resp_sem;nonresp_sem]');
eb(1).Color = 'b';
eb(2).Color = 'g';
xticks(1:5)
xlim([0 6])
xticklabels({'Baseline','Pre','Early','Mid','Late'})
hold on
x_val = repmat(2:5,[size(resp_ind_sez,1) 1]);
resp_sc = swarmchart([ones(numel(resp_ind_base),1); x_val(:)],[resp_ind_base; resp_ind_sez(:)],'filled','ob');
x_val = repmat(2:5,[size(nonresp_ind_sez,1) 1]);
nonresp_sc = swarmchart([ones(numel(nonresp_ind_base),1); x_val(:)],[nonresp_ind_base; nonresp_ind_sez(:)],'filled','og');
ylim([0 1])
legend([resp_sc nonresp_sc], {'Responsive','NonResponsive'})
