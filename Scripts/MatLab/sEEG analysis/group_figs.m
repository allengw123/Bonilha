clear all
close all
clc

% gitpath='C:\Users\allen\Documents\GitHub\Bonilha';
gitpath = '/home/bonilha/Documents/GitHub/Bonilha';

cd(gitpath)
allengit_genpath(gitpath,'imaging')
%% Insert Info
datadir='/media/bonilha/AllenProj/sEEG_project/PatientData';
analysisdir=fullfile(datadir,'Analysis');

% Find subject folders
subjID = [dir(fullfile(datadir,'*','Patient*'));dir(fullfile(datadir,'*','3T*'))];

% Treatment Outcome
ref_sheet = load(fullfile(datadir,'outcome.mat')).ref_sheet;
response = [];
tab = [];
for s = 1:numel(subjID)
    wk_sbj = subjID(s).name;
    wk_res = regexp(ref_sheet.ILAElatest{contains(ref_sheet.PreviousIDs,wk_sbj)},'\d*','Match');
    tab = [tab;ref_sheet(contains(ref_sheet.PreviousIDs,wk_sbj),:)];
    if isempty(wk_res)
        wk_res = -1;
    else
        wk_res = str2double(wk_res{1});
    end
    response = [response;wk_res];
end
responsive_idx = response==1;
nonresponsive_idx = response>1;

% Define electrodes
master_electrode={'LA','LAH','LAI','LLF','LMF','LPH','LPI','RA','RAH','RAI','RLF','RMF','RPH','RPI'};

master_electrode_labels=[];
depth={'_D','_M','_S'};
for i=1:numel(master_electrode)
    for z=1:3
        master_electrode_labels=[master_electrode_labels;{[master_electrode{i},depth{z}]}];
    end
end

master_electrode_labels_grouped=[];
group_label={'(D)','(M)','(S)'};
for i=1:numel(master_electrode)
    for z=1:3
        if z==1
            master_electrode_labels_grouped=[master_electrode_labels_grouped;{[master_electrode{i},'-',group_label{z},'-']}];
        else
            master_electrode_labels_grouped=[master_electrode_labels_grouped;{[group_label{z},'-']}];
        end
    end
end

% Create connectivity matrix
connectivitymat=nan(numel(master_electrode_labels),numel(master_electrode_labels));
%% Calculate group metrics

cum_coh = [];
cum_fa = [];
for m=1:numel(subjID)
            
    % Detect Baseline Files
    baseline_files = [dir(fullfile(subjID(m).folder,subjID(m).name,'sEEG','matdata_whole','*baseline*'));
        dir(fullfile(subjID(m).folder,subjID(m).name,'sEEG','matdata_whole','*rest*'))];
        
    % Load Coherence
    coh_matrix = [];
    for b = 1:numel(baseline_files)
        wk_coh = load(fullfile(baseline_files(b).folder,baseline_files(b).name));
        coh_matrix = cat(3,coh_matrix,wk_coh.connectivitymat_grouped);
    end
    coh_matrix = mean(coh_matrix,3);

    % Load FA
    fa_file = dir(fullfile(subjID(m).folder,subjID(m).name,'structural','Tractography','Connectivity','*fa.mat*'));
    fa_mat=load(fullfile(fa_file.folder,fa_file.name));
    wk_elec_labels=textscan(char(fa_mat.name),'%s');
    wk_elec_labels=wk_elec_labels{1,1}(1:end-1);
    wk_elec_labels = strrep(wk_elec_labels,'_P','_S');
    raw_fa_matrix = fa_mat.connectivity;

    % Organize fa data
    labelidx=[];
    for i=1:numel(wk_elec_labels)
        labelidx=[labelidx,find(strcmp(wk_elec_labels{i},master_electrode_labels))];
    end
    
    fa_matrix = connectivitymat;
    for row=1:size(raw_fa_matrix,1)
        for col=1:size(raw_fa_matrix,2)
            fa_matrix(labelidx(row),labelidx(col))=raw_fa_matrix(row,col);
        end
    end

    % Store data
    cum_coh = cat(3,cum_coh,coh_matrix);
    cum_fa = cat(3,cum_fa,fa_matrix);
end
%% Coherence Group Statistics

resp_coh = cum_coh(:,:,responsive_idx);
resp_coh_mean = mean(resp_coh,3,'omitnan');
resp_coh_std = std(resp_coh,[],3,'omitnan');

nonresp_coh = cum_coh(:,:,nonresponsive_idx);
nonresp_coh_mean = mean(nonresp_coh,3,'omitnan');
nonresp_coh_std = std(nonresp_coh,[],3,'omitnan');


[~,P,~,STATS] = ttest2(resp_coh,nonresp_coh,'dim',3);
T = STATS.tstat;
b_P = P;
b_T = T;
b_P(P > 0.05/numel(find(~isnan(STATS.tstat)))) = NaN;
b_T(P > 0.05/numel(find(~isnan(STATS.tstat)))) = NaN;

b_T(isinf(b_T)) = NaN;
b_T = abs(b_T);

func_ccm = customcolormap([0 0.5 1],{'#2d1111','#d36060','#e4b8b8'});
% 
% figure;
% imagesc(resp_coh_mean,[0 1])
% title('Responsive Coherence Mean')
% c=colorbar;
% ylabel(c,'Beta coherence','fontsize',12);
% axis('square')
% colormap(func_ccm)
%  
% figure;
% imagesc(nonresp_coh_mean,[0 1])
% title('Nonresponsive Coherence Mean')
% c=colorbar;
% ylabel(c,'Beta coherence','fontsize',12);
% axis('square')
% colormap(func_ccm)



figure;
tiledlayout(3,2)
nexttile
imagesc(resp_coh_mean)
title('Responsive Coherence Mean')
colorbar
colormap(func_ccm)
nexttile
imagesc(resp_coh_std)
title('Responsive Coherence STD')
colorbar
colormap(func_ccm)
nexttile
imagesc(nonresp_coh_mean)
title('Nonresponsive Coherence Mean')
colorbar
colormap(func_ccm)
nexttile
imagesc(nonresp_coh_std)
title('Nonresponsive Coherence STD')
colorbar
colormap(func_ccm)
nexttile([1 2])
imagesc(b_T)
title('Bon corrected T')
colorbar
    
%% FA Group Statistics
resp_fa = cum_fa(:,:,responsive_idx);
resp_fa_mean = mean(resp_fa,3,'omitnan');
resp_fa_std = std(resp_fa,[],3,'omitnan');

nonresp_fa = cum_fa(:,:,nonresponsive_idx);
nonresp_fa_mean = mean(nonresp_fa,3,'omitnan');
nonresp_fa_std = std(nonresp_fa,[],3,'omitnan');


[~,P,~,STATS] = ttest2(resp_fa,nonresp_fa,'dim',3);
T = STATS.tstat;
b_P = P;
b_T = T;
b_P(P > 0.05/numel(find(~isnan(STATS.tstat)))) = NaN;
b_T(P > 0.05/numel(find(~isnan(STATS.tstat)))) = NaN;

b_T(isinf(b_T)) = NaN;
b_T = abs(b_T);

struct_ccm=customcolormap([0 0.5 1],{'#09090f','#618bc7','#aac6e8'});
% 
% figure;
% imagesc(resp_fa_mean,[0 0.5])
% title('Responsive FA')
% c=colorbar;
% ylabel(c,'Fractional Anisotropy','fontsize',12);
% axis('square')
% colormap(struct_ccm)
% 
% 
% figure;
% imagesc(nonresp_fa_mean,[0 0.5])
% title('NonResponsive FA')
% c=colorbar;
% ylabel(c,'Fractional Anisotropy','fontsize',12);
% axis('square')
% colormap(struct_ccm)


figure;
tiledlayout(3,2)
nexttile
imagesc(resp_fa_mean)
title('Responsive FA Mean')
colormap(struct_ccm)
colorbar
nexttile
imagesc(resp_fa_std)
title('Responsive FA STD')
colormap(struct_ccm)
colorbar
nexttile
imagesc(nonresp_fa_mean)
title('Nonresponsive FA Mean')
colormap(struct_ccm)
colorbar
nexttile
imagesc(nonresp_fa_std)
title('Nonresponsive FA STD')
colorbar
colormap(struct_ccm)
nexttile([1 2])
imagesc(b_T)
title('Bon corrected T')
colorbar
%% Electrode Group Staistics
resp_coh = cum_coh(:,:,responsive_idx);
resp_elec = sum(~isnan(resp_coh),3);

nonresp_coh = cum_coh(:,:,nonresponsive_idx);
nonresp_elec = sum(~isnan(nonresp_coh),3);

elect_ccm=customcolormap([0 0.5 1],{'#151240','#3d439b','#b9b6db'});

figure;
imagesc(resp_elec,[0 10])
title('Responsive')
c=colorbar;
ylabel(c,'# of Individuals','fontsize',12);
axis('square')
colormap(elect_ccm)

figure
imagesc(nonresp_elec,[0 10])
title('Nonresponsive')
c=colorbar;
ylabel(c,'# of Individuals','fontsize',12);
axis('square')
colormap(elect_ccm)

