close all
clear all
clc

GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Directory
old_dir = '/media/bonilha/AllenProj/sEEG_project/PatientData/Original';
old_sbj = dir(fullfile(old_dir,'Patient*'));

new_dir = '/media/bonilha/AllenProj/sEEG_project/PatientData/CAPES_LEN';
new_sbj = dir(fullfile(new_dir,'3T*'));

% Labels
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

template_mat=nan(numel(master_electrode_labels),numel(master_electrode_labels));
trialnames={'Baseline','Early Sezuire','Mid Sezuire','Late Sezuire'};

%% Electrode comparison

old_elec_labels = [];
for s = 1:numel(old_sbj)
    wk_folder = fullfile(old_sbj(s).folder,old_sbj(s).name);
    wk_elec_folder = fullfile(wk_folder,'structural','Tractography','Electrodes');
    wk_elec = dir(fullfile(wk_elec_folder,'*.nii*'));

    if numel(wk_elec) == 1
        wk_elec_labels = readtable(fullfile(wk_elec_folder,'Electrodes.txt'));
        old_elec_labels = [old_elec_labels;{wk_elec_labels{:,2}}];
    else
        wk_elec_labels = extractBefore({wk_elec.name},'.nii');
        old_elec_labels = [old_elec_labels;{wk_elec_labels'}];
    end
end

new_elec_labels = [];
for s = 1:numel(new_sbj)
    wk_folder = fullfile(new_sbj(s).folder,new_sbj(s).name);
    wk_elec_folder = fullfile(wk_folder,'structural','Tractography','Electrodes');
    wk_elec = dir(fullfile(wk_elec_folder,'*.nii*'));

    wk_elec_labels = extractBefore({wk_elec.name},'.nii');
    new_elec_labels = [new_elec_labels;{wk_elec_labels'}];
end

figure;
nexttile
histogram(categorical(cat(1,old_elec_labels{:})))
title('Original Electrode Labels')

nexttile
histogram(categorical(cat(1,new_elec_labels{:})))
title('New Electrode Labels')

%% FA comparison

old_fa = [];
for s = 1:numel(old_sbj)
    wk_folder = fullfile(old_sbj(s).folder,old_sbj(s).name);
    wk_con_folder = fullfile(wk_folder,'structural','Tractography','Connectivity');
    wk_fa_mat = load(fullfile(wk_con_folder,'fa.mat'));
    
    wk_labels = textscan(char(wk_fa_mat.name),'%s');
    wk_labels = wk_labels{1,1}(1:end-1);
    wk_fa = wk_fa_mat.connectivity;

    % Find label idx
    labelidx=[];
    for i=1:numel(wk_labels)
        labelidx=[labelidx,find(strcmp(wk_labels{i},master_electrode_labels))];
    end

    % Organize data
    wk_fa_organized=template_mat;
    
    for row=1:size(wk_fa,1)
        for col=1:size(wk_fa,2)
            wk_fa_organized(labelidx(row),labelidx(col))=wk_fa(row,col);
        end
    end
    old_fa = cat(3,old_fa,wk_fa_organized);
end


new_fa = [];
for s = 1:numel(new_sbj)
    wk_folder= fullfile(new_sbj(s).folder,new_sbj(s).name);
    wk_con_folder= fullfile(wk_folder,'structural','Tractography','Connectivity');
    wk_fa_mat = load(fullfile(wk_con_folder,'fa.mat'));
    
    wk_labels = textscan(char(wk_fa_mat.name),'%s');
    wk_labels = wk_labels{1,1}(1:end-1);
    wk_labels = strrep(wk_labels,'_P','_S');
    wk_fa = wk_fa_mat.connectivity;

    % Find label idx
    labelidx=[];
    for i=1:numel(wk_labels)
        labelidx=[labelidx,find(strcmp(wk_labels{i},master_electrode_labels))];
    end

    % Organize data
    wk_fa_organized=template_mat;
    
    for row=1:size(wk_fa,1)
        for col=1:size(wk_fa,2)
            wk_fa_organized(labelidx(row),labelidx(col))=wk_fa(row,col);
        end
    end
    new_fa = cat(3,new_fa,wk_fa_organized);
end

[~,P,~,STATS] = ttest2(old_fa,new_fa,'dim',3);
T = STATS.tstat;
crit_p = 0.05/sum(P>0,'all');
PP = P; PP(P>crit_p) = NaN;
TT = T; TT(P>crit_p) = NaN;


figure;
tiledlayout(1,3);

nexttile
imagesc(mean(old_fa,3,'omitnan'),[0 1])
colorbar
title('Original FA')
xticks(1:numel(master_electrode_labels_grouped))
yticks(1:numel(master_electrode_labels_grouped))
xticklabels(master_electrode_labels_grouped)
yticklabels(master_electrode_labels_grouped)

nexttile
imagesc(mean(new_fa,3,'omitnan'),[0 1])
colorbar
title('New FA')
xticks(1:numel(master_electrode_labels_grouped))
yticks(1:numel(master_electrode_labels_grouped))
xticklabels(master_electrode_labels_grouped)
yticklabels(master_electrode_labels_grouped)

nexttile
imagesc(TT,[0 1])
colorbar
title('Bonf Corrected T-Stat')

%% Connectivity Comparison

old_coh = [];
for s = 1:numel(old_sbj)
    wk_folder = fullfile(old_sbj(s).folder,old_sbj(s).name);
    wk_con_folder = fullfile(wk_folder,'sEEG','matdata');
    wk_coh_mat_files = dir(fullfile(wk_con_folder,'*beta*'));
    wk_coh_mat_files(contains({wk_coh_mat_files.name},'baseline'),:) = [];
    
    for c = 1:numel(wk_coh_mat_files)
        wk_coh_mat = load(fullfile(wk_coh_mat_files(c).folder,wk_coh_mat_files(c).name));
        wk_coh_mat = wk_coh_mat.connectivitymat_grouped;

        wk_coh_mat = wk_coh_mat(:,:,[1 3 4 5]);
        old_coh = cat(4,old_coh,wk_coh_mat);
    end
end

new_coh = [];
for s = 1:numel(new_sbj)
    wk_folder = fullfile(new_sbj(s).folder,new_sbj(s).name);
    wk_con_folder = fullfile(wk_folder,'sEEG','matdata');
    wk_coh_mat_files = dir(fullfile(wk_con_folder,'*beta*'));
    wk_coh_mat_files(contains({wk_coh_mat_files.name},'baseline'),:) = [];
    
    for c = 1:numel(wk_coh_mat_files)
        wk_coh_mat = load(fullfile(wk_coh_mat_files(c).folder,wk_coh_mat_files(c).name));
        wk_coh_mat = wk_coh_mat.connectivitymat_grouped;

        wk_coh_mat = wk_coh_mat(:,:,[1 3 4 5]);
        new_coh = cat(4,new_coh,wk_coh_mat);
    end
end

figure;
tiledlayout(4,3);
for t = 1:4

    wk_old = permute(old_coh(:,:,t,:),[1 2 4 3]);
    wk_new = permute(new_coh(:,:,t,:),[1 2 4 3]);
    [~,P,~,STATS] = ttest2(wk_old,wk_new,'dim',3);
    T = STATS.tstat;
    crit_p = 0.05/sum(P>0,'all');
    PP = P; PP(P>crit_p) = NaN;
    TT = T; TT(P>crit_p) = NaN;
    
    nexttile
    imagesc(mean(wk_old,3,'omitnan'),[0 1])
    colorbar
    title([trialnames{t},' Original'])
    xticks(1:numel(master_electrode_labels_grouped))
    yticks(1:numel(master_electrode_labels_grouped))
    xticklabels(master_electrode_labels_grouped)
    yticklabels(master_electrode_labels_grouped)

    nexttile
    imagesc(mean(new_fa,3,'omitnan'),[0 1])
    colorbar
    title([trialnames{t},' New'])
    xticks(1:numel(master_electrode_labels_grouped))
    yticks(1:numel(master_electrode_labels_grouped))
    xticklabels(master_electrode_labels_grouped)
    yticklabels(master_electrode_labels_grouped)

    nexttile
    imagesc(TT,[0 1])
    colorbar
    title('Bonf Corrected T-Stat')
    xticks(1:numel(master_electrode_labels_grouped))
    yticks(1:numel(master_electrode_labels_grouped))
    xticklabels(master_electrode_labels_grouped)
    yticklabels(master_electrode_labels_grouped)

end