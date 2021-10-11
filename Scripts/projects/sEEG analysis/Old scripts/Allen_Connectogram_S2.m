clear all
close all
clc
Bonilha_start

%% Insert Info
datadir='C:\Users\allen\Box Sync\Desktop\Allen_Bonilha_EEG\sEEG project\PatientData';

analysisdir='C:\Users\allen\Box Sync\Desktop\Allen_Bonilha_EEG\';

Patient_ID={dir(fullfile(datadir,'Patient *')).name};

% Trials
trials_label={'pre-baseline','pre-trans','post-trans','mid-seiz','late-seiz'};

bandname = {'beta'};    

resp={'P001','P002','P003','P006','P009','P012','P013'};
nonresp={'P502','P503'};

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
%% Group-averaged coherence plots. To compare group 1 vs 2.
for m=1:numel(Patient_ID)
    for b=1:numel(bandname)
        
        % Patient matdata dir
        matdir=fullfile(datadir,Patient_ID{m},'sEEG','matdata');
        
        % Skip non-analyzed subjects
        if exist(matdir,'dir')==0
            continue
        end

        % Detect Seizure files
        S=dir(matdir);
        S={S.name};
        seizureidx=~cellfun('isempty',regexp(S,[Patient_ID{m},'_',bandname{b},'_P.*']));
        seizure_mat=S(seizureidx);
            
        clearvars coh_comb_seizure
        % Organize seizures (only pre-baseline,pre-trans,post-trans,mid-seiz,late-seiz)
        for q=1:numel(seizure_mat)
            coh_comb_temp=load(fullfile(matdir,seizure_mat{q}));
            coh_comb_temp=eval(['coh_comb_temp.',char(fieldnames(coh_comb_temp))]);
            coh_comb_seizure(:,:,:,q)=coh_comb_temp(:,:,1:5);
        end

        % Obtain average seizure data
        coh_comb_seizure_avg=mean(coh_comb_seizure,4,'omitnan');
        
        % Save seizures
        beta_coh.(['P',extractAfter(Patient_ID{m},'Patient ')])=coh_comb_seizure_avg;
    end
end

% Calculate means for responsive and nonresponsive group
for r=1:numel(resp)
    resp_coh(:,:,:,r)=beta_coh.(resp{r});
end
mean_resp_coh=mean(resp_coh,4,'omitnan');

for r=1:numel(nonresp)
    nonresp_coh(:,:,:,r)=beta_coh.(nonresp{r});
end
mean_nonresp_coh=mean(nonresp_coh,4,'omitnan');


%% Figures

% Create responsive figure
for i=1:5
    figure
    imagesc(resp_coh(:,:,i),[0 1]);
    c=colorbar;
    ylabel(c,'Beta coherence','fontsize',8);
    c.FontSize = 12;
    colormap jet;
    xlabel('Regions','fontsize',10);
    ylabel('Regions','fontsize',10);
    set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
    set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
    title(['Responders - ',trials_label{i}])
end

% Create nonresponsive figure
for i=1:5
    figure
    imagesc(nonresp_coh(:,:,i),[0 1]);
    c=colorbar;
    ylabel(c,'Beta coherence','fontsize',8);
    c.FontSize = 12;
    colormap jet;
    xlabel('Regions','fontsize',10);
    ylabel('Regions','fontsize',10);
    set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
    set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
    title(['Non Responders - ',trials_label{i}])
end


