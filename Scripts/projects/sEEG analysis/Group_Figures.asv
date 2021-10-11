clear all
close all
clc

gitpath='C:\Users\allen\Documents\GitHub\Bonilha';
cd(gitpath)
allengit_genpath(gitpath)
%% Insert Info
datadir='C:\Users\allen\Box Sync\Desktop\Allen_Bonilha_EEG\Projects\sEEG project\PatientData';

analysisdir='C:\Users\allen\Box Sync\Desktop\Allen_Bonilha_EEG\Projects\sEEG project\PatientData\Analysis';

Patient_ID={dir(fullfile(datadir,'Patient *')).name};

% Trials
trials_label={'pre-baseline','Baseline','Early','Mid','Late'};

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

% Create connectivity matrix
connectivitymat=nan(numel(master_electrode_labels),numel(master_electrode_labels));
%% Calculate group metrics
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
            
        % Organize seizures (only pre-baseline,pre-trans,post-trans,mid-seiz,late-seiz)
        clearvars coh_comb_seizure
        for q=1:numel(seizure_mat)
            coh_comb_temp=load(fullfile(matdir,seizure_mat{q}));
            coh_comb_temp=eval(['coh_comb_temp.',char(fieldnames(coh_comb_temp))]);
            coh_comb_seizure(:,:,:,q)=coh_comb_temp(:,:,1:5);
        end
%         
%         for i=1:size(coh_comb_seizure,3)
%             figure;
%             imagesc(coh_comb_seizure(:,:,i))
%             colormap winter;
%         end
            
        % Calculate electrodes
        electrodes.clips.(['P',extractAfter(Patient_ID{m},'Patient ')])=sum(~isnan(coh_comb_seizure),4);
        
        electrodes.subjects.(['P',extractAfter(Patient_ID{m},'Patient ')])=~isnan(coh_comb_seizure(:,:,1));
        
        % Save seizures
        beta_coh.(['P',extractAfter(Patient_ID{m},'Patient ')])=coh_comb_seizure;
        
        % Patient struct dir
        structdir=fullfile(datadir,Patient_ID{m},'structural','Tractography','Connectivity');
        
        % Load fa
        tempfa=load(fullfile(structdir,'fa.mat'));
        templabels=textscan(char(tempfa.name),'%s');
        templabels=templabels{1,1}(1:end-1);
        tempdata=tempfa.connectivity;
        
        % Find label idx
        labelidx=[];
        for i=1:numel(templabels)
            labelidx=[labelidx,find(strcmp(templabels{i},master_electrode_labels))];
        end

        % Organize data
        tempconmat=connectivitymat;
        
        for row=1:size(tempdata,1)
            for col=1:size(tempdata,2)
                tempconmat(labelidx(row),labelidx(col))=tempdata(row,col);
            end
        end
        
%         % 0-1 Normalization
%         tempconmat(~isnan(tempconmat))=mat2gray(tempconmat(~isnan(tempconmat)));
        
        % Save data
        struct_con.(['P',extractAfter(Patient_ID{m},'Patient ')])=tempconmat;
    end
end

% Calculate means for responsive and nonresponsive group
count=0;
for r=1:numel(resp)
    countstart=count+1;
    count=count+size(beta_coh.(resp{r}),4);
    resp_coh(:,:,:,countstart:count)=beta_coh.(resp{r});
end
mean_resp_coh=mean(resp_coh,4,'omitnan');

count=0;
for r=1:numel(nonresp)
    countstart=count+1;
    count=count+size(beta_coh.(nonresp{r}),4);
    nonresp_coh(:,:,:,countstart:count)=beta_coh.(nonresp{r});
end
mean_nonresp_coh=mean(nonresp_coh,4,'omitnan');

% Calculate electrodes (clips) for responsive and nonresponsive group
for r=1:numel(resp)
    resp_electrode(:,:,:,r)=electrodes.clips.(resp{r});
end
resp_electrode=sum(resp_electrode,4);

for r=1:numel(nonresp)
    nonresp_electrode(:,:,:,r)=electrodes.clips.(nonresp{r});
end
nonresp_electrode=sum(nonresp_electrode,4);

% Calculate electrodes (subjects) for responsive and nonresponsive group
for r=1:numel(resp)
    resp_electrode_subject(:,:,r)=electrodes.subjects.(resp{r});
end
resp_electrode_subject=sum(resp_electrode_subject,3);

for r=1:numel(nonresp)
    nonresp_electrode_subject(:,:,r)=electrodes.subjects.(nonresp{r});
end
nonresp_electrode_subject=sum(nonresp_electrode_subject,3);

% Calculate struct for responsive and nonresponsive group
for r=1:numel(resp)
    resp_struct(:,:,r)=struct_con.(resp{r});
end
mean_resp_struct=mean(resp_struct,3,'omitnan');

for r=1:numel(nonresp)
    nonresp_struct(:,:,r)=struct_con.(nonresp{r});
end
mean_nonresp_struct=mean(nonresp_struct,3,'omitnan');

%% Coherence Group Figures

% Create responsive figure
for i=2:5
    figure
    imagesc(mean_resp_coh(:,:,i),[0 1]);
    c=colorbar;
    ylabel(c,'Beta coherence','fontsize',8);
    c.FontSize = 12;
    colormap bone;
    xlabel('Regions','fontsize',10);
    ylabel('Regions','fontsize',10);
    set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
    set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
    title(['Responders - ',trials_label{i}])
    t=get(gca,'title');
    saveas(gcf,fullfile('C:\Users\allen\Google Drive\School\MUSC\Lab\Bonilha\sEEG paper\Figures\Coherence',t.String),'eps')
end

% Create nonresponsive figure
for i=2:5
    figure
    imagesc(mean_nonresp_coh(:,:,i),[0 1]);
    c=colorbar;
    ylabel(c,'Beta coherence','fontsize',8);
    c.FontSize = 12;
    colormap bone;
    xlabel('Regions','fontsize',10);
    ylabel('Regions','fontsize',10);
    set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
    set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
    title(['Non Responders - ',trials_label{i}])
    t=get(gca,'title');
    saveas(gcf,fullfile('C:\Users\allen\Google Drive\School\MUSC\Lab\Bonilha\sEEG paper\Figures\Coherence',t.String),'eps')
end

%% Electrode Group Figures (clips)

% Create responsive figure
figure
imagesc(resp_electrode(:,:,1),[0 35]);
c=colorbar;
ylabel(c,'Number of Electrodes','fontsize',8);
c.FontSize = 12;
colormap bone;
xlabel('Regions','fontsize',10);
ylabel('Regions','fontsize',10);
set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
title('Responders')

% Create nonresponsive figure
figure
imagesc(nonresp_electrode(:,:,1),[0 35]);
c=colorbar;
ylabel(c,'Electrodes','fontsize',8);
c.FontSize = 12;
colormap bone;
xlabel('Regions','fontsize',10);
ylabel('Regions','fontsize',10);
set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
title('Non Responders')



%% Electrode Group Figures (subject)

% Create responsive figure
figure
imagesc(resp_electrode_subject(:,:,1));
c=colorbar;
ylabel(c,'Number of Electrodes','fontsize',8);
c.FontSize = 12;
colormap jet;
xlabel('Regions','fontsize',10);
ylabel('Regions','fontsize',10);
set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
title('Responders')

% Create nonresponsive figure
figure
imagesc(nonresp_electrode_subject(:,:,1));
c=colorbar;
ylabel(c,'Electrodes','fontsize',8);
c.FontSize = 12;
colormap jet;
xlabel('Regions','fontsize',10);
ylabel('Regions','fontsize',10);
set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
title('Non Responders')

%% Structure Group Figures

% Create responsive figure
figure
imagesc(mean_resp_struct,[0 0.5]);
c=colorbar;
ylabel(c,'Mean Fractional Anisotropy','fontsize',8);
c.FontSize = 12;
colormap gray;
xlabel('Regions','fontsize',10);
ylabel('Regions','fontsize',10);
set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
title('Responders')

% Create nonresponsive figure
figure
imagesc(mean_nonresp_struct,[0 0.5]);
c=colorbar;
ylabel(c,'Mean Fractional Anisotropy','fontsize',8);
c.FontSize = 12;
colormap gray;
xlabel('Regions','fontsize',10);
ylabel('Regions','fontsize',10);
set(gca,'XTick',1:length(master_electrode_labels),'XTickLabel',master_electrode_labels,'fontsize',8.5,'TickLabelInterpreter','none');
set(gca,'YTick',1:length(master_electrode_labels),'YTickLabel',master_electrode_labels,'fontsize',8.5,'XTickLabelRotation',90,'TickLabelInterpreter','none');
title('Non Responders')
