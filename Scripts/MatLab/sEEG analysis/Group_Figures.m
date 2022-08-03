clear all
close all
clc

% gitpath='C:\Users\allen\Documents\GitHub\Bonilha';
gitpath = '/home/bonilha/Documents/GitHub/Bonilha';

cd(gitpath)
allengit_genpath(gitpath)
%% Insert Info
datadir='/media/bonilha/AllenProj/sEEG_project/PatientData';
analysisdir=fullfile(datadir,'Analysis');

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
            coh_comb_seizure(:,:,:,q)=coh_comb_temp(:,:,[1 3 4 5]);
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
std_resp_coh=std(resp_coh,[],4,'omitnan');

count=0;
for r=1:numel(nonresp)
    countstart=count+1;
    count=count+size(beta_coh.(nonresp{r}),4);
    nonresp_coh(:,:,:,countstart:count)=beta_coh.(nonresp{r});
end
mean_nonresp_coh=mean(nonresp_coh,4,'omitnan');
std_nonresp_coh=std(nonresp_coh,[],4,'omitnan');

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
std_resp_struct=std(resp_struct,[],3,'omitnan');

nonresp_struct =[];
for r=1:numel(nonresp)
    nonresp_struct(:,:,r)=struct_con.(nonresp{r});
end
mean_nonresp_struct=mean(nonresp_struct,3,'omitnan');
std_nonresp_struct=std(nonresp_struct,[],3,'omitnan');

%% Coherence Group Figures

% Create responsive figure
for i=1:4
    figure
    imagesc(mean_resp_coh(:,:,i),[0 1]);
    set(gcf,'color',[1 1 1]);
    c=colorbar;
    func_ccm=customcolormap([0 1],{'#4094E2','#063058'});
    colormap(func_ccm)
    ylabel(c,'Beta coherence','fontsize',12);
    title(['Responders - ',trials_label{i}])
    axis('square')
    meancoh=mean(resp_coh(:,:,i,:),'all','omitnan');
    stdcoh=std(mean_resp_coh(:,:,i),[],'all','omitnan');
    disp(['Mean is ', num2str(meancoh),'. STD is ',num2str(stdcoh)])
end

% Create nonresponsive figure
for i=1:4
    figure
    imagesc(mean_nonresp_coh(:,:,i),[0 1]);
    set(gcf,'color',[1 1 1]);
    c=colorbar;
    func_ccm=customcolormap([0 1],{'#4094E2','#063058'});
    colormap(func_ccm)
    ylabel(c,'Beta coherence','fontsize',12);
    title(['Non-Responders - ',trials_label{i}])
    axis('square')
    meancoh=mean(mean_nonresp_coh(:,:,i),'all','omitnan');
    stdcoh=std(mean_nonresp_coh(:,:,i),[],'all','omitnan');
    disp(['Mean is ', num2str(meancoh),'. STD is ',num2str(stdcoh)])
end

%% Electrode Group Figures (clips)

% Create responsive figure
figure
imagesc(resp_electrode(:,:,1),[0 35]);
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gcf,'color',[1 1 1]);
cb=colorbar;
set(cb,'Ticks',[0 35],'FontSize',16,'Location','southoutside')
ylabel(cb,'Clips','FontSize',16)
axis('square')
ccm=customcolormap([0 1],{'#33B585','#00613E'});
colormap(ccm)
title('Responders')

% Create nonresponsive figure
figure
imagesc(nonresp_electrode(:,:,1),[0 35]);
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gcf,'color',[1 1 1]);
cb=colorbar;
set(cb,'Ticks',[0 35],'FontSize',16,'Location','southoutside')
ylabel(cb,'Clips','FontSize',16)
axis('square')
ccm=customcolormap([0 1],{'#33B585','#00613E'});
colormap(ccm)
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
imagesc(mean_resp_struct,[0 1]);
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gcf,'color',[1 1 1]);
cb=colorbar;
set(cb,'Ticks',[0 0.5 1],'FontSize',16,'Location','southoutside')
ylabel(cb,'Fractional Anisotropy','FontSize',16)
axis('square')
struc_ccm=customcolormap([0 1],{'#E28E40','#6E3908'});
colormap(struc_ccm)
caxis([0 1]);
title('Responders')
meanfa=mean(mean_resp_struct,'all','omitnan');
stdfa=std(mean_resp_struct,[],'all','omitnan');
disp(['Mean is ', num2str(meanfa),'. STD is ',num2str(stdfa)])

figure
imagesc(std_resp_struct);
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gcf,'color',[1 1 1]);
cb=colorbar;
set(cb,'FontSize',16,'Location','southoutside')
ylabel(cb,'Standard Deviation of Fractional Anisotropy','FontSize',16)
axis('square')
struc_ccm=customcolormap([0 1],{'#E28E40','#6E3908'});
colormap(struc_ccm)
caxis([0 max(std_resp_struct,[],'all')]);
title('Responders')

% Create nonresponsive figure
figure
imagesc(mean_nonresp_struct,[0 1]);
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gcf,'color',[1 1 1]);
cb=colorbar;
set(cb,'Ticks',[0 0.5 1],'FontSize',16,'Location','southoutside')
ylabel(cb,'Fractional Anisotropy','FontSize',16)
axis('square')
struc_ccm=customcolormap([0 1],{'#E28E40','#6E3908'});
colormap(struc_ccm)
caxis([0 1]);
title('Non Responders')

figure
imagesc(std_nonresp_struct);
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gcf,'color',[1 1 1]);
cb=colorbar;
set(cb,'FontSize',16,'Location','southoutside')
ylabel(cb,'Standard Deviation of Fractional Anisotropy','FontSize',16)
axis('square')
struc_ccm=customcolormap([0 1],{'#0068FF','#022454'});
colormap(struc_ccm)
caxis([0 max(std_nonresp_struct,[],'all')]);
title('Non-Responders')

%% t-test structure non-resp vs resp

resp_fa = cat(3,struct_con.P001, ...
    struct_con.P002, ...
    struct_con.P003, ...
    struct_con.P006, ...
    struct_con.P009, ...
    struct_con.P012, ...
    struct_con.P013);

nonresp_fa =cat(3,struct_con.P502, ...
    struct_con.P503);

[~,P,~,STATS] = ttest2(resp_fa,nonresp_fa,'dim',3);
T = STATS.tstat;

b_P = P;
b_T = T;

b_P(P > 0.05/numel(find(~isnan(STATS.tstat)))) = NaN;
b_T(P > 0.05/numel(find(~isnan(STATS.tstat)))) = NaN;
figure;
imagesc(b_T)
title('t-values FA (bonf)')
colorbar
%% t-test structure non-resp vs resp coherence

phase = {'Basline','Early','Mid','Late'};
for t = 1:4
    resp_coh = cat(4,beta_coh.P001(:,:,t,:), ...
        beta_coh.P002(:,:,t,:), ...
        beta_coh.P003(:,:,t,:), ...
        beta_coh.P006(:,:,t,:), ...
        beta_coh.P009(:,:,t,:), ...
        beta_coh.P012(:,:,t,:), ...
        beta_coh.P013(:,:,t,:));
    
    nonresp_coh =cat(4,beta_coh.P502(:,:,t,:), ...
        beta_coh.P503(:,:,t,:));
    
    
    [~,P,~,STATS] = ttest2(permute(resp_coh,[1 2 4 3]),permute(nonresp_coh,[1 2 4 3]),'dim',3);
    T = STATS.tstat;
    
    b_P = P;
    b_T = T;
    
    b_P(P > 0.05/numel(find(~isnan(STATS.tstat)))) = NaN;
    b_T(P > 0.05/numel(find(~isnan(STATS.tstat)))) = NaN;
    figure;
    imagesc(b_T)
    title('t-values Coherence (bonf)')
    subtitle(phase{t})
    colorbar
end

fn = fieldnames(beta_coh);
for i = 1:9
    size(beta_coh.(fn{i}),4)
end