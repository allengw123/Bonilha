clear all
close all
clc

gitpath = '/home/bonilha/Documents/GitHub/Bonilha';

cd(gitpath)

allengit_genpath(gitpath)
%% Define Variables

datadir='/media/bonilha/AllenProj/sEEG_project/PatientData';

% Reference sheet
ref_sheet = load(fullfile(datadir,'outcome.mat')).ref_sheet;

% Find subjects
subjID = [dir(fullfile(datadir,'*','Patient*'));dir(fullfile(datadir,'*','3T*'))];
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

tab = tab(responsive_idx|nonresponsive_idx,:);


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

% Thesholds
thres=0.66;

% Create plots
plotfig=false;
%% Find Jaccard similarity coefficient between functional and structural connectivity
jac_coeff = [];
electrode_comp = [];

for sbj = 1:size(subjID,1)
    wk_sbjID = subjID(sbj).name;

    wk_structfolder = fullfile(subjID(sbj).folder,wk_sbjID,'structural','Tractography','Connectivity');
    wk_functionalfolder =fullfile(subjID(sbj).folder,wk_sbjID,'sEEG','matdata_whole');


    if ~isfolder(wk_structfolder)
        disp([wk_sbjID,' missing structural data'])
        continue
    elseif ~isfolder(wk_functionalfolder)
        disp([wk_sbjID,' missing functional data'])
        continue
    end

    wk_sbjID = strrep(wk_sbjID,' ','_');
    wk_sbjID = strrep(wk_sbjID,'3T_','');


    % load structural connectivity data
    struct_con = [];

    % Load data
    tempfv=load(fullfile(wk_structfolder,'fa'));
    templabels=textscan(char(tempfv.name),'%s');
    templabels=templabels{1,1}(1:end-1);

    % replace '_P' with '_S'
    templabels = strrep(templabels,'_P','_S');
    tempdata=tempfv.connectivity;

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

    % Save data
    struct_con=tempconmat;

    % Load functional matricies
    cohmats=dir(fullfile(wk_functionalfolder,'*.mat'));
    func_con=[];
    for i = 1:numel(cohmats)
        tempmat=load(fullfile(cohmats(i).folder,cohmats(i).name)).connectivitymat_grouped;

        % Save functional data
        func_con(:,:,i) = tempmat;
    end


    % Calculate Jaccard Idx
    for clip=1:numel(cohmats)

        funcmat=func_con(:,:,clip);
        structmat=struct_con;

        % Clip type
        if contains(cohmats(clip).name,'rest') || contains(cohmats(clip).name,'baseline')
            c_type = 'baseline';
        else
            c_type = 'seizure';
        end

        tempfunc=funcmat;
        tempstruct=structmat;
        
        if plotfig
            struc_img=figure('Position',[1483,658.333333333333,644,502.666666666667]);
            imagesc(tempstruct);
            set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
            set(struc_img,'color',[1 1 1]);
            cb=colorbar;
            set(cb,'Ticks',[0 0.5 1],'FontSize',16,'Location','eastoutside')
            ylabel(cb,'Fractional Anisotropy (FA)','FontSize',16)
            axis('square')
            struc_ccm=customcolormap([0 1],{'#E28E40','#6E3908'});
            colormap(struc_ccm)
            caxis([0 1]);

            func_img=figure('Position',[1483,658.333333333333,644,502.666666666667]);
            imagesc(tempfunc);
            set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
            set(func_img,'color',[1 1 1]);
            cb=colorbar;
            set(cb,'Ticks',[0 0.5 1],'FontSize',16,'Location','southoutside')
            ylabel(cb,'Beta Coherence','FontSize',16)
            axis('square')
            func_ccm=customcolormap([0 1],{'#4094E2','#063058'});
            colormap(func_ccm)
            caxis([0 1]);
        end


        % Make diag nans for remove nan function
        idx=logical(eye(size(tempfunc,2)));
        tempfunc(idx)=nan;
        tempstruct(idx)=nan;

        % Remove nans
        nanidx_row_struct=find((sum(~isnan(tempstruct),2))==0);
        nanidx_col_struct=find((sum(~isnan(tempstruct),1))==0);

        nanidx_row_func=find((sum(~isnan(tempfunc),2))==0);
        nanidx_col_func=find((sum(~isnan(tempfunc),1))==0);

        nanidx_row=unique([nanidx_row_struct ;nanidx_row_func]);
        nanidx_col=unique([nanidx_col_struct nanidx_col_func]);

        tempfunc(nanidx_row,:)=[];
        tempfunc(:,nanidx_col)=[];

        tempstruct(nanidx_row,:)=[];
        tempstruct(:,nanidx_col)=[];


        % Take only below the diag
        idx=logical(tril(ones(size(tempfunc,1)),-1));

        tempstruct=tempstruct(idx);
        tempfunc=tempfunc(idx);

        %                          % Rearrange matrix to square
        %                          tempfunc=reshape(tempfunc,size(funcmat,1)-1,[]);
        %                          tempstruct=reshape(tempstruct,[size(structmat,1)-1,numel(tempstruct)/(size(structmat,1)-1)]);

        % check if nans exist
        if logical(sum(isnan(tempstruct))) || logical(sum(isnan(tempfunc)))
            disp(['ERROR-- NAN removal failed',wk_sbjID])
            return
        end

        % Check if func mat matches struct mat
        if sum(isnan(tempfunc))>0 || sum(isnan(tempstruct))>0
            disp(['ERROR--func mat and struct mat does not match ',wrk_sbjID])

            eval(['tempfunc=',freq_bands{freq},'func_con(:,:,phas,clip);']);
            tempstruct=struct_con.(fa);

            figure
            subplot(1,2,1)
            imagesc(~isnan(tempfunc))

            subplot(1,2,2)
            imagesc(~isnan(tempstruct))


            return
        end

        % Find non-zero number count
        nz_num=min([nnz(tempstruct) nnz(tempfunc)]);

        % Threshold values
        percent_thres=thres;
        scarcity=round(nz_num*percent_thres);

        tempstruct_idx=sort(tempstruct);
        tempstruct_idx=tempstruct_idx(end-scarcity+1:end);
        tempstruc_min=min(tempstruct_idx);
        [~,tempstruct_idx]=ismember(tempstruct,tempstruct_idx);
        tempstruct_idx=find(tempstruct_idx);

        tempfunc_idx=sort(tempfunc);
        tempfunc_idx=tempfunc_idx(end-scarcity+1:end);
        tempfunc_min=min(tempfunc_idx);
        [~,tempfunc_idx]=ismember(tempfunc,tempfunc_idx);
        tempfunc_idx=find(tempfunc_idx);

        all_idx=unique([tempfunc_idx; tempstruct_idx]);


        if plotfig
            plotstruc=tempstruct(tempstruct_idx);
            plotfunc=tempfunc(tempfunc_idx);

            % Create vectorized plot
            figure('Position',[917,41.6666666666667,182,1319.33333333333]);
            set(gcf,'color',[1 1 1]);

            subplot(1,4,[1:3])
            plot(plotfunc,1:numel(plotfunc),'Color','#4094E2','LineWidth',2)
            yticklabels([]);
            yticks([])
            xlim([0.5 1])
            xticks([])
            tempx=get(gca,'XLim');

            subplot(1,4,4)
            imagesc(plotfunc)
            colormap(func_ccm)
            caxis([0 1])
            xlim(tempx)
            yticklabels([]);
            xticklabels([]);
            yticks([])
            xticks([])

            figure('Position',[917,41.6666666666667,182,1319.33333333333]);
            set(gcf,'color',[1 1 1]);

            subplot(1,4,[1:3])
            plot(plotstruc,1:numel(plotstruc),'Color','#E28E40','LineWidth',2)
            yticklabels([]);
            yticks([])
            xlim([0.5 1])
            xticks([])
            tempx=get(gca,'XLim');

            subplot(1,4,4)
            imagesc(plotstruc)
            colormap(struc_ccm)
            caxis([0 1])
            xlim(tempx)
            yticklabels([]);
            xticklabels([]);
            yticks([])
            xticks([])
        end

        % Copy matrix for label
        compstruct=tempstruct;
        compfunc=tempfunc;

        % Binarize matricies
        tempstruct=tempstruct>=tempstruc_min;
        tempfunc=tempfunc>=tempfunc_min;

        if plotfig

            figure('Position',[917,41.6666666666667,182,1319.33333333333]);
            set(gcf,'color',[1 1 1]);
            subplot(1,2,1)
            plot(tempfunc,1:numel(tempfunc),'Color','#4094E2','LineWidth',2)
            yticklabels([]);
            yticks([])
            xlim([0.5 1])
            xticks([])
            tempx=get(gca,'XLim');

            subplot(1,2,2)
            imagesc(tempfunc)
            colormap(func_ccm)
            caxis([0 1])
            xlim(tempx)
            yticklabels([]);
            xticklabels([]);
            yticks([])
            xticks([])

            figure('Position',[917,41.6666666666667,182,1319.33333333333]);
            set(gcf,'color',[1 1 1]);
            subplot(1,2,1)
            plot(tempstruct,1:numel(tempstruct),'Color','#E28E40','LineWidth',2)
            yticklabels([]);
            yticks([])
            xlim([0.5 1])
            xticks([])
            tempx=get(gca,'XLim');

            subplot(1,2,2)
            imagesc(tempstruct)
            colormap(struc_ccm)
            caxis([0 1])
            xlim(tempx)
            yticklabels([]);
            xticklabels([]);
            yticks([])
            xticks([])
        end

        % Calculate Jaccard Coeff
        tempfunc=tempfunc(all_idx);
        tempstruct=tempstruct(all_idx);
        jac_coeff.(c_type){sbj,clip}=jaccard(tempfunc,tempstruct);


        % Calculate label
        compstruct=compstruct(all_idx);
        compfunc=compfunc(all_idx);

        edge_idx=tempfunc&tempstruct;

        compstruct=compstruct(edge_idx);
        compfunc=compfunc(edge_idx);

        struc_idx=ismember(round(structmat,6),round(compstruct,6));
        func_idx=ismember(round(funcmat,6),round(compfunc,6));

        electrode_idx=struc_idx&func_idx;

        if (nnz(electrode_idx)/2)~=nnz(edge_idx)
            disp('ERROR in label matching between func and struct')
            return
        end


        % Save label matrix
        electrode_comp{sbj,clip}=double(electrode_idx);
    end
end

%% Baseline

% Baseline
wk_dat = jac_coeff.baseline;

% Calculate mean
mean_dat = [];
num_clips = [];
for i = 1:size(wk_dat,1)
    mean_dat = [mean_dat; mean(cell2mat(wk_dat(i,:)))];
    num_clips = [num_clips;sum(~cellfun(@isempty,(wk_dat(i,:))))];
end

% Divide into resp/nonresp
resp_mean_base = mean_dat(responsive_idx);
resp_ind_base = cat(1,wk_dat{responsive_idx,:});
resp_clip_base = num_clips(responsive_idx);
resp_name = {subjID(responsive_idx).name};

nonresp_mean_base = mean_dat(nonresponsive_idx);
nonresp_ind_base = cat(1,wk_dat{nonresponsive_idx,:});
nonresp_clip_base = num_clips(nonresponsive_idx);
nonresp_score = response(nonresponsive_idx);
nonresp_name = {subjID(nonresponsive_idx).name};


% Find statistics of resp/nonresp
mean_resp_base = mean(resp_mean_base);
sem_resp_base = std(resp_mean_base)/sqrt(numel(resp_mean_base));

mean_nonresp_base = mean(nonresp_mean_base);
sem_nonresp_base = std(nonresp_mean_base)/sqrt(numel(nonresp_mean_base));

figure
tiledlayout(2,1)
nexttile
bar(resp_clip_base)
ylim([0 15])
xticklabels(resp_name)
ylabel('# of baseline clips')
title('Responders')
nexttile
bar(nonresp_clip_base)
xticklabels(nonresp_name)
ylim([0 15])
ylabel('# of baseline clips')
title('NonResponders')


figure
bar([1 2],[mean_resp_base mean_nonresp_base])
hold on
errorbar([mean_resp_base mean_nonresp_base],[sem_resp_base sem_nonresp_base],'vertical','Color',[0 0 0],'Linestyle','none')
swarmchart(ones(size(resp_ind_base)),resp_ind_base)
swarmchart(ones(size(nonresp_ind_base))*2,nonresp_ind_base,'x')
ylabel('Mean Jaccard Index at Rest')
xticklabels({'Resp','NonResp'})
[h,p,ci,stats] = ttest2(resp_mean_base,nonresp_mean_base);
title(['Baseline pval=',num2str(p)])
ylim([0 1])

%% Seizure

% Seizure
wk_dat = jac_coeff.seizure;

% Calculate mean
num_clips = [];
mean_dat = [];
for i = 1:size(wk_dat,1)
    mean_dat = [mean_dat; mean(cell2mat(wk_dat(i,:)))];
    num_clips = [num_clips;sum(~cellfun(@isempty,(wk_dat(i,:))))];

end

% Divide into resp/nonresp
resp_mean_sez = mean_dat(responsive_idx);
resp_clip_sez = num_clips(responsive_idx);

nonresp_mean_sez = mean_dat(nonresponsive_idx);
nonresp_clip_sez = num_clips(nonresponsive_idx);

% Find statistics of resp/nonresp
mean_resp_sez = mean(resp_mean_sez);
sem_resp_sez = std(resp_mean_sez)/sqrt(numel(resp_mean_sez));

mean_nonresp_sez = mean(nonresp_mean_sez);
sem_nonresp_sez = std(nonresp_mean_sez)/sqrt(numel(nonresp_mean_sez));


figure
bar([1 2],[mean_resp_sez mean_nonresp_sez])
hold on
errorbar([mean_resp_sez mean_nonresp_sez],[sem_resp_sez sem_nonresp_sez],'vertical','Color',[0 0 0],'Linestyle','none')
xticklabels({'Resp','NonResp'})
[h,p,ci,stats] = ttest2(resp_mean_sez,nonresp_mean_sez);
title(['Seizure ',num2str(p)])
ylim([0 1])
%% Baseline vs Seizure

figure

% Responsive
nexttile
bar([1 2],[mean_resp_base mean_resp_sez])
hold on
errorbar([mean_resp_base mean_resp_sez],[sem_resp_base sem_resp_sez],'vertical','Color',[0 0 0],'Linestyle','none')
plot([resp_mean_base resp_mean_sez]','-o')
xticklabels({'Baseline','Seizure'})
[h,p,ci,stats] = ttest2(resp_mean_base,resp_mean_sez);
title(['Responsive p=',num2str(p)])
ylim([0 1])

% nonresponsive
nexttile
bar([1 2],[mean_nonresp_base mean_nonresp_sez])
hold on
errorbar([mean_nonresp_base mean_nonresp_sez],[sem_nonresp_base sem_nonresp_sez],'vertical','Color',[0 0 0],'Linestyle','none')
plot([nonresp_mean_base nonresp_mean_sez]','-o')
xticklabels({'Baseline','Seizure'})
[h,p,ci,stats] = ttest2(nonresp_mean_base,nonresp_mean_sez);
title(['Nonresponsive p=',num2str(p)])
ylim([0 1])
