clear all
close all
clc

gitpath = '/home/bonilha/Documents/GitHub/Bonilha';

% gitpath='C:\Users\allen\Documents\GitHub\Bonilha';
cd(gitpath)

allengit_genpath(gitpath)
%% Define Variables

% datadir='C:\Users\allen\Box Sync\Desktop\Bonilha\Projects\sEEG project\PatientData';
datadir='/media/bonilha/AllenProj/sEEG_project/PatientData';

% Create Functional analysis folder
functionaldir=fullfile(datadir,'Analysis','Functional');
mkdir(functionaldir);

% Create Structural analysis folder
structuraldir=fullfile(datadir,'Analysis','Structural');
mkdir(structuraldir);

% Find subjects
subjID = {dir(fullfile(datadir,'Patient *')).name};
subjnum = regexp(subjID,'\d*','Match');

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

% Phases
phases={'Pre-Baseline','Pre-Trans','Post-Trans','Mid-Seiz','Late-Seiz','Early-Post','Late-Post'};
% Thesholds
thres_name={'p66'};
thres_val=[0.66];

% Frequences
freq_bands={'alpha_theta','beta','low_gamma','high_gamma'};

% Measurments
funcvars={'fa'};

% Create plots
plotfig=false;
%% Find Jaccard similarity coefficient between functional and structural connectivity
for sbj=1:numel(subjID)
    wrk_sbjID=subjID{sbj};
    
    temp_structfolder=fullfile(datadir,wrk_sbjID,'structural','Tractography','Connectivity');
    temp_functionalfolder=fullfile(datadir,wrk_sbjID,'sEEG','matdata');

    
    if ~isfolder(temp_structfolder)
        disp([wrk_sbjID,' missing structural data'])
        continue
    elseif ~isfolder(temp_functionalfolder)
        disp([wrk_sbjID,' missing functional data'])
        continue
    end
    
    temp_matrixfolder=fullfile(datadir,wrk_sbjID,'Matrix Images');
    mkdir(temp_matrixfolder)
    
    % Check sEEG electrodes and structural electrodes are matched
    count=load(fullfile(temp_structfolder,'count.mat'));
    labels=textscan(char(count.name),'%s');
    
    struct_electrodes=labels{1,1}(1:end-1);
    struct_electrodes=unique(extractBefore(struct_electrodes,'_'));
    
    func_electrodes=load(fullfile(datadir,wrk_sbjID,'sEEG','Electrodes.mat')).Electrodes;
    func_electrodes=unique(cellfun(@(x) [x{:}],regexp(func_electrodes,'\D','match'),'UniformOutput',false));
    
%     if nnz(~strcmp(sort(func_electrodes),sort(struct_electrodes))) ~=0
%         disp(['Structural and Functional Electrode MISMATCH....',wrk_sbjID])
%         return
%     end
    
    % load structural connectivity data
    struct_con=[];
    for fv=1:numel(funcvars)
        
        % Load data
        tempfv=load(fullfile(temp_structfolder,funcvars{fv}));
        templabels=textscan(char(tempfv.name),'%s');
        templabels=templabels{1,1}(1:end-1);
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
        
        % 0-1 Normalization
        tempconmat(~isnan(tempconmat))=mat2gray(tempconmat(~isnan(tempconmat)));
        
        % Save data
        struct_con.(funcvars{fv})=tempconmat;
    end
    
    
    % Load functional matricies
    for freq=1:numel(freq_bands)
        
        % load coherence seizure data
        cohmats={dir(fullfile(datadir,wrk_sbjID,'sEEG','matdata',['*',freq_bands{freq},'_P*'])).name};
        func_con.(freq_bands{freq})=[];
        for i=1:numel(cohmats)
            tempmat=load(fullfile(datadir,wrk_sbjID,'sEEG','matdata',cohmats{i})).connectivitymat_grouped;

            % Save functional data
            func_con.(freq_bands{freq})(:,:,:,i)=tempmat;
        end
    end
    
    
    % Jaccard Idx
    for freq=1:numel(freq_bands)
        for thres=1:numel(thres_val)
            for fv=1:numel(funcvars)
                for clip=1:numel(cohmats)
                    for phas=1:numel(phases)
                        funcmat=func_con.(freq_bands{freq})(:,:,phas,clip);
                        structmat=struct_con.(funcvars{fv});
                        
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
                            disp(['ERROR-- NAN removal failed',wrk_sbjID])
                            return
                        end

    %                     % Check if func mat matches struct mat
    %                     if sum(isnan(tempfunc))>0 || sum(isnan(tempstruct))>0
    %                         disp(['ERROR--func mat and struct mat does not match ',wrk_sbjID])
    %                         
    %                          eval(['tempfunc=',freq_bands{freq},'func_con(:,:,phas,clip);']);
    %                          tempstruct=struct_con.(funcvars{fv});
    %                          
    %                          figure
    %                          subplot(1,2,1)
    %                          imagesc(~isnan(tempfunc))
    %                          
    %                          subplot(1,2,2)
    %                          imagesc(~isnan(tempstruct))
    % 
    %                         
    %                         return
    %                     end

                        % Find non-zero number count
                        nz_num=min([nnz(tempstruct) nnz(tempfunc)]);

                        % Threshold values
                        percent_thres=thres_val(thres);
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
                            plotfunc=tempstruct(tempstruct_idx);
                            plotstruc=tempfunc(tempfunc_idx);
                            
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
%                         snapnow;

                        jac_coeff.(freq_bands{freq}).(thres_name{thres}).(['sbj_',extractAfter(wrk_sbjID,'Patient ')]).(funcvars{fv}){clip}(phas)=jaccard(tempfunc,tempstruct);

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
                        electrode_comp.(freq_bands{freq}).(thres_name{thres}).(['sbj_',extractAfter(wrk_sbjID,'Patient ')]).(funcvars{fv}){clip}{phas}=double(electrode_idx);
                        
                        
                        
                        
%                     % Create Binary Matrix Colormap of Jaccard comparison and save
%                     cm=figure('units','normalized','outerposition',[0 0 1 1]);
%                     subplot(1,2,1)
%                     imagesc(tempfunc);
%                     title([freq_bands{freq},'Coherence'])
%                     subplot(1,2,2)
%                     imagesc(tempstruct);
%                     title(funcvars{fv})
%                     sgtitle([alphacohmats{clip},'--',phases{phas},'--',num2str(percent_thres),' percentile--JC ',num2str(jaccard(tempfunc,tempstruct))])
% 
%                     temptitle=[funcvars{fv},'--',freq_bands{freq},'--',phases{phas},'--',alphacohmats{clip},'--',num2str(percent_thres)];
%                     savefig(cm,fullfile(temp_matrixfolder,[temptitle,'.fig']))
% 
%                     close(cm)
                    end
                end
            end
        end
    end
end
%% Figure creation

% Organize figure data
for band=1:numel(freq_bands)
    for thres=1:numel(thres_val)
        fnames=fieldnames(jac_coeff.(freq_bands{band}).(thres_name{thres}));
        for sbj=1:numel(fnames)
            for fv=1:numel(funcvars)
                tempdat=jac_coeff.(freq_bands{band}).(thres_name{thres}).(fnames{sbj}).(funcvars{fv});
                figdat.(freq_bands{band}).(thres_name{thres}).(funcvars{fv}){sbj,:}=cell2mat(tempdat');
            end
        end
    end
end 

active_sbj=extractAfter(fnames,'sbj_');

% Scatter/Bar avg Figure
for fv=1:numel(funcvars)
    for thres=1:numel(thres_val)
        figure;
        for band=1:numel(freq_bands)
            for resp=1:2
                currentfig=subplot(2,numel(freq_bands),band+(resp-1)*4);
                ylim([0 1])
                wkdat=figdat.(freq_bands{band}).(thres_name{thres}).(funcvars{fv});
                hold on
                
                if resp==1
                    wkdat=wkdat(1:end-3);
                elseif resp==2
                    wkdat=wkdat(end-2:end);
                end
                
                % Bar Mean
                bardat=cell2mat(wkdat);
                cb=bar(1:size(bardat,2),mean(bardat,1));
                error_bar_location = cb.XEndPoints;

                % Error
                err=std(bardat,0,1)./sqrt(size(bardat,1));
                errorbar(error_bar_location',mean(bardat,1),err,'k', 'linestyle', 'none');

                % Check for sig.
                normality=kstest(bardat(:));

                max_data=max(bardat,[],'all');

                if normality==0
                    [~,~,stats]=anova1(bardat,[],'off');
                    pvalues=multcompare(stats,'Display','off');
                    test='an';
                else
                    [~,~,stats]=kruskalwallis(bardat,[],'off');
                    pvalues=multcompare(stats,'Display','off');
                    test='kw';
                end


                % Find 0.05 significance
                last_max=[];
                if any(le(pvalues(:,6),0.05))
                   idx=find(le(pvalues(:,6),0.05));
                   spacer=diff(get(currentfig,'YLim'))*0.02;
                   for m=1:numel(idx)
                       lstart=error_bar_location(pvalues(idx(m),1));
                       lend=error_bar_location(pvalues(idx(m),2));
                       if lstart>missing
                          delay=sum(lstart>missing);
                          lstart=lstart+delay;
                       end
                       if lend>missing
                          delay=sum(lend>missing);
                          lend=lend+delay;
                       end
                       l=line(currentfig,[lstart lend],[1 1]*max_data+m*spacer);
                       set(l,'linewidth',2)
                       t=text(currentfig,mean([lstart lend]),max_data+(m+.5)*spacer,[test,num2str(pvalues(idx(m),6))],'HorizontalAlignment','center','FontSize',10);
                   end
                   last_max=t.Position(2);
                end

                % Find 0.10 trending
                if any(le(pvalues(:,6),0.10)&ge(pvalues(:,6),0.05))
                   idx=find(le(pvalues(:,6),0.10)&ge(pvalues(:,6),0.05));
                   if isempty(last_max)
                       spacer=diff(get(currentfig,'YLim'))*0.02;
                   else
                       max_data=last_max;
                   end
                   for m=1:numel(idx)
                       lstart=error_bar_location(pvalues(idx(m),1));
                       lend=error_bar_location(pvalues(idx(m),2));
                       if any(lstart>=missing)
                          delay=sum(lstart>=missing);
                          lstart=lstart+delay;
                       end
                       if any(lend>=missing)
                          delay=sum(lend>=missing);
                          lend=lend+delay;
                       end
                       l=line(currentfig,[lstart lend],[1 1]*max_data+m*spacer);
                       set(l,'linewidth',2)
                       t=text(currentfig,mean([lstart lend]),max_data+(m+.5)*spacer,[test,num2str(pvalues(idx(m),6))],'HorizontalAlignment','center','FontSize',10);
                       set(l,'Color','r')
                   end
                end


                scat=[];
                for sbj=1:size(wkdat,1)
                    tempdat=wkdat{sbj};
                    scatterx=1:size(tempdat,2);
                    offset=-0.25:0.5/size(wkdat,1):0.25;
                    scatterxmat=repmat(scatterx+offset(sbj),size(tempdat,1),1);
                    markers={'o','+','*','.','x','_','|','s','d','^','v','>','<','p','h'};
                    scat{sbj}=scatter(scatterxmat(:),tempdat(:),100,markers{sbj});
                end
                set(gca,'XTick',(1:7),'XTickLabel',phases,'XTickLabelRotation',45);
                ylabel('Jaccard Idx')
                xlabel('Phases')
                title([freq_bands{band}]);
                switch band+(resp-1)*4
                    case 4
                        legend([scat{:}],active_sbj(1:end-3),'Orientation','horizontal')
                    case 8
                        legend([scat{:}],active_sbj(end-2:end),'Orientation','horizontal')
                end
            end
        end
        sgtitle([funcvars{fv},'-',thres_name{thres}]);
    end
end


% Line Figure
for fv=1:numel(funcvars)
    for thres=1:numel(thres_val)
        figure;
        for band=1:numel(freq_bands)
            currentfig=subplot(1,numel(freq_bands),band);
            ylim([0 1])
            wkdat=figdat.(freq_bands{band}).(thres_name{thres}).(funcvars{fv});
            hold on

            % line Mean
            linedat=cell2mat(cellfun(@(x) mean(x,1),wkdat,'UniformOutput',false));
            ln=plot(linedat','LineWidth',3);

            % Error
            err=cell2mat(cellfun(@(x) std(x,0,1)./sqrt(size(x,1)),wkdat,'UniformOutput',false));
            err_bar=errorbar(repmat([1:7],size(linedat,1),1)',linedat',err','k', 'linestyle', 'none');
            for line_num=1:numel(ln)
                err_bar(line_num).Color=ln(line_num).Color;
            end
            
            % Change line style if treatment resistant
            for line_num=1:numel(ln)
                if strcmp(active_sbj{line_num},'502') || strcmp(active_sbj{line_num},'503')
                    ln(line_num).LineStyle=':';
                end
            end
            
            set(gca,'XTick',(1:7),'XTickLabel',phases,'XTickLabelRotation',45);
            ylabel('Jaccard Idx')
            xlabel('Phases')
            title([freq_bands{band}]);
        end
        sgtitle([funcvars{fv},'-',thres_name{thres}]);
        legend(active_sbj,'Orientation','horizontal')
    end
end

%% 

%%%% Examine Beta Freq, 66% threshold, ncount2
tempdat=figdat.beta.p66.ncount2;

% % Calculate median
% tempavg=cellfun(@(x) median(x,1),tempdat,'UniformOutput',false);
% inputmat=reshape([tempavg{:}],[7,numel(tempdat)])';

% % Normalize to baseline
% for r=1:size(inputmat,1)
%     tempbl=inputmat(r,1);
%     for c=1:size(inputmat,2)
%         inputmat(r,c)=inputmat(r,c)-tempbl;
%     end
% end

% % Calculate phase difference
% phasediff=[];
% for c=2:size(inputmat,2)
%     phasediff(:,c-1)=inputmat(:,c)-inputmat(:,c-1);
% end

% % Calculate phase difference (% change)
% phasediff=[];
% for c=2:size(inputmat,2)
%     phasediff(:,c-1)=(inputmat(:,c)-inputmat(:,c-1))./inputmat(:,c-1)*100;
% end


% Assume each clip independent
respdat=cell2mat(tempdat(1:7));
nonrespdat=cell2mat(tempdat(8:9));

% Normalize to baseline
for r=1:size(respdat,1)
    tempbl=respdat(r,1);
    for c=1:size(respdat,2)
        respnorm(r,c)=respdat(r,c)-tempbl;
    end
end

for r=1:size(nonrespdat,1)
    tempbl=nonrespdat(r,1);
    for c=1:size(nonrespdat,2)
        nonrespnorm(r,c)=nonrespdat(r,c)-tempbl;
    end
end

% Normalize to baseline (%)
for r=1:size(respdat,1)
    tempbl=respdat(r,1);
    for c=1:size(respdat,2)
        respnorm(r,c)=(respdat(r,c)-tempbl)./tempbl*100;
    end
end

for r=1:size(nonrespdat,1)
    tempbl=nonrespdat(r,1);
    for c=1:size(nonrespdat,2)
        nonrespnorm(r,c)=(nonrespdat(r,c)-tempbl)./tempbl*100;
    end
end


% Calculate phase difference
respdatdiff=[];
for c=2:size(respdat,2)
    respdatdiff(:,c-1)=respdat(:,c)-respdat(:,c-1);
end

nonrespdatdiff=[];
for c=2:size(nonrespdat,2)
    nonrespdatdiff(:,c-1)=nonrespdat(:,c)-nonrespdat(:,c-1);
end


% Calculate phase difference (% Change)
respdatdiff=[];
for c=2:size(respdat,2)
    respdatdiff(:,c-1)=(respdat(:,c)-respdat(:,c-1))./respdat(:,c)*100;
end

nonrespdatdiff=[];
for c=2:size(nonrespdat,2)
    nonrespdatdiff(:,c-1)=(nonrespdat(:,c)-nonrespdat(:,c-1))./nonrespdat(:,c)*100;
end

%%%%%%% Examine Beta Freq, 66% threshold, fa
tempdat=figdat.beta.p66.fa;

% Calculate median
tempavg=cellfun(@(x) median(x,1),tempdat,'UniformOutput',false);
inputmat=reshape([tempavg{:}],[7,numel(tempdat)])';

% Normalize to baseline
for r=1:size(inputmat,1)
    tempbl=inputmat(r,1);
    for c=1:size(inputmat,2)
        inputmat(r,c)=inputmat(r,c)-tempbl;
    end
end

% Calculate phase difference
phasediff=[];
for c=2:size(inputmat,2)
    phasediff(:,c-1)=inputmat(:,c)-inputmat(:,c-1);
end

% Calculate phase difference (% change)
phasediff=[];
for c=2:size(inputmat,2)
    phasediff(:,c-1)=(inputmat(:,c)-inputmat(:,c-1))./inputmat(:,c-1)*100;
end


% Assume each clip independent
respdat=cell2mat(tempdat(1:7));
nonrespdat=cell2mat(tempdat(8:9));


% Assume each clip independent (LiTT)
% respdat=cell2mat(tempdat([4 7]));
% nonrespdat=cell2mat(tempdat(8:9));

% Normalize to baseline
respnorm = [];
for r=1:size(respdat,1)
    tempbl=respdat(r,1);
    for c=1:size(respdat,2)
        respnorm(r,c)=respdat(r,c)-tempbl;
    end
end

nonrespnorm = [];
for r=1:size(nonrespdat,1)
    tempbl=nonrespdat(r,1);
    for c=1:size(nonrespdat,2)
        nonrespnorm(r,c)=nonrespdat(r,c)-tempbl;
    end
end

x=respnorm(:,[3 4 5]);
y=nonrespnorm(:,[3 4 5]);

x=respnorm(:,[1 3 4 5]);
y=nonrespnorm(:,[1 3 4 5]);


% Calculate phase difference
respdatdiff=[];
for c=2:size(respdat,2)
    respdatdiff(:,c-1)=respdat(:,c)-respdat(:,c-1);
end

nonrespdatdiff=[];
for c=2:size(nonrespdat,2)
    nonrespdatdiff(:,c-1)=nonrespdat(:,c)-nonrespdat(:,c-1);
end


% Calculate phase difference (% Change)
respdatdiff=[];
for c=2:size(respdat,2)
    respdatdiff(:,c-1)=(respdat(:,c)-respdat(:,c-1))./respdat(:,c)*100;
end

nonrespdatdiff=[];
for c=2:size(nonrespdat,2)
    nonrespdatdiff(:,c-1)=(nonrespdat(:,c)-nonrespdat(:,c-1))./nonrespdat(:,c)*100;
end


%% Examine Beta Freq, 66% threshold (electrode)
inputdat=electrode_comp.beta.p66;
fn=fieldnames(inputdat);
for sbj=1:numel(fn)
    for meas=1:numel(funcvars)
        tempdat=inputdat.(fn{sbj}).(funcvars{meas});
        for trials=1:numel(tempdat)
            temptrialdat=tempdat{trials};
            for phas=1:size(temptrialdat,2)
                tempelectrode{phas}(:,:,trials)=temptrialdat{phas};
            end
        end
        electrodes.(funcvars{meas})(sbj,:)=tempelectrode;
        tempelectrode=[];
    end
end

%%% FA
tempdat=electrodes.fa;
tempdat=cellfun(@(x) sum(x,3),tempdat,'UniformOutput',false);

% Group Treatment vulnerable participants
electrode_figdat=[];
for c=1:size(tempdat,2) 
    temp=[];
    for r=1:7
        temp=cat(3,temp,tempdat{r,c});
    end
    electrode_figdat{1,c}=temp;
end


% Group Treatment resistant participants
for c=1:size(tempdat,2) 
    temp=[];
    for r=8:9
        temp=cat(3,temp,tempdat{r,c});
    end
    electrode_figdat{2,c}=temp;
end

electrode_figdat=cellfun(@(x) sum(x,3),electrode_figdat,'UniformOutput',false);

for i=1:size(electrode_figdat,2)
    figure;
    
    % Plot treament vul
    vuldat=electrode_figdat{1,i};
    subplot(1,2,1)   
    imagesc(vuldat);
    set(gca,'XTick',1:numel(master_electrode_labels_grouped),'XTickLabel',master_electrode_labels_grouped,'XTickLabelRotation',90)
    set(gca,'YTick',1:numel(master_electrode_labels_grouped),'YTickLabel',master_electrode_labels_grouped)
    title('Treatment Vul');
    cb=colorbar;
    ylabel(cb,'# of electrode pairs')
    colormap('jet')
    
    % Plot treament resist
    resdat=electrode_figdat{2,i};
    subplot(1,2,2)   
    imagesc(resdat);
    set(gca,'XTick',1:numel(master_electrode_labels_grouped),'XTickLabel',master_electrode_labels_grouped,'XTickLabelRotation',90)
    set(gca,'YTick',1:numel(master_electrode_labels_grouped),'YTickLabel',master_electrode_labels_grouped)
    title('Treatment resist');
    cb=colorbar;
    ylabel(cb,'# of electrode pairs')
    colormap('jet')
    
    % calculate jac coef between two groups
    jc=jaccard((vuldat>0),(resdat>0));
    
    sgtitle([phases{i},'-FA-(',num2str(jc),')']);
end
    

%%% ncount2
tempdat=electrodes.ncount2;
tempdat=cellfun(@(x) sum(x,3),tempdat,'UniformOutput',false);

% Group Treatment vulnerable participants
electrode_figdat=[];
for c=1:size(tempdat,2) 
    temp=[];
    for r=1:7
        temp=cat(3,temp,tempdat{r,c});
    end
    electrode_figdat{1,c}=temp;
end


% Group Treatment resistant participants
for c=1:size(tempdat,2) 
    temp=[];
    for r=8:9
        temp=cat(3,temp,tempdat{r,c});
    end
    electrode_figdat{2,c}=temp;
end

electrode_figdat=cellfun(@(x) sum(x,3),electrode_figdat,'UniformOutput',false);

for i=1:size(electrode_figdat,2)
    figure;
    
    % Plot treament vul
    vuldat=electrode_figdat{1,i};
    subplot(1,2,1)   
    imagesc(vuldat);
    set(gca,'XTick',1:numel(master_electrode_labels_grouped),'XTickLabel',master_electrode_labels_grouped,'XTickLabelRotation',90)
    set(gca,'YTick',1:numel(master_electrode_labels_grouped),'YTickLabel',master_electrode_labels_grouped)
    title('Treatment Vul');
    cb=colorbar;
    ylabel(cb,'# of electrode pairs')
    colormap('jet')
    
    % Plot treament resist
    resdat=electrode_figdat{2,i};
    subplot(1,2,2)   
    imagesc(resdat);
    set(gca,'XTick',1:numel(master_electrode_labels_grouped),'XTickLabel',master_electrode_labels_grouped,'XTickLabelRotation',90)
    set(gca,'YTick',1:numel(master_electrode_labels_grouped),'YTickLabel',master_electrode_labels_grouped)
    title('Treatment resist');
    cb=colorbar;
    ylabel(cb,'# of electrode pairs')
    colormap('jet')
    
    % calculate jac coef between two groups
    jc=jaccard((vuldat>0),(resdat>0));
    
    sgtitle([phases{i},'-ncount2-(',num2str(jc),')']);
end
    
%% Prep data for lin reg (extract jac index for each subject

jacdat=jac_coeff.beta.p66;
fn=fieldnames(jacdat);
regdat=[];
for s=1:numel(fn)
    sbjnum=extractAfter(fn{s},'_');
    sbjdat=vertcat(jacdat.(fn{s}).fa{:});
    jac=[ones(size(sbjdat,1),1)*str2num(sbjnum) sbjdat(:,3)];
    regdat=vertcat(regdat,earlyjac);
end
