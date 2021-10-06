clc
close all
clear all

Bonilha_start
%% Define variables
wkfolder='C:\Users\allen\Box Sync\Allen_Bonilha_EEG\PatientData';
analysisfolder=fullfile(wkfolder,'Analysis');
structuralfolder=fullfile(analysisfolder,'Structural');
functionalfolder=fullfile(analysisfolder,'Functional');

mkdir(structuralfolder);

patientnumbers=dir(fullfile(wkfolder,'Patient *'));
patientnumbers={patientnumbers.name};


master_electrode={'LA','LAH','LAI','LLF','LMF','LPH','LPI','RA','RAH','RAI','RLF','RMF','RPH','RPI'};

master_labels=[];
for i=1:numel(master_electrode);
    temp=master_electrode{i};
    templabel={[temp,'_D'],[temp,'_M'],[temp,'_S']};
    master_labels=[master_labels,templabel];
end

%% Connectivity Measures

% Count
networkmeasures.count=nan(numel(master_labels),numel(master_labels),numel(patientnumbers));
for pat=1:numel(patientnumbers)
    try
        patientfold=fullfile(wkfolder,patientnumbers{pat},'structural','Tractography','Connectivity');
        
        % Load Count/labels data
        count=load(fullfile(patientfold,'count.mat'));
        labels=textscan(char(count.name),'%s');
        labels=labels{1,1}(1:end-1);
        count=count.connectivity;
        
        % Load nCount data
        ncount=load(fullfile(patientfold,'ncount.mat'));
        ncount=ncount.connectivity;
        
        % Load nCount2 data
        ncount2=load(fullfile(patientfold,'ncount2.mat'));
        ncount2=ncount2.connectivity;
        
        % Load Mean Length data
        mean_length=load(fullfile(patientfold,'mean_length.mat'));
        mean_length=mean_length.connectivity;
        
        % Load FA data
        fa=load(fullfile(patientfold,'fa.mat'));
        fa=fa.connectivity;
        
        % Find label idx
        labelidx=[];
        for i=1:numel(labels)
            labelidx=[labelidx,find(strcmp(labels{i},master_labels))];
        end

        % Organize info
        for row=1:size(count,1)
            for col=1:size(count,2)
                networkmeasures.count(labelidx(row),labelidx(col),pat)=count(row,col);
                networkmeasures.ncount(labelidx(row),labelidx(col),pat)=ncount(row,col);
                networkmeasures.ncount2(labelidx(row),labelidx(col),pat)=ncount2(row,col);
                networkmeasures.mean_length(labelidx(row),labelidx(col),pat)=mean_length(row,col);
                networkmeasures.fa(labelidx(row),labelidx(col),pat)=fa(row,col);
            end
        end
        
        % Save Electrode list
        save(fullfile(wkfolder,patientnumbers{pat},'structural','Tractography','ElectrodeList'),'label');
    catch
        disp([patientnumbers{pat},' network measures not found'])
    end
end

save(fullfile(structuralfolder,'struct_conn'),'networkmeasures');
%% Compare Structural and Function connectivity

% Load functional data
functionalconn=load(fullfile(functionalfolder,'tot_con.mat'));
functionalconn=functionalconn.alphaconnectivity;

% Organize functional data
functionalconn_reorganized=[];
for phase=1:8
    for sbj=1:numel(patientnumbers)
        functionalidx=find(functionalconn.count.sbj==sbj & functionalconn.count.ident==phase);
        functionalconn_reorganized(:,:,sbj,phase)=mean(functionalconn.raw(:,:,functionalidx),3);
    end
end

% Compare Struct vs Functional

funcvars={'count','ncount','ncount2','mean_length','fa'};

structfunc_comp=[];
for fv=1:numel(funcvars)
    tempfv=funcvars{fv};
    for phase=1:8
        tempfunctional=functionalconn_reorganized(:,:,:,phase);
        tempcount=networkmeasures.(tempfv);
        for rows=1:size(tempfunctional,1)
            for cols=1:size(tempfunctional,2)
                tf=permute(tempfunctional(rows,cols,:),[3 2 1]);
                tc=permute(tempcount(rows,cols,:),[3 2 1]);
                tempdat=[tf tc];
                tempdat=tempdat(~any(isnan(tempdat),2),:);
                [r,p]=corrcoef(tempdat);

                if numel(r)==1
                    structfunc_comp.(tempfv).r(rows,cols,phase)=r(1,1);
                    structfunc_comp.(tempfv).p(rows,cols,phase)=p(1,1);
                    structfunc_comp.(tempfv).n(rows,cols,phase)=size(tempdat,1);
                else
                    structfunc_comp.(tempfv).r(rows,cols,phase)=r(2,1);
                    structfunc_comp.(tempfv).p(rows,cols,phase)=p(2,1);
                    structfunc_comp.(tempfv).n(rows,cols,phase)=size(tempdat,1);
                end
            end
        end
        fdr=structfunc_comp.(tempfv).p(:,:,phase);
        pvalidx=find(~isnan(fdr));
        [~, ~, ~, adj_p]=fdr_bh(fdr(pvalidx));
        fdr(pvalidx)=adj_p;
        structfunc_comp.(tempfv).fdrp(:,:,phase)=fdr;
    end
end
%% Create graphs

figure('Name','Number of Subjects')
imagesc(structfunc_comp.ncount.n(:,:,1))
xticks(1:numel(master_labels))
set(gca,'XTickLabel',master_labels,'fontsize',10,'XTickLabelRotation',90,'YTickLabel',master_labels)
yticks(1:numel(master_labels))
c=colorbar;
ylabel(c,'Number of Subjects','fontsize',8);
c.FontSize = 12;
colormap jet;
hold on
for i=1:size(structfunc_comp.ncount.n,2)
    for z=1:size(structfunc_comp.ncount.n,1)
        if isnan(structfunc_comp.ncount.n(i,z))
            text(z,i,'0','HorizontalAlignment', 'Center')
        else
            text(z,i,num2str(round(structfunc_comp.ncount.n(i,z))),'HorizontalAlignment', 'Center')
        end
    end
end
title('Number of Subjects')


funcvars={'count','ncount','ncount2','mean_length','fa'};

for fv=1:numel(funcvars)
    tempfv=funcvars{fv};
    figure
    for i=1:8
        subplot(3,3,i)
        tempdat=structfunc_comp.(tempfv).fdrp(:,:,i);
        tempdat(tempdat>0.05)=1;
        tempdat(isnan(tempdat))=1;
        imagesc(tempdat);
        cb=colorbar;
        caxis([0 0.10])
        ylabel(cb,'FDR Corr pvalue','fontsize',8);
        cb.FontSize = 12;
        colormap(flipud(jet));
        xticks(1:numel(master_labels))
        yticks(1:numel(master_labels))
        set(gca,'XTickLabel',master_labels,'fontsize',6,'XTickLabelRotation',90,'YTickLabel',master_labels)
        sgtitle(tempfv);
        switch i
            case 1
                title('Baseline')
            case 2
                title('Pre-Baseline')
            case 3
                title('Pre-Trans')
            case 4
                title('Post-Trans')
            case 5
                title('Mid-Seiz')
            case 6
                title('Late-Seiz')
            case 7
                title('Early-Post')
            case 8
                title('Late-Post')     
        end
    end
end

