gitPath='C:\Users\allen\Documents\GitHub\Bonilha';

cd(gitPath)
allengit_genpath(gitPath,'imaging')

dataPath='C:\Users\allen\Box Sync\Desktop\Bonilha\Projects\ep_imaging_AI\CNN output\featureWeights';
cd(dataPath)
files={dir(fullfile(dataPath,'*.nii')).name};
example=load_nii('EXAMPLE.nii');

%% Create feature weight niftis
for i=1:numel(files)
    if strcmp(files{i},'EXAMPLE.nii')
        continue
    end
    
    nifti=load_nii(files{i});
    temp=example;
    
    % Normalize each layer
    for l=1:size(nifti.img,3)
        nifti.img(:,:,l)=mat2gray(nifti.img(:,:,l));
    end
    
    % Add row dimension
    row=mean(nifti.img(56:57,:,:),1);
    nifti.img=cat(1,nifti.img(1:56,:,:),row,nifti.img(57:end,:,:));
    
    % Add col dimension
    col=mean(nifti.img(:,68:69,:),2);
    nifti.img=single(cat(2,nifti.img(:,1:68,:),col,nifti.img(:,69:end,:)));
    
    % Save full brain
    temp.img=nifti.img;
    save_nii(temp,['FULL_',files{i}]);

    % Get Top 0.75
    if isempty(regexp(files{i},'.*STD.nii','match'))
        temp.img(~(temp.img>0.75))=0;
%         temp.img((temp.img>0.75))=mat2gray(temp.img((temp.img>0.75)));
    else
        toptemp=load_nii(['TOP_',extractBefore(files{i},'_STD.nii'),'_Act.nii']);
        temp.img(~logical(toptemp.img))=0;
%         temp.img(logical(toptemp.img))=mat2gray(temp.img(logical(toptemp.img)));
    end
    
    % Save TOP brain
    save_nii(temp,['TOP_',files{i}]);
end

%% Create Histogram of TOP means
aal_regions=readtable(fullfile(dataPath,'aal','aal regions.xlsx'));
xmlFiles={dir(fullfile(dataPath,'*.xlsx')).name};

for d=1:numel(xmlFiles)
    tempxml=readtable(fullfile(dataPath,xmlFiles{d}));
    
    bardata=[];
    for r=1:size(aal_regions,1)
        ROI_idx=find(strcmp(aal_regions.Structure{r},tempxml.Structure));
        if isempty(ROI_idx)
            bardata=[bardata;0];
            disp([aal_regions.Structure{r},' NOT FOUND'])
        else
            bardata=[bardata;mean(tempxml.Mean(ROI_idx))];
        end
        if sum(strcmp(aal_regions.Structure{r},tempxml.Structure))>1
            disp(aal_regions.Structure{r})
        end
    end
    
    figure('Units','normalized','Position',[0 0 .33 1],'Name',extractBefore(xmlFiles{d},'.xlsx'));
    set(gcf,'color','w');
    barh(bardata)
    set(gca,'box','off')
    yticks(10:10:numel(bardata))
    ylim([0 numel(bardata)+1])
    xlim([0 0.5])
    
    [sortDat,idx]=sort(bardata,'descend');
    
    % Identify top 5 relationship
    for i=1:5
       text(sortDat(i)+0.02,idx(i),aal_regions.Structure{aal_regions.Index==idx(i)},'Interpreter','none')
       disp(aal_regions.Structure{aal_regions.Index==idx(i)})
    end
end

%% Create Histogram of TOP means (normalized)
aal_regions=readtable(fullfile(dataPath,'aal','aal regions.xlsx'));
xmlFiles={dir(fullfile(dataPath,'*.xlsx')).name};

for d=1:numel(xmlFiles)
    tempxml=readtable(fullfile(dataPath,xmlFiles{d}));
    
    bardata=[];
    for r=1:size(aal_regions,1)
        regionIdx=find(strcmp(aal_regions.Structure{r},tempxml.Structure));
        if isempty(regionIdx)
            bardata=[bardata;0];
            disp([aal_regions.Structure{r},' NOT FOUND'])
        else
            bardata=[bardata;mean(tempxml.Mean(regionIdx))/mean(tempxml.Volume(regionIdx))];
        end
        if sum(strcmp(aal_regions.Structure{r},tempxml.Structure))>1
            disp(aal_regions.Structure{r})
        end
    end
    
    figure('Units','normalized','Position',[0 0 .33 1],'Name',[extractBefore(xmlFiles{d},'.xlsx'),' Normalized']);
    set(gcf,'color','w');
    barh(bardata)
    set(gca,'box','off')
    yticks(10:10:numel(bardata))
    ylim([0 numel(bardata)+1])
%     xlim([0 0.5])
    
    [sortDat,idx]=sort(bardata,'descend');
    
    % Identify top 5 relationship
    for i=1:5
       text(sortDat(i),idx(i),aal_regions.Structure{aal_regions.Index==idx(i)},'Interpreter','none')
    end
end
title('TLE TOP activation regions NORMALIZED')
