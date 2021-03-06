clear all
clc

% gitPath='C:\Users\allen\Documents\GitHub\Bonilha';
gitPath = 'C:\Users\bonilha\Documents\GitHub\Bonilha';

cd(gitPath)
allengit_genpath(gitPath,'imaging')

% dataPath='C:\Users\allen\Box Sync\Desktop\Bonilha\Projects\ep_imaging_AI\CNN output\featureWeights';
dataPath = 'F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\Ip_RELU';
cd(dataPath)
files={dir(fullfile(dataPath,'*.nii')).name};
example=load_nii('EXAMPLE.nii');

%% Create feature weight niftis
for i=1:numel(files)
    if strcmp(files{i},'Example.nii')
        continue
    end
    
    nifti=load_nii(files{i});
    temp=example;

    % Resize
    if any(size(nifti.img) ~= size(temp.img))
        nifti.img = imresize3(nifti.img, [113 137 113]);
    end
    
    % Normalize each layer
    for l=1:size(nifti.img,3)
        nifti.img(:,:,l)=mat2gray(nifti.img(:,:,l));
    end
    
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

%% Create Histogram of means of CNN
dataPath='F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\RELU';
aal_regions=readtable('F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\aal.xlsx');
xmlFiles={dir(fullfile(dataPath,'*.xlsx')).name};
TLE_Regions_name = {'Angular_R','Parietal_Inf_R','Angular_L','Temporal_Inf_L','Amygdala_L','Temporal_Inf_R','Occipital_Mid_R','Occipital_Inf_L','Temporal_Mid_R','Fusiform_L','Thalamus_R','Occipital_Mid_L','Occipital_Inf_R','Thalamus_L','Frontal_Mid_R','Hippocampus_L','Fusiform_R','ParaHippocampal_L','Amygdala_R','Frontal_Mid_L'};
TLE_Regions=sort(cellfun(@(x) find(strcmp(x,aal_regions.Structure)),TLE_Regions_name));
vectdat=[];

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
    
%     figure('Units','normalized','Position',[0 0 .33 1],'Name',extractBefore(xmlFiles{d},'.xlsx'));
    figure('Name',extractBefore(xmlFiles{d},'.xlsx'));
    set(gcf,'color','w');
    bar(bardata(TLE_Regions))
    set(gca,'box','off')
    xticks(1:20)
    ylim([0 1])
    xticklabels(TLE_Regions);
    
    hold on
    y=bardata(TLE_Regions);
    x=1:20;
    lengthX=length(x);
    
    samplingRateIncrease = 10;
    newXSamplePoints = linspace(min(x), max(x), lengthX * samplingRateIncrease);
    smoothedY = spline(x, y, newXSamplePoints);

    ySmooth = newXSamplePoints;
    xSmooth = smoothedY;

    plot(newXSamplePoints, smoothedY);
    
    xlabel('AAL Atlas Regions')
    ylabel('Mean Activation')
    
    vectdat=[vectdat y];
    
    m=mean(bardata);
    s=std(bardata);
    
    disp([extractBefore(xmlFiles{d},'.xlsx'),' m=',num2str(m),' s=',num2str(s)]) 
    
end

comp=nchoosek(xmlFiles,2);
figure
for c=1:size(comp,1)
    
    v1=vectdat(:,strcmp(comp{c,1},xmlFiles));
    v2=vectdat(:,strcmp(comp{c,2},xmlFiles));
    
    n1=mat2gray(v1);
    n2=mat2gray(v2);
    D  = norm(n1 - n2);
    
    nexttile
    plot(1:numel(n1),n1,'r-',1:numel(n2),n2,'b-')
    title([extractBefore(comp{c,1},'.xlsx'),'(R) vs ',extractBefore(comp{c,2},'.xlsx'),'(B) (',num2str(D),')'])

    
    xlabel('AAL atlas regions')
    ylabel('Normalized Activations')
    xticks(1:20)
    xticklabels(TLE_Regions)
end
sgtitle('CNN features')

%% Create Histogram of TOP means of VBM
dataPath='C:\Users\allen\Box Sync\Desktop\Bonilha\Projects\ep_imaging_AI\VBM ouput\Corrected-ttest';
aal_regions=readtable(fullfile(dataPath,'aal','aal regions.xlsx'));
xmlFiles={dir(fullfile(dataPath,'*.xlsx')).name};
TLE_Regions=sort([35;41;33;42;74;72;67;34;25;36;40;31;71;78;73;37;38;21;68;22]);

vectdat=[];
for d=1:numel(xmlFiles)
    tempxml=readtable(fullfile(dataPath,xmlFiles{d}));
    
    bardata=[];
    for r=1:size(aal_regions,1)
        regionIdx=find(strcmp(aal_regions.Structure{r},tempxml.Structure));
        if isempty(regionIdx)
            bardata=[bardata;0];
            disp([aal_regions.Structure{r},' NOT FOUND'])
        else
            bardata=[bardata;mean(tempxml.Mean(regionIdx))];
        end
        if sum(strcmp(aal_regions.Structure{r},tempxml.Structure))>1
            disp(aal_regions.Structure{r})
        end
    end
    
   
    figure('Name',extractBefore(xmlFiles{d},'.xlsx'));
    set(gcf,'color','w');
    bar(bardata(TLE_Regions))
    set(gca,'box','off')
    xticks(1:20)
    ylim([0 0.4])
    xticklabels(TLE_Regions);
    
    hold on
    y=bardata(TLE_Regions);
    x=1:20;
    lengthX=length(x);
    
    samplingRateIncrease = 10;
    newXSamplePoints = linspace(min(x), max(x), lengthX * samplingRateIncrease);
    smoothedY = spline(x, y, newXSamplePoints);

    ySmooth = newXSamplePoints;
    xSmooth = smoothedY;

    plot(newXSamplePoints, smoothedY);
    
    xlabel('AAL Atlas Regions')
    ylabel('Mean Activation')
    
    vectdat=[vectdat y];
    
    m=mean(bardata);
    s=std(bardata);
    
    disp([extractBefore(xmlFiles{d},'.xlsx'),' m=',num2str(m),' s=',num2str(s)]) 
end


comp=nchoosek(xmlFiles,2);
figure
for c=1:size(comp,1)
    
    v1=vectdat(:,strcmp(comp{c,1},xmlFiles));
    v2=vectdat(:,strcmp(comp{c,2},xmlFiles));
    
    n1=mat2gray(v1);
    n2=mat2gray(v2);
    D  = norm(n1 - n2);
    
    nexttile
    plot(1:numel(n1),n1,'r-',1:numel(n2),n2,'b-')
    title([extractBefore(comp{c,1},'.xlsx'),'(R) vs ',extractBefore(comp{c,2},'.xlsx'),'(B) (',num2str(D),')'])
    
    xlabel('AAL atlas regions')
    ylabel('Normalized Activations')
    xticks(1:20)
    xticklabels(TLE_Regions)
end
sgtitle('VBM comparisons')


%% Create FULL Histogram of means of CNN
dataPath='F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\RELU';
aal_regions=readtable('F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\aal.xlsx');
xmlFiles={dir(fullfile(dataPath,'*.xlsx')).name};
cd(dataPath)
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
    
%     figure('Units','normalized','Position',[0 0 .33 1],'Name',extractBefore(xmlFiles{d},'.xlsx'));
    figure('Name',extractBefore(xmlFiles{d},'.xlsx'));
    set(gcf,'color','w');
    barh(bardata)
    yticks(1:10:117);  

    set(gca, 'YDir','reverse','box','off')
    xlabel('Activation')
    ylabel('AAL Region Index')
    
    pbaspect([1 5 1])

end
%%
dataPath='F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\Ip_RELU';
aal_regions=readtable('F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\aal.xlsx');
xmlFiles={dir(fullfile(dataPath,'*.xlsx')).name};
cd(dataPath)
for d=1:numel(xmlFiles)
    tempxml=readtable(fullfile(dataPath,xmlFiles{d}));
    matrix = nan(10,12);
    
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
    
%     figure('Units','normalized','Position',[0 0 .33 1],'Name',extractBefore(xmlFiles{d},'.xlsx'));
    figure('Name',extractBefore(xmlFiles{d},'.xlsx'));
    set(gcf,'color','w');
    
    for i = 1:numel(bardata)
        matrix(i+1) = bardata(i);
    end
    matrix = matrix';
    imagesc(matrix)
    yticks([1:12])
    yticklabels([0:11])

    xticks([1:10])
    xticklabels([0:9])
    
    map =interp1([0;1],[1 1 1; 1 0 0],linspace(0,1,256))
    colormap(map)
    colorbar
end