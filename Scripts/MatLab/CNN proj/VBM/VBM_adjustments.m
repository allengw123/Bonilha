%% VBM nifti adjustments
clear
clc

% githubpath='C:\Users\allen\Documents\GitHub\Bonilha';
githubpath='C:\Users\bonilha\Documents\GitHub\Bonilha';

cd(githubpath)
allengit_genpath(githubpath,'imaging')


%% Load nifti
datapath='F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\Ip_VBM';
nifti={dir(fullfile(datapath,'*vs*.nii')).name};
cd(datapath)

for n=1:numel(nifti)

    P{1} = fullfile(datapath,nifti{n});
    P{2} = fullfile(datapath,'grey.nii');

    flags.interp = 0;
    spm_reslice(P,flags);

    %  get the name of the resliced map
    [a,b,c] = fileparts(P{2});
    M = fullfile(a, ['r' b c]);

    % load both
    MM = load_nii(M);
    G = load_nii(P{1});

    % get voxels with high probability of being gray matter (i.e., get rid of the voxels in the rim of the brain)

    gray = G.img>0.82;

    % mask the map
    map = MM.img.*gray;

    % Min-Max Noramlize each layer
    map=abs(map);
    for i=1:size(map,3)
        tempmap=map(:,:,i);
        if any(~isnan(tempmap),'all')
            tempmap(~isnan(tempmap))=mat2gray(tempmap(~isnan(tempmap)));
        else
            disp(['Layer ',num2str(i),' all nan'])
        end
        map(:,:,i)=tempmap;
    end
    
    % Obtain top 0.75
    map(map<0.75)=0;
    
    % save
    MM.img = map;
    
    save_nii(MM,fullfile(datapath,['TOP_',nifti{n}]))
end

%% Load Age Regression
datapath='F:\VBM ouput\Age_Regress\flip_ttest';
nifti={dir(fullfile(datapath,'*vs*.nii')).name};
cd(datapath)

for n=1:numel(nifti)

    P{1} = fullfile(datapath,'grey.nii');
    P{2} = fullfile(datapath,nifti{n});

    flags.interp = 0;
    spm_reslice(P,flags);

    %  get the name of the resliced map
    [a,b,c] = fileparts(P{2});
    M = fullfile(a, ['r' b c]);

    % load both
    MM = load_nii(M);
    G = load_nii(P{1});

    % get voxels with high probability of being gray matter (i.e., get rid of the voxels in the rim of the brain)

    gray = G.img>0.82;

    % mask the map
    map = MM.img.*gray;

    % Min-Max Noramlize each layer
    map=abs(map);
    for i=1:size(map,3)
        tempmap=map(:,:,i);
        if any(~isnan(tempmap),'all')
            tempmap(~isnan(tempmap))=mat2gray(tempmap(~isnan(tempmap)));
        else
            disp(['Layer ',num2str(i),' all nan'])
        end
        map(:,:,i)=tempmap;
    end
    
    % Obtain top 0.75
    map(map<0.75)=0;
    
    % save
    MM.img = map;
    
    save_nii(MM,fullfile(datapath,['TOP_',nifti{n}]))
end


%% Create FULL matrix of means of VBM
dataPath='F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\Ip_VBM';
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
    
    colormap(flipud(gray))
    colorbar

end