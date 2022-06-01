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
datapath='F:\VBM ouput\Age_Regress\ttest';
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
