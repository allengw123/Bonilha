clear
clc

% githubpath='C:\Users\allen\Documents\GitHub\Bonilha';
githubpath='C:\Users\bonilha\Documents\GitHub\Bonilha';

cd(githubpath)
allengit_genpath(githubpath,'imaging')
%%
comps = {'TLE','AD','Control'};

relu_folder = 'F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\Ip_RELU';
occlusion_folder = 'F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\Ip_Occlusion';

for c = 1:numel(comps)

    r_nii = load_nii(fullfile(relu_folder,['TOP_',comps{c},'_Act.nii']));
    o_nii = load_nii(fullfile(occlusion_folder,['TOP_',comps{c},'_Occlusion_Act.nii']));

    overlap = r_nii.img>0 & o_nii.img>0;

    output_nii = r_nii;
    output_nii.img = overlap;
    save_nii(output_nii,[comps{c},'_overlap.nii'])
end

%%

CNN = load_nii('F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\OccVRelu\TLE_overlap.nii');
VBM = load_nii('F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\VBM\TOP_Control vs TLE_T_bonf.nii');


overlap = CNN.img>0 & VBM.img>0;

output_nii = CNN;
output_nii.img = overlap;
save_nii(output_nii,'VBM_CNN_overlap.nii')c=1

%%
dataPath='F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\Ip_OccVRelu';
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
    
    map =interp1([0;1],[1 1 1; 0 1 0],linspace(0,1,256))
    colormap(map)
    colorbar
end