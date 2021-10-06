    savepath='C:\Users\bonilha\Documents\Project_Eleni\JHU atlas-selected';
    atlaspath='jhu.nii';


    atlas=load_nii(atlaspath);
    ROI_idx=double(unique(atlas.img));
    ROI_txt=importdata('jhu.txt');
    ROI_txt=ROI_txt.textdata;
    for ri=ROI_idx'
        tempfilewrite=atlas;
        tempfilewrite.fileprefix=ROI_txt{ri};
        tempfilewrite.img=atlas.img==ri;
        save_nii(tempfilewrite,fullfile(savepath,[ROI_txt{ri,1},'_',ROI_txt{ri,2},'.nii']));
    end
