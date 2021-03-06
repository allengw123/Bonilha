%ENSURE WORKSPACE IS CLEAR OF CONFLICTING VARIABLES
clear
clc


%Start local parpool
parpool('local');

%Assigning main folder: will look through all subfolders for matching files
datafolder={'F:\Patient_Data\ADNI_PD_nifti','F:\Patient_Data\ADNI_CN_nifti'};

%Reassiging common arbitrary variables
n = parcluster('local');
nworkers = n.NumWorkers;

%Constructing Matlabbatch for Mapping normalized volumes to template
%surface. Note, to ensure rows are equal (to prevent spm_jobman error),
%emtpy cells are populated with previous entry in row. Thus, the resulting
%jobman output will overwrite the previous output with identical data.


for d=1:numel(datafolder)
    disp(['Running...',datafolder{d}])
    tic
    tempdat=datafolder{d};
    subjects=dir(fullfile(tempdat,'ADNI*'));
    
    i = 1;
    j = 1;
    k = 1;
    while i <= length(subjects)
        for j = 1:nworkers
           if i <= length(subjects)
            matlabbatch{j,k}.spm.tools.cat.stools.vol2surftemp.data_vol = {fullfile(subjects(i).folder,subjects(i).name,[subjects(i).name,'.nii'])};
            matlabbatch{j,k}.spm.tools.cat.estwrite.data = {fullfile(subjects(i).folder,subjects(i).name)};
            matlabbatch{j,k}.spm.tools.cat.estwrite.data_wmh = {''};
            matlabbatch{j,k}.spm.tools.cat.estwrite.nproc = 2;
            matlabbatch{j,k}.spm.tools.cat.estwrite.useprior = '';
            matlabbatch{j,k}.spm.tools.cat.estwrite.opts.tpm = {location_tpm};
            matlabbatch{j,k}.spm.tools.cat.estwrite.opts.affreg = 'mni';
            matlabbatch{j,k}.spm.tools.cat.estwrite.opts.biasacc = 0.5;
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.APP = 1070;
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.spm_kamap = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.LASstr = 0.5;
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.gcutstr = 2;
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.WMHC = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.registration.shooting.shootingtpm = {location_shooting_tpm};
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.registration.shooting.regstr = 0.5;
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.vox = 1.5;
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.restypes.optimal = [1 0.1];
            matlabbatch{j,k}.spm.tools.cat.estwrite.extopts.ignoreErrors = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.surface = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.surf_measures = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.ROImenu.atlases.neuromorphometrics = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.ROImenu.atlases.lpba40 = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.ROImenu.atlases.cobra = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.ROImenu.atlases.hammers = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.ROImenu.atlases.ownatlas = {''};
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.GM.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.GM.mod = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.GM.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.WM.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.WM.mod = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.WM.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.CSF.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.CSF.warped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.CSF.mod = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.CSF.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.ct.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.ct.warped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.ct.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.pp.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.pp.warped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.pp.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.WMH.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.WMH.warped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.WMH.mod = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.WMH.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.SL.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.SL.warped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.SL.mod = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.SL.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.TPMC.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.TPMC.warped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.TPMC.mod = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.TPMC.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.atlas.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.atlas.warped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.atlas.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.label.native = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.label.warped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.label.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.labelnative = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.bias.warped = 1;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.las.native = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.las.warped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.las.dartel = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.jacobianwarped = 0;
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.warps = [0 0];
            matlabbatch{j,k}.spm.tools.cat.estwrite.output.rmat = 0; 
            i = i + 1;
           elseif i > length(subjects)
               matlabbatch{j,k} = matlabbatch{j,k-1};
           end
        end
        k = k + 1;
    end

    spmd
        spm_jobman('run',matlabbatch(labindex,:));
    end
    toc
end

%toc %66.46 seconds