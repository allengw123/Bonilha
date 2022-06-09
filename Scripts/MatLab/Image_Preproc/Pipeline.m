clear all
clc

% Git path
gitpath='C:\Users\allen\Google Drive\GitHub\Bonilha';

% Gen path
gitpath='C:\Users\allen\Google Drive\GitHub\Bonilha';
cd(gitpath)
allengit_genpath(gitpath,'imaging')


%%%% DON'T CHANGE (unless paths are incorrect; paths should be correct if
%%%% cloning git)

% SPM
spm_path=fullfile(gitpath,'Toolbox','imaging','spm12');

% DSI studio
dsipath=fullfile(gitpath,'Toolbox','imaging','dsi_studio_64');

% Brain age
brainage_path = fullfile(gitpath,'Toolbox','imaging','brainage');

%%
subjectdir =  'C:\Users\allen\Desktop\pipelinetest\PIT_PL_0012';
[~,subjectname] = fileparts(subjectdir);
cd(subjectdir)

rawdir = fullfile(subjectdir,'Raw');

%% T1 (spm/cat12)

disp('Step 1 - T1 image preprocessing')

% Create T1 folder
t1_folder = fullfile(subjectdir,'T1');
mkdir(t1_folder)

% Detect T1 acquisitions
t1_acq = dir(fullfile(rawdir,'*T1*'));
disp([num2str(numel(t1_acq)),' ... T1 Acquisitions Detected'])

% Copy T1 acquisitions
for a = 1:numel(t1_acq)
    
    % Copy Raw files over
    current_aq_name = char(extractBetween(t1_acq(a).name,[subjectname,'_'],'.nii'));
    aq_folder = fullfile(t1_folder,current_aq_name);
    mkdir(aq_folder)
    disp(['... Copying ',current_aq_name])
    niftifile = fullfile(t1_acq(a).folder,t1_acq(a).name);
    copyfile(niftifile,aq_folder)
    
    % Setup spm_jobman
    matlabbatch{a}.spm.tools.cat.estwrite.data = {niftifile};
    matlabbatch{a}.spm.tools.cat.estwrite.data_wmh = {''};
    matlabbatch{a}.spm.tools.cat.estwrite.nproc = 2;
    matlabbatch{a}.spm.tools.cat.estwrite.useprior = '';
    matlabbatch{a}.spm.tools.cat.estwrite.opts.tpm = {fullfile(spm_path,'tpm','TPM.nii')};
    matlabbatch{a}.spm.tools.cat.estwrite.opts.affreg = 'mni';
    matlabbatch{a}.spm.tools.cat.estwrite.opts.biasacc = 0.5;
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.APP = 1070;
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.spm_kamap = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.LASstr = 0.5;
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.gcutstr = 2;
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.WMHC = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.registration.shooting.shootingtpm = {fullfile(spm_path,'toolbox','cat12','templates_MNI152NLin2009cAsym','Template_0_GS.nii')};
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.registration.shooting.regstr = 0.5;
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.vox = 1.5;
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.restypes.optimal = [1 0.1];
    matlabbatch{a}.spm.tools.cat.estwrite.extopts.ignoreErrors = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.surface = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.surf_measures = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.ROImenu.atlases.neuromorphometrics = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.ROImenu.atlases.lpba40 = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.ROImenu.atlases.cobra = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.ROImenu.atlases.hammers = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.ROImenu.atlases.ownatlas = {''};
    matlabbatch{a}.spm.tools.cat.estwrite.output.GM.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.GM.mod = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.GM.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.WM.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.WM.mod = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.WM.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.CSF.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.CSF.warped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.CSF.mod = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.CSF.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.ct.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.ct.warped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.ct.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.pp.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.pp.warped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.pp.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.WMH.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.WMH.warped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.WMH.mod = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.WMH.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.SL.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.SL.warped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.SL.mod = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.SL.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.TPMC.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.TPMC.warped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.TPMC.mod = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.TPMC.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.atlas.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.atlas.warped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.atlas.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.label.native = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.label.warped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.label.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.labelnative = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.bias.warped = 1;
    matlabbatch{a}.spm.tools.cat.estwrite.output.las.native = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.las.warped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.las.dartel = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.jacobianwarped = 0;
    matlabbatch{a}.spm.tools.cat.estwrite.output.warps = [0 0];
    matlabbatch{a}.spm.tools.cat.estwrite.output.rmat = 0; 
end

disp('Running SPM/Cat12')
spm_jobman('run',matlabbatch);

%% Diffusion (DSI studios)
disp('Step 2 - Running Diffusion Tractograhpy')

% Create DKI folder
dki_folder = fullfile(subjectdir,'Diffusion');
mkdir(dki_folder)

% Find diffusion acquisitions
diffusion_aq = dir(fullfile(rawdir,'*.bval'));
disp([num2str(numel(diffusion_aq)),' ... Diffusion Acquisitions Detected'])

% Tractography program
for a = 1:numel(diffusion_aq)
    
    % Copy Raw files over
    current_aq_name = char(extractBetween(diffusion_aq(a).name,[subjectname,'_'],'.bval'));
    disp(['... Copying ',current_aq_name])
    current_aq_files = dir(fullfile(rawdir,['*',current_aq_name,'.*']));
    
    % Check to see if all 4 files are present
    if numel(current_aq_files) ~= 4
        disp([current_aq_name,' aquisition MISSING FILES PLEASE CHECK... NOT COPYING'])
        continue
    end
    
    % Copy Diffusion Acquistion files
    aq_folder = fullfile(dki_folder,current_aq_name);
    mkdir(aq_folder)
    for i = 1:numel(current_aq_files)
        copyfile(fullfile(current_aq_files(i).folder,current_aq_files(i).name),aq_folder)
    end
    
    % Run DSI studios
    cd(dsipath)
    disp(['...Running DSI studios'])
    
    % Nifti --> SRC
    disp('......creating src file')
    dsi_output_folder=fullfile(aq_folder,'DSI_ouput');
    mkdir(dsi_output_folder)
    nifti_file = fullfile(aq_folder,dir(fullfile(aq_folder,'*.nii')).name);
    srcfilename=fullfile(dsi_output_folder,current_aq_name);
    cmd=sprintf('call dsi_studio --action=src --source=%s --output=%s.src.gz > %s.txt',nifti_file,srcfilename,fullfile(dsi_output_folder,'niftisrc_conversion'));
    system(cmd);
    
    % SRC quality chcek
    disp(['......src quality check'])
    cmd=sprintf('call dsi_studio --action=qc --source=%s.src.gz',srcfilename);
    system(cmd);

    % SRC --> fib file
    disp('......creating fib file')
    cmd=sprintf('call dsi_studio --action=rec --source=%s.src.gz --method=4  --param0=1.25',srcfilename);
    system(cmd);

    % Fiber tracking
    fib_file = fullfile(dsi_output_folder,dir(fullfile(dsi_output_folder,'*fib*')).name);
    disp('......fiber tracking')
    cmd = sprintf('call dsi_studio --action=trk --source=%s --seed_count=1000000 --thread_count=8 --output=no_file --connectivity=HCP842_tractography > %s.log.txt',fib_file,fullfile(dsi_output_folder,'tractography'));
    system(cmd);
end

%% Brain Age
disp('Step 3 - Running Brain Age')
cd(fullfile(brainage_path,'software'))

% Prep Brain Age Files
setup_brainagedir(brainage_path,spm_path)

% Make brain age folder
brain_age_folder = fullfile(subjectdir,'BrainAge');
mkdir(brain_age_folder)

% Detect T1 acquisitions
t1_acq = dir(fullfile(rawdir,'*T1*'));
disp([num2str(numel(t1_acq)),' ... T1 Acquisitions Detected'])

% Copy T1 acquisitions
for a = 1:numel(t1_acq)
    
    % Copy Raw files over
    current_aq_name = char(extractBetween(t1_acq(a).name,[subjectname,'_'],'.nii'));
    aq_folder = fullfile(brain_age_folder,current_aq_name);
    mkdir(aq_folder)
    niftifile = fullfile(t1_acq(a).folder,t1_acq(a).name);
    copyfile(niftifile,aq_folder)

    input = niftifile;
    output = fullfile(aq_folder,[subjectname,'_T1_brain_predicted.csv']);
    
    cmd = sprintf('brainageR.sh -f %s -o %s',input,output);
    system(cmd)
    
end

%% Functinon

function setup_brainagedir(brainage_path,spm_path)
    
% Detect brainage SH file
brainage_file_path = fullfile(brainage_path,'software','brainageR');

% Read brainage SH file
brainage_file = fopen(brainage_file_path,'r');
eof = false;
fileContents = [];
while ~eof
    str = fgetl(brainage_file);
    if str==-1
        eof = true;
    else
        fileContents = [fileContents; {str}];
    end
end
fclose(brainage_file);

% Replace brainageR_dir path
idx = find(~cellfun(@isempty,(regexp(fileContents,'brainageR_dir'))));
brainageR_dir_line = sprintf('brainageR_dir=%s',brainage_path);
fileContents{idx(1)} = brainageR_dir_line;

% Replace spm_dir path
idx = find(~cellfun(@isempty,(regexp(fileContents,'spm_dir'))));
spm_path_line = sprintf('spm_dir=%s',spm_path);
fileContents{idx(1)} = spm_path_line;

% Replace matlab_path path
idx = find(~cellfun(@isempty,(regexp(fileContents,'matlab_path'))));
matlab_path_line = sprintf('matlab_path=%s',matlabroot);
fileContents{idx(1)} = matlab_path_line;

% write/replace brainageR sh file
brainage_file = fopen(brainage_file_path,'w');
fprintf(brainage_file,'%s\n',string(fileContents));
fclose(brainage_file);
end
