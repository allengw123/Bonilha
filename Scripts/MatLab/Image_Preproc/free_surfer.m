clear all
close all


GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Setup Freesurfer

%% Run 
subject_dir = '/media/bonilha/Elements/freesurfer/BONPL0141';
nifti = fullfile(subject_dir,'T1_BONPL0141.nii');
command = ['segment_subregions thalamus ',nifti];
system(command,"-echo")

%% Functions
function freesurfer_setenv
FREESURFER_HOME = '/usr/local/freesurfer/7.3.2';
FSFAST_HOME = fullfile(freesurfer_home,'fsfast');
if ~exist(freesurfer_home,'dir')
    error(['freesurfer is not found in ',freesurfer_home,' please change function'])
end

setenv('FREESURFER_HOME',FREESURFER_HOME);
setenv('FSFAST_HOME',FSFAST_HOME)
fshome = getenv('FREESURFER_HOME');
fsmatlab = sprintf('%s/matlab',fshome);
if (exist(fsmatlab) == 7)
path(path,fsmatlab);
end
fsfasthome = getenv('FSFAST_HOME');
fsfasttoolbox = sprintf('%s/toolbox',fsfasthome);
if (exist(fsfasttoolbox) == 7)
path(path,fsfasttoolbox);
end
end
