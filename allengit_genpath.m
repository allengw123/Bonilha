function allengit_genpath(gitpath,type)

addpath(genpath(fullfile(gitpath,'Scripts')))

if strcmp(type,'imaging')
    addpath(genpath(fullfile(gitpath,'Toolbox','imaging')))
    
    % Don't add all spm subfloders
    spm_rmpath
    addpath(fullfile(gitpath,'Toolbox','imaging','spm12'))
    spm
    close all
    clc
end
end