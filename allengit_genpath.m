function allengit_genpath(gitpath,type)

addpath(genpath(fullfile(gitpath,'Scripts')))

if strcmp(type,'imaging')
    addpath(genpath(fullfile(gitpath,'imaging')))
end
end