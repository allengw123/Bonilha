function nii_preprocess_subfolders(opt)
%find all limegui.mat files in subfolders

% Turn off parallel workers
setpool(0);

% Define paths
pth = opt.paths.harvest_output;
outpath = opt.paths.processed_output;

% Check to see if paths exist
if ~exist('pth','var'), pth = pwd; end;
f = subFolderSub(pth);
if isempty(f), error('No folders in parent folder %s', pth); end

% Sync processed with harvest_output folder
if opt.syncPreprocessed 

    harvest_patients = dir(pth);
    harvest_patients(contains({harvest_patients.name},'.')) = [];
    processed_mat = dir(fullfile(outpath,'*','*.mat'));
    rm_idx = find(~cellfun(@(x) any(contains({harvest_patients.name},extractBefore(x,'.mat'))),{processed_mat.name}));
    if ~isempty(rm_idx)
        disp(['Removing ',num2str(numel(rm_idx)),' subject found in processed but not harvest'])
        for r = 1:numel(rm_idx)
            disp(['Removed ',processed_mat(rm_idx(r)).name])
            delete(fullfile(processed_mat(rm_idx(r)).folder,processed_mat(rm_idx(r)).name))
        end
    end
end


% Create Output Folder
mkdir(outpath)

% Copy matfiles from harvest output
vers = nii_matver;
vers = sprintf('%.4f', vers.lime);
outpth = fullfile(outpath, ['M.', vers]);
mkdir(outpth);
textprogressbar(0,'Extracting .mat files from harvest output');
for i = 1: numel(f) %change 1 to larger number to restart after failure
    cpth = char(deblank(f(i))); %local child path
    %fprintf('===\t%s participant %d/%d : %s\t===\n', mfilename, i, numel(f), cpth);
    %if ~isempty(strfind(cpth,'M2082')), error('all done'); end; %to stop at specific point
    if contains(cpth,'_')
        fprintf('Warning: "_" in folder name: skipping %s\n', char(cpth) );
        continue
    end
    %cpth = char(fullfile(pth,cpth)); %full child path
    inpth = fullfile(pth, cpth);
    mfile = dir(char(fullfile(inpth,'**','*lime.mat')));

    if isempty(mfile)
        fprintf('Warning: no .mat file found. skipping %s\n', char(cpth) );
        continue
    end

    outname = fullfile(outpth, [cpth, '.mat']);
    if exist(outname,'file') && ~opt.forcedPull
        continue
    end
    

    save_mfile = [];
    for m = 1:numel(mfile)
        load_mfile = load(fullfile(mfile(m).folder,mfile(m).name));
        [~,ses]=fileparts(mfile(m).folder);
        save_mfile.(ses) = load_mfile;
    end
    
    
    saveparfor(outname,'-struct','save_mfile','-v7.3',save_mfile)

    textprogressbar(1,i/numel(f)*100,['Created .mat for ',cpth])
end
textprogressbar(2,'done')

if opt.isMakeModalityTable
    nii_modality_table(outpth);
end
%end nii_preprocess_subfolders()

function nameFolds=subFolderSub(pathFolder)
d = dir(pathFolder);
isub = [d(:).isdir];
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];
%end subFolderSub()

function saveparfor(outname,opt1,opt2,opt3,save_mfile)
save(outname,opt1,opt2,opt3)
%end saveparfor()
