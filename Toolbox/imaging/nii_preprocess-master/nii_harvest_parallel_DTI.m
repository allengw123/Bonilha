function error_sbjs = nii_harvest_parallel_DTI (baseDir,outDir,opt,debug_sbj)

sync_with_formated = opt.sync_with_formated;
interweave = opt.interweave;

% if the correct environment variables are set in the environment, override
% the above
if ~isempty(getenv('nii_harvest_isExitAfterTable'))
    isExitAfterTable = strcmpi(getenv('nii_harvest_isExitAfterTable'),'true')
end
if ~isempty(getenv('nii_harvest_isPreprocess'))
    isPreprocess = strcmpi(getenv('nii_harvest_isPreprocess'),'true')
end
if ~isempty(getenv('nii_harvest_isReportDims'))
    isReportDims = strcmpi(getenv('nii_harvest_isReportDims'),'true')
end
if ~isempty(getenv('nii_harvest_reprocessRest'))
    reprocessRest = strcmpi(getenv('nii_harvest_reprocessRest'),'true')
end
if ~isempty(getenv('nii_harvest_reprocessfMRI'))
    reprocessfMRI = strcmpi(getenv('nii_harvest_reprocessfMRI'),'true')
end
if ~isempty(getenv('nii_harvest_reprocessASL'))
    reprocessASL = strcmpi(getenv('nii_harvest_reprocessASL'),'true')
end
if ~isempty(getenv('nii_harvest_reprocessDTI'))
    reprocessDTI = strcmpi(getenv('nii_harvest_reprocessDTI'),'true')
end
if ~isempty(getenv('nii_harvest_reprocessVBM'))
    reprocessVBM = strcmpi(getenv('nii_harvest_reprocessVBM'),'true')
end
if ~isempty(getenv('nii_harvest_explicitProcess'))
    explicitProcess = strcmpi(getenv('nii_harvest_explicitProcess'),'true')
end

if ~exist('baseDir','var') || isempty(baseDir)
    %baseDir = pwd; %
    baseDir = uigetdir('','Pick folder that contains all subjects');
end



%***Ignores directories containing '_' symbol
subjDirs = subFolderSub(baseDir);
subjDirs = sort(subjDirs);

if ~isempty(getenv('nii_harvest_subjDirs'))
    subjDirs = {getenv('nii_harvest_subjDirs')};
end


% Sync with format database
if sync_with_formated
    [input_sbj,~] = subFolderSub(baseDir);
    [output_sbj,output_dir] = subFolderSub(outDir);

    rm_sbjs = output_sbj(~cellfun(@(x) any(strcmp(x,input_sbj)),output_sbj));

    if ~isempty(rm_sbjs)
        disp([num2str(numel(rm_sbjs)),' Subjects detected in output folder that isnt in input folder'])
        for i = 1:numel(rm_sbjs)
            rmdir(fullfile(output_dir,rm_sbjs{i}),'s')
            disp(['Removed Subject ',rm_sbjs{i},' from output folder'])
        end
    end
end


if nargin<4

    numOfGPU = gpuDeviceCount("available");
    %set up parallel loop variables
    if ~isempty(gcp('nocreate'))
        pool = gcp('nocreate');
        if ~pool.NumWorkers == numOfGPU
            delete(pool)
            c = parcluster;
            c.NumWorkers = numOfGPU;
            pool = c.parpool(numOfGPU);
        end
    else
        c = parcluster;
        c.NumWorkers = numOfGPU;
        pool = c.parpool(numOfGPU);
    end
    

    numLoops = pool.NumWorkers;
    subjDirsSize = length(subjDirs);
    div = floor(subjDirsSize/numLoops);
    modRemainder = mod(subjDirsSize,numLoops);

    idex = cell(numLoops,1);
    for i = 1:numLoops
        idex{i} = i:numLoops:subjDirsSize;
    end

    parfor i = 1:numLoops
        worker_num = getCurrentTask();
        worker_num = worker_num.ID;
        disp(['Using GPU number ',num2str(worker_num)])
        if(numLoops <= subjDirsSize)
            %subjDirsX = subjDirs(1+(i-1)*div:1+i*div)
            start = ((i-1)*div)+1;
            fin = ((i-1)*div)+div;
            if i == numLoops
                fin = fin+modRemainder;
            end

            if interweave
                disp('INTERWEAVE OPTION ACTIVE')
                
                error_sbjs{i} =startParallelHarvest(subjDirs(idex{i}),baseDir,outDir,opt,worker_num)
                %error_sbjs{1} =startParallelHarvest({'EMOPL0018'},baseDir,outDir,setOrigin,isExitAfterTable,isPreprocess,isReportDims,reprocessfMRI,reprocessRest,reprocessASL,reprocessDTI,reprocessVBM);

            else
                fprintf("Process #%d uses subjDirs--> %d:%d \n", i, start, fin);

                error_sbjs{i} = startParallelHarvest(subjDirs(start:fin),baseDir,outDir,opt,worker_num)
            end
        else
            fprintf("Can't have more parallel loops than subjects to process...");
        end
    end
else
    startParallelHarvest(debug_sbj,baseDir,outDir,opt,3,true)
end

function error_sbjs = startParallelHarvest(subjDirs,baseDir,outDir,opt,gpu_idx,DEBUG)

setOrigin = opt.setOrigin ; %attempt to crop and set anterior commissure for images
isExitAfterTable = opt.isExitAfterTable; % <- if true, only generates table, does not process data
isPreprocess = opt.isPreprocess; % <- if true full processing, otherwise just cropping
isReportDims = opt.isReportDims; %if true, report dimensions of raw data
reprocessRest = opt.reprocessRest;
reprocessfMRI = opt.reprocessfMRI;
reprocessASL = opt.reprocessASL;
reprocessDTI = opt.reprocessDTI;
reprocessVBM = opt.reprocessVBM ;
explicitProcess = opt.explicitProcess; % <- if true, will only process if the reprocess flag is true

modalityKeysVerbose = {'Lesion','T1','T2','DTI_','DTIrev','ASL','ASLrev','Rest_','fMRI','fme1','fme2','fmph','fMRI_PASS','fMRI_FAM'}; %DTIREV before DTI!!! both "DTIREV.nii" and "DTI.nii" have prefix "DTI"
modalityDependency =  [0,       1,   1,   0,     4,       0,    6,       0,      0,     0,     0,     0,     0           0         ]; %e.g. T1 and T2 must be from same study as lesion

modalityKeys = strrep(modalityKeysVerbose,'_','');
xperimentKeys = {'pre','pos','session'}; %order specifies priority: 1st item checked first!
error_sbjs = [];
for xper = 1: numel(xperimentKeys)

    %create empty structure
    blank = [];
    blank.subjName = [];
    for i = 1: numel(modalityKeys)
        blank.nii.(modalityKeys{i}) =[];
    end;

    %1st: acquire data
    nSubj = 0;
    for s = 1: size(subjDirs,1)%1:nSubjDir2 %(nSubjDir2+1):nSubjDir
        subjName = deblank(subjDirs{s});
        if subjName(1) == '.', continue; end;
        %if (numel(subjName) > 1) && (subjName(2) == '4'), fprintf('SKIPPING %s\n', subjName); continue; end; %ignore folders with underscore, "M2015_needsmatfile"
        if isStringInKeySub (subjName,'_'), continue; end; %ignore folders with underscore, "M2015_needsmatfile"
        subjDir = [baseDir,filesep, subjName]; %no filesep
        %fprintf('%s\n', subjDir);
        nSubj = nSubj + 1;
        imgs(nSubj) = blank;
        imgs(nSubj).subjName = subjName;
        for m = 1:numel(modalityKeysVerbose)
            modality = modalityKeysVerbose{m};
            xLabel = deblank(xperimentKeys{xper}); %e.g. "R01"
            xDir = [subjDir,filesep, xLabel]; %no filesep
            if ~exist(xDir, 'file'), continue; end
            %check the following line which CHris says is pseudocode - DPR
            if strcmpi(xLabel,'ABC') && (strcmpi(modality,'fMRI') || strcmpi(modality,'fMRI_fam')), continue; end;
            %fprintf('%s\n', xDir);
            imgs(nSubj) = findImgsSub(imgs(nSubj), xDir, xLabel, modality, m, modalityDependency(m));
            %imgs(nSubj) = findImgsSub(imgs(nSubj), xDir, xLabel, modalityKeysVerbose, modalityDependency);
            %imgs(nSubj) = findImgsSub(imgs(nSubj), xDir, xLabel)
        end
    end
    fprintf('Found %d subjects in %s\n', nSubj, baseDir);
    if nSubj < 1, return; end;
    if isReportDims
        reportDimsSub(imgs, nSubj);
    end;
    %report results
    startTime = tic;
    % 1st row: describe values
    f = fieldnames(imgs(1).nii);
    str = 'n,subj';
    for i = 1: numel(f)
        str = sprintf('%s\t%s',str, f{i} );
    end
    fprintf('%s\n', str);
    % subsequent rows: source of images
    for s = 1: nSubj
        subj = deblank(imgs(s).subjName);
        subjDir = fullfile(outDir, subj);
        matName = fullfile(subjDir, [subj,'_',xperimentKeys{xper}, '_limegui.mat']);
        imgs(s) = findNovelImgs(subjDir, imgs(s), modalityKeysVerbose,xperimentKeys{xper});
        str = [int2str(s), ',', imgs(s).subjName];
        for i = 1: numel(f)
            x = '-';
            if ~isempty(imgs(s).nii.(f{i})) && isfield(imgs(s).nii.(f{i}), 'x')
                x = imgs(s).nii.(f{i}).x;
                if ~imgs(s).nii.(f{i}).newImg, x = ['~', x]; end;
            end
            str = sprintf('%s\t%s',str, x );
        end
        fprintf('%s\n', str);
    end

    fprintf('Table required %g seconds\n', toc(startTime));
    %copy core files to new folder
    if isExitAfterTable
        fprintf('Disable isExitAfterTable for full analyses\n', str);
        return; %return when we are done
    end
    if exist(outDir, 'file') ~= 7, error('Unable to find folder %s', outDir); end;
    %find images we have already processed
    if isempty(spm_figure('FindWin','Graphics')), spm fmri; end; %launch SPM if it is not running
    process1st = false; % do not check for updates on the cluster!  DPR 20200205



    t_start=tic;

    for s =  1: nSubj
        anyNewImg = false;
        subj = deblank(imgs(s).subjName);
        subjDir = fullfile(outDir, subj);
        if ~isfield(imgs(s).nii.T1,'img')
            fprintf('Skipping %s: no T1!\n', subj);
            continue;
        end
        %imgs(s) = findNovelImgs(subjDir, imgs(s), modalityKeysVerbose);
        global ForcefMRI;
        global ForceRest;
        global ForceASL;
        global ForceDTI;
        global ForceVBM;
        global GPU;
        ForcefMRI=[];
        ForceRest=[];
        ForceASL=[];
        ForceDTI =[];
        ForceVBM = [];
        GPU = gpu_idx-1;
        %666x -
        %imgs(s).nii.fMRI.newImg = false;
        %imgs(s).nii.Rest.newImg = false;
        %666x <-
        %following lines always force reprocessing...
        %if imgs(s).nii.fMRI.newImg, ForcefMRI = true; end;
        %if imgs(s).nii.Rest.newImg, ForceRest = true; end;
        %if imgs(s).nii.ASL.newImg, ForceASL = true; end;
        %666 if imgs(s).nii.DTI.newImg, ForceDTI = true; end;
        %to reprocess one modality for EVERYONE....

        if reprocessRest && isfield(imgs(s).nii.Rest,'img')
            ForceRest = true;
            anyNewImg = true;
        end
        if reprocessfMRI && isfield(imgs(s).nii.fMRI,'img')
            ForcefMRI = true;
            error('xx');
            anyNewImg = true;
        end
        if reprocessASL && isfield(imgs(s).nii.ASL,'img')
            ForceASL = true;
            anyNewImg = true;
        end

        if reprocessVBM && isfield(imgs(s).nii.T1,'img')
            ForceVBM = true;
            anyNewImg = true;
        end
        if reprocessDTI && isfield(imgs(s).nii.DTI,'img')
            ForceDTI = true;
            anyNewImg = true;
        end

        %if imgs(s).nii.DTI.function nii_harvest_parallel (baseDir,outDir)newImg, ForceDTI = true; end;

        %fprintf('%s %d\n', f{i}, imgs.nii.(f{i}).newImg);
        if exist(subjDir,'file') == 0, mkdir(subjDir); end;
        mkdir(fullfile(subjDir,xperimentKeys{xper}))

        matName = fullfile(subjDir,xperimentKeys{xper},[subj,'_',xperimentKeys{xper},'_limegui.mat']);
        mat = [];

        if exist(matName,'file'), mat = load(matName); end;

        for i =  1:numel(f)
            if ~isempty(imgs(s).nii.(f{i})) && (imgs(s).nii.(f{i}).newImg)
                m = f{i}; % modality: T1, T2..
                if ~isfield(imgs(s).nii.(f{i}),'img')
                    mat.(m) = ''; %e.g. we used to have DTI+DTIrev, now we have DTI
                    %warning('Expected field "img" for participant %s modality %s', subj, m);
                    %error('123'); %fix this if it ever happens again
                elseif isempty(imgs(s).nii.(f{i}).img)
                    mat.(m) = ''; %e.g. we used to have DTI+DTIrev, now we have DTI
                else
                    anyNewImg = true;
                    m = f{i}; % modality: T1, T2..
                    x = imgs(s).nii.(f{i}).x; %e.g. experiment name "LIME", "CT"
                    imgin = imgs(s).nii.(f{i}).img; %e.g. '~/dir/m2000/CT/T1.nii'
                    imgout = fullfile(subjDir,x, sprintf('%s_%s_%s.nii',m, subj, x));
                    fprintf('%s -> %s\n',imgin, imgout);
                    moveImgUnGz(imgin, imgout);
                    mat.(m) = imgout;
                end
            end
            if ~isempty(imgs(s).nii.(f{i})) && isfield(imgs(s).nii.(f{i}), 'x') && (~imgs(s).nii.(f{i}).newImg) %CR 2/2017: in case folder names have changed
                m = f{i}; % modality: T1, T2..
                x = imgs(s).nii.(f{i}).x; %e.g. experiment name "LIME", "CT"
                imgout = fullfile(subjDir,x, sprintf('%s_%s_%s.nii',m, subj, x));
                mat.(m) = imgout;
            end
        end

        if anyNewImg
            disp(['GPU allocation activated...#',num2str(GPU)])
            if ~exist('DEBUG','var')
                try
                    matNameGUI = fullfile(subjDir,xperimentKeys{xper}, [subj,'_',xperimentKeys{xper}, '_limegui.mat']);
                    fprintf('Creating %s\n',matNameGUI);
                    save(matNameGUI,'-struct', 'mat');
                    if setOrigin
                        %determine T1 name even if output folder renamed...
                        if imgs(s).nii.T1.newImg || imgs(s).nii.Lesion.newImg, setAcpcSubT1(matNameGUI); end;
                        %if imgs(s).nii.T1.newImg, setAcpcSubT1(matNameGUI); end;
                        %if imgs(s).nii.Lesion.newImg, setAcpcSubT1 (matNameGUI); end;
                        %666 ROGER comments out for parallel version of nii_harvest
                        %if imgs(s).nii.DTI.newImg, setAcpcSubDTI (matNameGUI); end;
                    end
                    %process the datanargin

                    nii_preprocess(mat,[],process1st,true,false);

                    matName = fullfile(subjDir,xperimentKeys{xper}, sprintf('T1_%s_%s_lime.mat', subj, imgs(s).nii.T1.x));
                catch e
                    disp(['ERROR IN ',subj])
                    error_sbjs = [error_sbjs;{subj,xperimentKeys{xper},{e}}];
                end
            else
                matNameGUI = fullfile(subjDir,xperimentKeys{xper}, [subj,'_',xperimentKeys{xper}, '_limegui.mat']);
                fprintf('Creating %s\n',matNameGUI);
                save(matNameGUI,'-struct', 'mat');
                if setOrigin
                    %determine T1 name even if output folder renamed...
                    if imgs(s).nii.T1.newImg || imgs(s).nii.Lesion.newImg, setAcpcSubT1(matNameGUI); end;
                    %if imgs(s).nii.T1.newImg, setAcpcSubT1(matNameGUI); end;
                    %if imgs(s).nii.Lesion.newImg, setAcpcSubT1 (matNameGUI); end;
                    %666 ROGER comments out for parallel version of nii_harvest
                    %if imgs(s).nii.DTI.newImg, setAcpcSubDTI (matNameGUI); end;
                end
                %process the data
                nii_preprocess(mat,[],process1st,true,false);
                error_sbjs=[];
            end
        end
    end
end
fprintf('All done\n');
fprintf ('nii_harvest required %f seconds to run.\n', toc(t_start) );
%end nii_harvest

function reportDimsSub(imgs,nSubj)
for s = 1: nSubj
    subj = deblank(imgs(s).subjName);
    f = fieldnames(imgs(1).nii);
    fprintf('\nID\tMRI\tstudy\tx\ty\tz\tvols\tTR\n');
    for i = 1: numel(f)
        x = '-';
        if ~isempty(imgs(s).nii.(f{i})) && isfield(imgs(s).nii.(f{i}), 'x') && exist(imgs(s).nii.(f{i}).img,'file')
            fnm = imgs(s).nii.(f{i}).img;
            hdr = readNiftiHdrSub(fnm);
            tr = 0;
            if isfield (hdr(1).private, 'timing')
                tr =hdr(1).private.timing.tspace;
            end
            fprintf('%s\t%s\t%s\t%d\t%d\t%d\t%d\t%g\n', subj, f{i}, imgs(s).nii.(f{i}).x, hdr(1).dim(1),hdr(1).dim(2),hdr(1).dim(3),numel(hdr),tr  );
        end
    end


    %     subjDir = fullfile(outDir, subj);
    %     matName = fullfile(subjDir, [subj, '_limegui.mat']);
    %     imgs(s) = findNovelImgs(subjDir, imgs(s), modalityKeysVerbose);
    %     str = [int2str(s), ',', imgs(s).subjName];
    %     for i = 1: numel(f)
    %         x = '-';
    %         if ~isempty(imgs(s).nii.(f{i})) && isfield(imgs(s).nii.(f{i}), 'x')
    %            x = imgs(s).nii.(f{i}).x;
    %            if ~imgs(s).nii.(f{i}).newImg, x = ['~', x]; end;
    %         end
    %         str = sprintf('%s\t%s',str, x );
    %     end
    %     fprintf('%s\n', str);
end
%end reportDimsSub

function tf = endsWithSub(str,pattern) %endsWith only in Matlab 2016 and later
tf = false;
if numel(str) < numel(pattern), return; end
strEnd = str(end-numel(pattern)+1:end);
tf = strncmpi(strEnd,pattern, numel(pattern));
%endsWithSub


function imgs = findNovelImgs(subjDir, imgs, ~, xper)
f = fieldnames(imgs.nii);
for i = 1: numel(f)
    imgs.nii.(f{i}).newImg = true;%??
end
%'fMRI'
if ~isfield(imgs.nii,'T1') || isempty(imgs.nii.T1), return; end
matname = dir(fullfile(subjDir,xper,'T1_*_lime.mat'));
if isempty(matname), return; end
matname = fullfile(matname(1).folder,matname(1).name);
m = load(matname);
if isfield(m,'T1')
    imgs.nii.T1.newImg = ~endsWithSub(m.T1.hdr.fname, ['_',imgs.nii.T1.x,'.nii']);
    imgs.nii.T2.newImg = imgs.nii.T1.newImg;
    imgs.nii.Lesion.newImg = imgs.nii.T1.newImg;
end
if isfield(m,'cbf') && isfield(imgs.nii.ASL, 'x')
    % m.cbf.hdr.fname

    imgs.nii.ASL.newImg = ~endsWithSub(m.cbf.hdr.fname, ['_',imgs.nii.ASL.x,'M0CSF.nii']);
end
if isfield(m,'fa') && isfield(imgs.nii.DTI, 'x')
    imgs.nii.DTI.newImg = ~endsWithSub(m.fa.hdr.fname, ['_',imgs.nii.DTI.x,'d_FA.nii']);
    imgs.nii.DTIrev.newImg = imgs.nii.DTI.newImg;
end
if isfield(m,'RestAve') && isfield(imgs.nii.Rest, 'x')
    imgs.nii.Rest.newImg = ~endsWithSub(m.RestAve.hdr.fname, ['_',imgs.nii.Rest.x,'.nii']);
end
if isfield(m,'fMRIave') && isfield(imgs.nii.fMRI, 'x')
    imgs.nii.fMRI.newImg = ~endsWithSub(m.fMRIave.hdr.fname, ['_',imgs.nii.fMRI.x,'.nii']);
end


%for i = 1: numel(f)
%    fprintf('%s %d\n', f{i}, imgs.nii.(f{i}).newImg);
%end
%end findNovelImgs()

function setAcpcSubT1 (matname)
m = load(matname);
if isfield(m,'T2') && isfield(m,'Lesion') && ~isempty(m.T2) && ~isempty(m.Lesion)
    nii_setOrigin12({m.T2,m.Lesion}, 2, true); %T2
    if isfield(m,'T1') && isfield(m,'Lesion')
        nii_setOrigin12({m.T1}, 1, true); %T1 - crop with lesion
    end
    return;
end
if isfield(m,'T1') && isfield(m,'Lesion') && ~isempty(m.T1) && ~isempty(m.Lesion)
    nii_setOrigin12({m.T1,m.Lesion}, 1, true); %T1 - crop with lesion
    return;
end
if isfield(m,'T1') && ~isempty(m.T1)
    nii_setOrigin12(m.T1, 1, true); %T1 - crop
end

function setAcpcSubDTI (matname)
m = load(matname);
if isfield(m,'DTI') && isfield(m,'DTIrev') && ~isempty(m.DTI) && ~isempty(m.DTIrev)
    nii_setOrigin12({m.DTI,m.DTIrev}, 3, false); %DTI
elseif isfield(m,'DTI') && ~isempty(m.DTI)
    nii_setOrigin12(m.DTI, 3, false); %DTI
end
%end setAcpcSub();

function moveImgUnGz(inname, outname)
[ipth, inam,iext] = fileparts(inname);
[opth, onam,~] = fileparts(outname);
%load data
if strcmpi(iext,'.gz') %unzip compressed data
    inname = gunzip(inname);
    inname = deblank(char(inname));
    [ipth, inam,iext] = fileparts(inname);
end


if(~exist(inname,'file'))
    return;
end
copyfile(inname, outname);


if strcmpi(iext,'.gz') %fsl can not abide with coexisting img.nii and img.nii.gz
    delete(filename);
end;
%copy bvec
ibvec = fullfile(ipth, [inam, '.bvec']);
if exist(ibvec, 'file'),
    obvec = fullfile(opth, [onam, '.bvec']);
    copyfile(ibvec, obvec);
end;
%copy bval
ibval = fullfile(ipth, [inam, '.bval']);
if exist(ibval, 'file')
    obval = fullfile(opth, [onam, '.bval']);
    copyfile(ibval, obval);
end;

%copy .json
ijson = fullfile(ipth, [inam, '.json']);
if exist(ijson, 'file')
    ojson = fullfile(opth, [onam, '.json']);
    copyfile(ijson, ojson);
end
%end moveImgUnGz()


function imgs = findImgsSub(imgs, xDir, xLabel, modalityKey, modalityNum, modalityDependency)
nameFiles = subImgSub(xDir);
if isempty(nameFiles), return; end;
%nameFiles = sort(nameFiles); %take first file for multiimage sequences, e.g. ASL
%nameFiles
%nameFiles(1)
f = fieldnames(imgs.nii);
if ~isempty(imgs.nii.(f{modalityNum})), return; end;
if exist('modalityDependency','var') && (modalityDependency ~= 0)
    dep = f{modalityDependency}; %e.g. T1 scan may depend on Lesion
    if ~isempty(imgs.nii.(dep))
        x = imgs.nii.(dep).x;
        if ~strncmpi(xLabel,x, numel(xLabel))
            %fprintf('"%s" must be from same experiment as "%s" (%s not %s)\n', modalityKey, dep, x, xLabel);
            return;
        end;
        %fprintf('"%s" must be from same experiment as "%s" (%s)\n', modalityKey, dep, x);
    end;
end
for j = 1: numel(nameFiles)
    if strncmpi(modalityKey,nameFiles(j), numel(modalityKey))
        fname = fullfile(xDir, char(nameFiles(j)) );
        %fprintf('%d %s %s %s\n', i, char(f{i}), char(modalityKey), char(nameFiles(j)) );
        imgs.nii.(f{modalityNum}).x = xLabel;
        imgs.nii.(f{modalityNum}).img = fname;
        break;
    end
end;

function nVol = nVolSub(filename)
%report number of volumes
nVol = 0;
[hdr, img] = readNiftiSub(filename);
if isempty(hdr), return; end;
nVol = numel(hdr);
%end nVolSub

function hdr = readNiftiHdrSub(filename)
[p, n,x] = fileparts(filename);
if strcmpi(x,'.gz') %unzip compressed data
    filename = gunzip(filename);
    filename = deblank(char(filename));
    hdr = spm_vol(filename);
    error(fnm);
    delete(fnm);
    return;
end;
hdr = spm_vol(filename);

function [hdr, img] = readNiftiSub(filename)
%load NIfTI (.nii, .nii.gz, .hdr/.img) image and header
% filename: image to open
% open4d: if true all volumes are loaded
%To do:
%  endian: rare, currently detected and reported but not handled
%Examples
% hdr = nii_loadhdrimg('myimg.nii');
% [hdr, img] = nii_loadhdrimg('myimg.nii');
% [hdr, img] = nii_loadhdrimg('img4d.nii');
if ~exist('filename','var')  %fnmFA not specified
    [A,Apth] = uigetfile({'*.nii;*.gz;*.hdr;';'*.*'},'Select image');
    filename = [Apth, A];
end
[fpth, fnam,fext] = fileparts(filename);
if strcmpi(fext,'.img') %hdr/img pair
    filename = fullfile(fpth, [fnam, '.hdr']);
end
if ~exist(filename, 'file')
    error('Unable to find file %s', filename);
end
%load data
if strcmpi(fext,'.gz') %unzip compressed data
    filename = gunzip(filename);
    filename = deblank(char(filename));
end;
hdr = spm_vol(filename);
if hdr(1).dt(1) == 128
    fprintf('Skipping RGB image %s\n', filename);
    hdr = [];
    img = [];
    return;
end
img = spm_read_vols(hdr);
if strcmpi(fext,'.gz') %fsl can not abide with coexisting img.nii and img.nii.gz
    delete(filename);
end;
%end nii_loadhdrimg()

%end findImgsSub()

function nameFiles=subImgSub(pathFolder)
nameFiles=subFileSub(pathFolder);
if isempty(nameFiles), return; end;
n = nameFiles; nameFiles = [];
for i = 1: numel(n)
    [~,~,x] = fileparts(char(deblank(n(i))));
    if ~strncmpi('.gz',x, 3) && ~strncmpi('.nii',x, 4), continue; end;
    nameFiles = [nameFiles; n(i)]; %#ok<AGROW>
end
%end subFileSub()


function nameFiles=subFileSub(pathFolder)
d = dir(pathFolder);
isub = ~[d(:).isdir];
nameFiles = {d(isub).name}';
%end subFileSub()

function isKey = isStringInKeySub (str, imgKey)
isKey = true;
for k = 1 : size(imgKey,1)
    key = deblank(imgKey(k,:));
    pos = strfind(lower(char(str)),lower(key));
    if ~isempty(pos), isKey = pos(1);
        return;
    end;
end
isKey = false;
%isStringInKey()

function [nameFolds,folder]=subFolderSub(pathFolder)
d = dir(pathFolder);
isub = [d(:).isdir];
nameFolds = {d(isub).name}';
folder = d(isub).folder;
nameFolds = nameFolds(cellfun(@(s)isempty(regexp(s,'_')),nameFolds)); %remove folders with underscores
nameFolds = nameFolds(cellfun(@(s)isempty(regexp(s,'\.')),nameFolds)); %remove folders with periods
nameFolds = nameFolds(cellfun(@(s)isempty(regexp(s,' ')),nameFolds)); %remove folders with spaces
nameFolds(ismember(nameFolds,{'.','..'})) = [];
%end subFolderSub()