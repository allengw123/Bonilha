close all
clear all
clc

% Input
GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
image_database = '/media/bonilha/Elements/Image_database';
request_database = '/media/bonilha/Elements/Image_Requests/Anees_9.7.22';

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')
%% Pull code

% Detect request name
name = extractAfter(request_database,[fileparts(request_database),'/']);

% Load request xlsx
request_xlsx_pth = fullfile(request_database,[name,'.xlsx']);
sheets = sheetnames(request_xlsx_pth);

% Create request save folder
request_save_path = fullfile(request_database,'request');
mkdir(request_save_path)
cd(request_save_path)

% Find post_qc processed .mat
processedmats = dir(fullfile(image_database,'*','post_qc','*','*.mat'));
processedmats_name = {processedmats.name};

% Find raw images
raw_images = dir(fullfile(image_database,'*','*nii_proc_format*','*','*'));
raw_images = raw_images(~cellfun(@(x) contains(x,'.'),{raw_images.name}));
raw_images_name = {raw_images.name};

error = [];
for s = 1:numel(sheets)

    wk_sheet = char(sheets(s));
    wk_request_save_path = fullfile(request_save_path,wk_sheet);
    if exist(wk_request_save_path,'dir')
        mkdir(wk_request_save_path)
    end

    textprogressbar(0,['Pulling Requested subjects - ',wk_sheet,' (',num2str(s),'/',num2str(numel(sheets)),')']);

    save_processed_folder = fullfile(wk_request_save_path,'processed');
    mkdir(save_processed_folder)

    save_raw_folder = fullfile(wk_request_save_path,'raw_images');
    mkdir(save_raw_folder)

    request_xlsx = readtable(request_xlsx_pth,'Sheet',wk_sheet,'ReadVariableNames',false);

    % Delete nonrequested subjects
    requested_subjects = cellfun(@(x) strrep(x,'_',''),request_xlsx{:,1},'UniformOutput',false);

    proccess_subjects = dir(save_processed_folder);
    proccess_subjects(strcmp({proccess_subjects.name},'.')|strcmp({proccess_subjects.name},'..')) = [];
    rm_processed_subjects = proccess_subjects(~cellfun(@(x) contains(x,requested_subjects),{proccess_subjects.name}));
    for i = 1:numel(rm_processed_subjects)
        delete(fullfile(rm_processed_subjects(i).folder,rm_processed_subjects(i).name))
        disp(['Removing ',rm_processed_subjects(i).name,' from proccessed folder'])
    end

    raw_subjects = dir(save_raw_folder);
    raw_subjects(strcmp({raw_subjects.name},'.')|strcmp({raw_subjects.name},'..')) = [];
    rm_raw_subjects = raw_subjects(~cellfun(@(x) contains(x,requested_subjects),{raw_subjects.name}));
    for i = 1:numel(rm_raw_subjects)
        rmdir(fullfile(rm_raw_subjects(i).folder,rm_raw_subjects(i).name),'s')
        disp(['Removing ',rm_processed_subjects(i).name,' from raw folder'])
    end

    for r = 1:height(request_xlsx)

        % Request subject
        rq_sbj = strrep(request_xlsx{r,1}{:},'_','');

        % Find processed mat file idx
        p_idx = find(contains(processedmats_name,rq_sbj));

        % Find raw image idx
        r_idx = find(contains(raw_images_name,rq_sbj));

        if ~isempty(p_idx)

            % Copy matfile
            rq_mat = processedmats(p_idx);
            if exist(fullfile(save_processed_folder,rq_mat.name),'file')
                continue
            else
                copyfile(fullfile(rq_mat.folder,rq_mat.name),save_processed_folder)
            end

            % Copy raw folder
            r_folder = raw_images(r_idx);
            if exist(fullfile(save_raw_folder,r_folder.name),'dir')
                continue
            else
                copyfile(fullfile(r_folder.folder,r_folder.name),fullfile(save_raw_folder,r_folder.name))
            end
        else
            % Save errors
            s_idx = find(contains(raw_images_name,rq_sbj));
            if isempty(s_idx)
                status = 'not found';
            else
                found_name = raw_images_name{s_idx};
                status = [found_name,' found but not processed'];
            end
            error = [error; [{sheets(s)},{rq_sbj},{status}]];
        end

        % Display progress bar
        textprogressbar(1,r/height(request_xlsx)*100,[wk_sheet,' - Subject ',r_folder.name,' pulled']);
    end
end

% Complete progress bar
textprogressbar(2,'Pull Request Completed');

% Save error file
if exist(fullfile(request_save_path,'errors.xlsx'),'file')
    delete(fullfile(request_save_path,'errors.xlsx'))
end
if ~isempty(error)
    writecell(error,fullfile(request_save_path,'errors.xlsx'))
end

% Transfer mat2nifti script
cmd = sprintf( 'cp %s %s', fullfile(GITHUB_PATH,'Functions','matoutput2nifti.m'), request_save_path);
system( cmd );