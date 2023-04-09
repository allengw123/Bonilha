close all
clear all
clc

% Input
GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
image_database = '/media/bonilha/Elements/Image_database';
request_database = '/media/bonilha/Elements/Image_Requests/forZeke_BNT_4.7.23';

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Tags
tags =  {'problem','lesion'};

% Options
pull_raw = true;
%% Pull code

% Detect request name
name = extractAfter(request_database,[fileparts(request_database),'/']);

% Load request xlsx
request_xlsx_pth = dir(fullfile(request_database,'*.xlsx'));
sheets = sheetnames(fullfile(request_xlsx_pth.folder,request_xlsx_pth.name));

% Create request save folder
request_save_path = fullfile(request_database,'request');
mkdir(request_save_path)
cd(request_save_path)

% Find post_qc processed .mat
processedmats = dir(fullfile(image_database,'*','post_qc','*','*.mat'));
processedmats_name = {processedmats.name};

% Find raw images
raw_images = dir(fullfile(image_database,'*','*raw*','*','*'));
raw_images = raw_images(~cellfun(@(x) contains(x,'.'),{raw_images.name}));
raw_images_name = {raw_images.name};
raw_images_name = cellfun(@(x) strrep(x,'_',''),raw_images_name,'UniformOutput',false);

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
    save_processed_folder_dir = dir(save_processed_folder);


    if pull_raw
        save_raw_folder = fullfile(wk_request_save_path,'raw_images');
        mkdir(save_raw_folder)
        save_raw_folder_dir = dir(save_raw_folder);
    end

    request_xlsx = readtable(fullfile(request_xlsx_pth.folder,request_xlsx_pth.name),'Sheet',wk_sheet,'ReadVariableNames',false);
    
    for r = 1:height(request_xlsx)

        % Request subject
        rq_sbj = strrep(request_xlsx{r,1}{:},'_','');

        if any(contains({save_processed_folder_dir.name},rq_sbj))
            textprogressbar(1,r/height(request_xlsx)*100,[wk_sheet,' - Subject ',rq_sbj,' pulled']);
            continue
        end

        % Find processed mat file idx
        p_idx = find(contains(processedmats_name,rq_sbj));

        if numel(p_idx) == 1

            % Copy matfile
            rq_mat = processedmats(p_idx);
            if exist(fullfile(save_processed_folder,rq_mat.name),'file')
                continue
            else
                copyfile(fullfile(rq_mat.folder,rq_mat.name),save_processed_folder)
            end
            if pull_raw
                % Find raw image idx
                r_idx = find(contains(raw_images_name,rq_sbj));

                % Copy raw folder
                r_folder = raw_images(r_idx);
                if exist(fullfile(save_raw_folder,r_folder.name),'dir')
                    continue
                elseif numel(r_folder)>1
                    temp = r_folder(contains({r_folder.name},tags));
                    copyfile(fullfile(temp.folder,temp.name),fullfile(save_raw_folder,r_folder.name));
                else
                    copyfile(fullfile(r_folder.folder,r_folder.name),fullfile(save_raw_folder,r_folder.name))
                end
            end

            % Display progress bar
            textprogressbar(1,r/height(request_xlsx)*100,[wk_sheet,' - Subject ',rq_sbj,' pulled']);

        elseif numel(p_idx)>1
            status = ['multiple copies found ',[processedmats(p_idx).name]];
            error = [error; [{sheets(s)},{rq_sbj},{status}]];
            textprogressbar(1,r/height(request_xlsx)*100,status);
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
            textprogressbar(1,r/height(request_xlsx)*100,[rq_sbj,' ',status]);
        end
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
