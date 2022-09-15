close all
clear all
clc

% Input
request_xlsx_pth = '/media/bonilha/Elements/Image_Requests/Anees_9.7.22/Data_forAnees_9.7.22.xlsx';
image_database = '/media/bonilha/Elements/Image_database';
save_dir = '/media/bonilha/Elements/Image_Requests';

%%


[~,name,~] = fileparts(request_xlsx_pth);
request_xlsx = readtable(request_xlsx_pth);
save_folder = fullfile(save_dir,name);
mkdir(save_folder)

processedmats = dir(fullfile(image_database,'*','post_qc','*','*.mat'));
processedmats_name = {processedmats.name};

raw_images = dir(fullfile(image_database,'*','*raw*','*','*'));
raw_images = raw_images(~cellfun(@(x) contains(x,'.'),{raw_images.name}));
raw_images_name = {raw_images.name};


error = [];
for r = 1:height(request_xlsx)
    rq_sbj = request_xlsx{r,1}{:};

    idx = find(strcmp(processedmats_name,[strrep(rq_sbj,'_',''),'.mat']));

    if ~isempty(idx)
        rq_mat = processedmats(idx);
        copyfile(fullfile(rq_mat.folder,rq_mat.name),save_folder)
    else
        s_idx = find(contains(raw_images_name,rq_sbj));
        if isempty(s_idx)
            status = 'not found';
        else
            found_name = raw_images_name{s_idx};
            status = [found_name,' found but not processed'];
        end

        error = [error; {rq_sbj, status}];
    end
end

