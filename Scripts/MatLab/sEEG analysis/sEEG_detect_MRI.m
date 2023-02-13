close all
clear all
clc

GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Assign directory vars
PATIENT_DIR = '/media/bonilha/AllenProj/sEEG_project/PatientData/CAPES_LEN/';
patients = dir(PATIENT_DIR);
patients(contains({patients.name},'.')) = [];

master_electrode={'LA','LAH','LAI','LLF','LMF','LPH','LPI','RA','RAH','RAI','RLF','RMF','RPH','RPI'};


%% Create electrodes
for p = 1:numel(patients)


    %%%%%%%%%%% Create folders/Detect files
    subject_dir = fullfile(patients(p).folder,patients(p).name);

    electrode_path = dir(fullfile(subject_dir,'raw','derivatives','ieeg_recon','module2','*T1w_electrode_spheres.nii*'));
    electrode_path = fullfile(electrode_path.folder,electrode_path.name);
    
    ct_path = dir(fullfile(subject_dir,'raw','derivatives','ieeg_recon','module2','*mri_ct_thresholded.nii*'));
    ct_path = fullfile(ct_path.folder,ct_path.name);

    electrode_labels = dir(fullfile(subject_dir,'raw','derivatives','ieeg_recon','module2','*names.txt'));
    electrode_labels = fullfile(electrode_labels.folder,electrode_labels.name);

    diff_labels = {'DIFF','Diff'};
    for d = 1: numel(diff_labels)
        diffusion_path = dir(fullfile(subject_dir,'raw','Pre_MRI',['*',diff_labels{d},'*.nii*']));
        if numel(diffusion_path) == 1
            diffusion_path = fullfile(diffusion_path.folder,diffusion_path.name);
            break
        elseif numel(diffusion_path) > 1
            diffusion_path = fullfile(diffusion_path(1).folder,diffusion_path(1).name);
            break
        else
            continue
        end
    end
    [~,~,ext] = fileparts(diffusion_path);

    if strcmp(ext,'.gz')
        ext = ['.nii',ext];
    end
    json_path = strrep(diffusion_path,ext,'.json');
    bval_path = strrep(diffusion_path,ext,'.bval');
    bvec_path = strrep(diffusion_path,ext,'.bvec');

    if exist(electrode_path,'file') && exist(electrode_labels,"file")
    else
        continue
    end

    save_folder = fullfile(subject_dir,'structural');
    electrode_savefolder = fullfile(save_folder,'Tractography','Electrodes');

    mkdir(electrode_savefolder)

    copyfile(electrode_path,fullfile(save_folder,'Post Implant'))
    copyfile(ct_path,fullfile(save_folder,'Post Implant'))
    copyfile(electrode_labels,fullfile(save_folder,'Post Implant'))
    copyfile(diffusion_path,fullfile(save_folder,'Diffusion'))
    copyfile(json_path,fullfile(save_folder,'Diffusion'))
    copyfile(bval_path,fullfile(save_folder,'Diffusion'))
    copyfile(bvec_path,fullfile(save_folder,'Diffusion'))

    %%%%%%%%%%% Find Centroid

    % Define electrode contacts per rod
    electrode_contact_num = 10;

    % Electrode radius
    electrode_radius_perc = 0.0119;

    % Import electrode nifti and binarize
    import = load_untouch_nii(electrode_path);

    % Import csv
    electrode_csv = readtable(electrode_labels,'ReadVariableNames',false);

    % Find centers
    electrode_idx = unique(import.img);
    electrode_idx(electrode_idx ==0) = [];

    circles = [];
    for e = 1:numel(electrode_idx)
        cents = regionprops3(import.img==electrode_idx(e),'Centroid','Volume','VoxelList');
        circles = [circles; [electrode_csv{e,1},{round(cents.Centroid)}]];
    end

    % Check to see if the number electrode center match
    if size(circles,1) == size(electrode_csv,1)
        disp('Automated electrode contact successful')
    else
        error('Automatic electrode contact failed')
    end

    % Define Electrode Tags
    electrode_targets = unique(cellfun(@(x) x(isstrprop(x,'alpha')),circles(:,1),'UniformOutput',false));
    electrode_locs = cellfun(@(x) x(isstrprop(x,'alpha')),circles(:,1),'UniformOutput',false);
    electrode_group_labels = {'D','M','P'};
    electrode_group_idx = {1:3,4:6,7:9};

    for e = 1:numel(electrode_targets)

        if any(cellfun(@(x) strcmp(x,electrode_targets{e}),master_electrode))
        else
            continue
        end
        for eg = 1:numel(electrode_group_labels)

            % Dedicate working electrode
            wk_electrode = circles(strcmp(electrode_locs,electrode_targets{e}),:);
            wk_electrode = wk_electrode(electrode_group_idx{eg},:);

            % Create electrode group niftis
            template = import;
            template.img = zeros(size(template.img));
            for sc = 1:size(wk_electrode,1)
                [X,Y,Z] = sphere(40);
                radius = electrode_radius_perc*(sum(size(template.img))/3);

                xr = round(X*radius);
                yr = round(Y*radius);
                zr = round(Z*radius);
                for i = 1:numel(xr)
                    template.img(yr(i)+wk_electrode{sc,2}(2), ...
                        xr(i)+wk_electrode{sc,2}(1), ...
                        zr(i)+wk_electrode{sc,2}(3)) = 1;
                end
            end

            % Fill in sphere
            slice_idx = find(permute(any(template.img,[1 2]),[3 2 1]));
            for s = 1:numel(slice_idx)
                wk_img = template.img(:,:,slice_idx(s));
                row_idx = find(any(wk_img,2));
                for r = 1:numel(row_idx)
                    wk_row = wk_img(row_idx(r),:);
                    range = find(wk_row);
                    range = sort(range);
                    wk_row(min(range):max(range)) = 1;
                    wk_img(row_idx(r),:) = wk_row;
                end
                template.img(:,:,slice_idx(s)) = wk_img;
            end


            filename = sprintf('%s_%s.nii',electrode_targets{e},electrode_group_labels{eg});
            save_untouch_nii(template,fullfile(electrode_savefolder,filename))
        end
    end
end


%% Find average electrode size
% old_dir = dir(fullfile('/media/bonilha/AllenProj/sEEG_project/PatientData/Original','*','structural','Tractography','Electrodes','*.nii.gz'));
%
% e_size = [];
% for e = 1:size(old_dir,1)
%     temp_elec = load_nii(fullfile(old_dir(e).folder,old_dir(e).name));
%     vol = sum(temp_elec.img>0,'all')/3;
%     radius = ((vol*3)/(4*pi))^(1/3);
%     radius_percent = radius/(sum(size(temp_elec.img))/3);
%     e_size = [e_size;radius_percent];
% end
%
% electrod_radius = mean(e_size)



