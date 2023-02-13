close all
clear all
clc

GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Assign directory vars
PATIENT_DIR = '/media/bonilha/AllenProj/sEEG_project/PatientData/CAPES_LEN/Patients';
patients = dir(PATIENT_DIR);
patients(contains({patients.name},'.')) = [];

%% Create electrodes
for p = 1:numel(patients)


    %%%%%%%%%%% Create folders/Detect files
    subject_dir = fullfile(patients(p).folder,patients(p).name);
    electrode_path = dir(fullfile(subject_dir,'images','*CTelectrodelabels_spheres.nii*'));
    electrode_path = fullfile(electrode_path.folder,electrode_path.name);

    electrode_labels = fullfile(subject_dir,'images','electrodelabels.csv');

    if exist(electrode_path,'file') && exist(electrode_labels,"file")
    else
        continue
    end

    save_folder = fullfile(subject_dir,'electrode_niftis');
    mkdir(save_folder)

    %%%%%%%%%%% Find Centroid

    % Define electrode contacts per rod
    electrode_contact_num = 10;

    % Electrode radius
    electrode_radius_perc = 0.0119;

    % Import electrode nifti and binarize
    import = load_nii(electrode_path);
    electrode_img = import.img>0;

    % Import csv
    electrode_csv = readtable(electrode_labels);

    %%%%%%%%%%%%%%%%%%%% Fix overlaping circles

    % Find centroids
    circles = regionprops3(electrode_img,'Centroid','Volume','VoxelList');
    circles.Centroid = round(circles.Centroid);
    
    % Find average circle size
    circle_size = mode(circles.Volume);

    % Find Multi idx
    multi_idx = find(abs(zscore(circles.Volume)) > 1);

    % Fix each multi
    for cir = 1:numel(multi_idx)

        % Working circle
        wk_circle = circles(multi_idx(cir),:);

        % Calculate likely amount of overlap
        num_circle = round(wk_circle.Volume/circle_size);

        % Find farthest two points in object
        wk_vox = wk_circle.VoxelList{:};
        distances = pdist2(wk_vox, wk_vox);
        maxDistance = max(distances(:));
        [index1, index2] = find(distances == maxDistance);
        farthest_points = [wk_vox(index1(1),1) wk_vox(index1(1),2) wk_vox(index1(1),3);...
            wk_vox(index2(1),1) wk_vox(index2(1),2) wk_vox(index2(1),3)];

        % Estimate centers of circles
        step = diff(farthest_points,1)/(num_circle+1);
        est_center = [];
        for nc = 1:num_circle
            est_center = [est_center; farthest_points(1,:)+(step*nc)];
        end
        est_center = round(est_center);

        % Find voxels closest to estimated centers
        template = zeros(size(electrode_img));
        for es = 1:size(est_center,1)
            template(est_center(es,1),est_center(es,2),est_center(es,3)) = 1;
        end
        [~,IDX] = bwdist(template);

        vox_id = [];
        for wv = 1:size(wk_vox,1)
            vox_id = [vox_id; IDX(wk_vox(wv,1),wk_vox(wv,2),wk_vox(wv,3))];
        end

        sep_vox = cell(num_circle,1);
        unique_idx = unique(IDX);
        for sc = 1:size(sep_vox,1)
            sep_vox{sc} = wk_vox(vox_id == unique_idx(sc),:);
        end

        % Find centers of seperated circle
        sep_centroid = [];
        for sc = 1:numel(sep_vox)

            % Find farthest two points in circle
            distances = pdist2(sep_vox{sc}, sep_vox{sc});
            maxDistance = max(distances(:));
            [row,col] = find(distances == maxDistance);

            wk_centroid = [];
            for i = 1:size(row)
                differnce = (sep_vox{sc}(row(i),:) + sep_vox{sc}(col(i),:))./2;
                wk_centroid= [wk_centroid;differnce];
            end
            sep_centroid = [sep_centroid; round(mean(wk_centroid,1))];
        end

        %     % Visualize centers
        %     template = zeros(size(electrode_img));
        %     for wv = 1:size(wk_vox,1)
        %         template(wk_vox(wv,1),wk_vox(wv,2),wk_vox(wv,3)) = 100;
        %     end
        %     for sc = 1:size(sep_centroid,1)
        %         template(sep_centroid(sc,1),sep_centroid(sc,2),sep_centroid(sc,3)) = 400;
        %     end
        %     imshow3D(template(:,:,any(template,[1 2])))

        % Add new circles
        for sc = 1:size(sep_centroid,1)
            volume = size(sep_vox{sc},1);
            centroid = sep_centroid(sc,:);
            vox_list = sep_vox(sc);
            circles = [circles; [volume, centroid, vox_list]];
        end
    end

    % Remove old circles
    circles(multi_idx,:) = [];

    % Check to see if the number electrode center match
    if size(circles,1) == size(electrode_csv,1)
        disp('Automated electrode contact successful')
    else
        error('Automatic electrode contact failed')
    end

    %%%%%%%%%%%%%%%%%%% Find connecting electrodes
    distances = pdist2(circles.Centroid, circles.Centroid);
    distances(distances == 0) = 2000;
    [~,m_idx] = sort(distances,2);
    cents = circles.Centroid;

    electrodes = [];
    iter = 0;
    while any(m_idx,'all')
        iter = iter + 1;
        wk_idx = find(any(m_idx,2));

        for wi = 1:numel(wk_idx)
            disp(['Checking ',num2str(wk_idx(wi)),' and ',num2str(m_idx(wk_idx(wi),1))])

            % Get two points
            p1 = circles.Centroid(wk_idx(wi),:);
            p2 = circles.Centroid(m_idx(wk_idx(wi),1),:);

            % Extrapolate line
            u = (p2-p1)/norm(p2-p1);   % unit vector, p1 to p2
            d = (0:.01:100)';            % displacement from p1, along u
            xyz = p1 + d*u;

            
            


            % Find points closest to line
            distances = pdist2(cents, xyz);
            distances(isnan(distances)) = 10000;

            [d,d_idx] = sort(distances,1);
            [rod_idx,~,g] = unique(d_idx(1,:));

            % Check if 10 electrode contacts exist
            if numel(rod_idx) < electrode_contact_num
                continue
            elseif numel(rod_idx) >  electrode_contact_num
                counts = accumarray(g,1);
                [~, s_idx] = sort(counts,'descend');
                rod_idx = rod_idx(s_idx(1: electrode_contact_num));
            end

            m_idx(rod_idx,:) = nan;
            cents(rod_idx,:) = nan;
            electrodes = [electrodes; {rod_idx}];
            iter = 0;
            break
        end
        if iter == 10
%             v = zeros(size(electrode_img));
%             v_cent = cents(any(cents,2),:);
%             for sc = 1:size(v_cent,1)
%                 [X,Y,Z] = sphere(40);
% 
%                 xr = round(X*6);
%                 yr = round(Y*6);
%                 zr = round(Z*6);
%                 for i = 1:numel(xr)
%                     v(yr(i)+v_cent(sc,2), ...
%                         xr(i)+v_cent(sc,1), ...
%                         zr(i)+v_cent(sc,3)) = 1;
%                 end
%             end
%             volshow(v)
            rod_idx = find(any(m_idx,2));
            m_idx(rod_idx,:) = nan;
            cents(rod_idx,:) = nan;
            electrodes = [electrodes; {rod_idx}];
        end
    end

    % Check to see if the number electrode rods match
    if size(circles,1)/electrode_contact_num == size(electrodes,1)
        disp('Automated electrode rod detection successful')
    else
        error('Automatic electrode rod detection failed')
    end

    %%%%%%%%%%%%% Save Electrode Grouping

    
    % Define Electrode Tags
    electrode_tag = {'deep','mid','per'};
    electrode_idx = {1:3,4:6,7:9};

    for e = 1:numel(electrodes)

        % Dedicate working electrode
        wk_electrode = electrodes{e};
        wk_electrode_coordinates = circles.Centroid(wk_electrode,:);

        % Find points closest to sagital center
        [~,d_idx] = sort(wk_electrode_coordinates(:,2),1);


        for et = 1:numel(electrode_tag)

            % Dedicate working centroids
            sEEG_center= circles.Centroid(wk_electrode(d_idx(electrode_idx{et})),:);

            % Create electrode group niftis
            template = import;
            template.img = zeros(size(template.img));
            for sc = 1:size(sEEG_center,1)
                [X,Y,Z] = sphere(40);
                radius = electrode_radius_perc*(sum(size(template.img))/3);

                xr = round(X*radius);
                yr = round(Y*radius);
                zr = round(Z*radius);
                for i = 1:numel(xr)
                    template.img(yr(i)+sEEG_center(sc,2), ...
                        xr(i)+sEEG_center(sc,1), ...
                        zr(i)+sEEG_center(sc,3)) = 1;
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

            filename = sprintf('electrode%d_%s.nii',e,electrode_tag{et});
            save_nii(template,fullfile(save_folder,filename))
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



