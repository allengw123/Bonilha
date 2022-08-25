clear all
clc

% Input
GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
box_folder = '/media/bonilha/Elements/MasterSet/post_qc';
disease = 'Patients';
save_folder = '/home/bonilha/Downloads';
%%

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Create save folder
disease_save_folder = fullfile(save_folder,'mat2nii_savefolder',disease);
mkdir(disease_save_folder)

% Input folder
wk_folder = fullfile(box_folder,disease);

% Detect mat files
mat_files = dir(fullfile(wk_folder,'*.mat'));

% Save segmented files as nifti
matter = {'gm','wm'};
for i = 1:numel(mat_files)
    wk_mat = load(fullfile(mat_files(i).folder,mat_files(i).name));
    fn = fieldnames(wk_mat);

    wk_sbj_name = extractBefore(mat_files(i).name,'.mat');
    wk_save_folder = fullfile(disease_save_folder,wk_sbj_name);

    mkdir(wk_save_folder)
    for f = 1:numel(fn)

        % Skip pipeline info
        if contains(fn{f},'pipelineinfo')
            continue
        end

        wk_ses = wk_mat.(fn{f});

        % Save mat as nifti
        for m = 1:2
            save_nifti =[];
            wk_save_name = fullfile(wk_save_folder,sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},matter{m}));
            wk_save_name_cus = fullfile(wk_save_folder,sprintf('%s_%s_%s_%s.nii',wk_sbj_name,fn{f},matter{m},'custom'));

            wk_nifti = wk_ses.(['vbm_',matter{m}]);
            hdr = wk_nifti.hdr;
            hdr.fname = wk_save_name;
            spm_write_vol(hdr,wk_nifti.dat);
            save_nii(create_nifti_template(wk_nifti),wk_save_name_cus)
        end
    end
end
%% Functions
function out = create_nifti_template(input)

out = [];

out.filetype = 2;
out.fileprefix = input.hdr.fname;
out.machine = 'ieee-le';
out.img = single(input.dat);

% hk
out.hdr.hk.sizeof_hdr = 348;
out.hdr.hk.data_type = '';
out.hdr.hk.db_name ='';
out.hdr.hk.extents = 0;
out.hdr.hk.session_error = 0;
out.hdr.hk.regular = 'r';
out.hdr.hk.dim_info = 0;

% dime
out.hdr.dime.dim = [3,113,137,113,1,1,1,1];
out.hdr.dime.intent_p1 = 0;
out.hdr.dime.intent_p2 = 0;
out.hdr.dime.intent_p3 = 0;
out.hdr.dime.intent_code = 0;
out.hdr.dime.datatype = 16;
out.hdr.dime.bitpix = 32;
out.hdr.dime.slice_start = 0;
out.hdr.dime.pixdim = [1,1.500000000000000,1.500000000000000,1.500000000000000,0,0,0,0];
out.hdr.dime.vox_offset = 352;
out.hdr.dime.scl_slope = 1;
out.hdr.dime.scl_inter = 0;
out.hdr.dime.slice_end = 0;
out.hdr.dime.slice_code = 0;
out.hdr.dime.xyzt_units = 10;
out.hdr.dime.cal_max = 0;
out.hdr.dime.cal_min = 0;
out.hdr.dime.slice_duration = 0;
out.hdr.dime.toffset = 0;
out.hdr.dime.glmax = max(input.dat,[],'all','omitnan');
out.hdr.dime.glmin = min(input.dat,[],'all','omitnan');

% hist
out.hdr.hist.descrip = input.hdr.descrip;
out.hdr.hist.aux_file = '';
out.hdr.hist.qform_code = 0;
out.hdr.hist.sform_code = 0;
out.hdr.hist.quatern_b = 0;
out.hdr.hist.quatern_c = 1;
out.hdr.hist.quatern_d = 0;
out.hdr.hist.qoffset_x = 84;
out.hdr.hist.qoffset_y = -120;
out.hdr.hist.qoffset_z = -72;
out.hdr.hist.srow_x = [-1.500000000000000,0,0,84];
out.hdr.hist.srow_y = [0,1.500000000000000,0,-120];
out.hdr.hist.srow_z = [0,0,1.500000000000000,-72];
out.hdr.hist.intent_name = '';
out.hdr.hist.magic = 'n+1';
out.hdr.hist.originator = [57,81,49,0,-32768];
out.hdr.hist.rot_orient = [1,2,3];
out.hdr.hist.flip_orient = [3,0,0];
end
