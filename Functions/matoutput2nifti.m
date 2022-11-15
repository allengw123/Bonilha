function matoutput2nifti(input_mat,output_path,non_seg,smooth)
% Converts allen pipeline matfiles to niftis
%
% Requires SPM toolbox
%
%
%   input_mat = path to matfile
%   output_path = output folde
%   non_seg = true/false - option to output nonsegmented T1 (default:false)
%   smooth = true/false - option to output smoothed segmented images (default:false)
%
% Example:
%   matoutput2nifti('/home/bonilha/Downloads/BONPL003.nii','/home/bonilha/Documents/NiftiRequest/BONPL003')
%   matoutput2nifti('/home/bonilha/Downloads/BONPL003.nii','/home/bonilha/Documents/NiftiRequest/BONPL003',true)


% Check Dependencies
if isempty(which('spm_write_vol'))
    error('Cannot find spm functions. Please make sure spm is in path')
end
if ~exist(input_mat,'file')
    error('input file does not exist')
end
if ~exist(output_path,'dir')
    error('output folder does not exist')
end
if ~exist('non_seg','var')
    non_seg = false;
end
if ~exist('smooth','var')
    smooth = false;
end
if ~islogical(non_seg)
    error('3rd argument must be true or false denoting whether you want segmented image output')
end
if ~islogical(smooth)
    error('4th argument must be true or false denoting whether you want smoothed image output')
end

% Save segmented files as nifti
matter = {'gm','wm'};

wk_mat = load(input_mat);
fn = fieldnames(wk_mat);

[~,name,~] = fileparts(input_mat);
wk_sbj_name = name;

for f = 1:numel(fn)

    % Skip pipeline info
    if contains(fn{f},'pipelineinfo')
        continue
    end

    wk_ses = wk_mat.(fn{f});

    % Save mat as nifti
    for m = 1:2
        wk_save_name = fullfile(output_path,sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},matter{m}));

        wk_nifti = wk_ses.(['vbm_',matter{m}]);
        hdr = wk_nifti.hdr;
        hdr.fname = wk_save_name;
        spm_write_vol(hdr,wk_nifti.dat);
    end


    if isfield(wk_ses,'lesion')
        wk_save_name = fullfile(output_path,sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},'lesion'));

        wk_nifti = wk_ses.lesion;
        hdr = wk_nifti.hdr;
        hdr.fname = wk_save_name;
        spm_write_vol(hdr,wk_nifti.dat);
    end

    % Save non-segmented T1
    if non_seg
        wk_save_name = fullfile(output_path,sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},'T1'));

        wk_nifti = wk_ses.T1;
        hdr = wk_nifti.hdr;
        hdr.fname = wk_save_name;
        spm_write_vol(hdr,wk_nifti.dat);
    end

    % Save smooth T1
    if smooth
        for m = 1:2
            wk_save_name = fullfile(output_path,sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},['smooth_',matter{m}]));

            wk_nifti = wk_ses.(['smooth_vbm_',matter{m}]);
            hdr = wk_nifti.hdr;
            hdr.fname = wk_save_name;
            spm_write_vol(hdr,wk_nifti.dat);
        end
    end
end
end
