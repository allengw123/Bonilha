function matoutput2nifti(input_mat,output_path,non_seg)
% Converts allen pipeline matfiles to niftis
%
% Requires SPM toolbox
%
%
%   input_mat = path to matfile
%   output_path = output folde
%   non_seg = true/false - option to output nonsegmented T1 (default:false)
%
% Example:
%   matoutput2nifit('~/Downloads/BONPL003.nii','~/Documents/NiftiRequest/BONPL003')
%   matoutput2nifit('~/Downloads/BONPL003.nii','~/Documents/NiftiRequest/BONPL003',true)


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

    % Save non-segmented T1
    if non_seg
        wk_save_name = fullfile(output_path,sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},'T1'));

        wk_nifti = wk_ses.T1;
        hdr = wk_nifti.hdr;
        hdr.fname = wk_save_name;
        spm_write_vol(hdr,wk_nifti.dat);
    end
end
end
