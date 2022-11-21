function matoutput2nifti(input_mat,output_path,opt)
% Converts allen pipeline matfiles to niftis
%
% Requires SPM toolbox
%
%
%   input_mat = path to matfile
%   output_path = output folder
%   opt = variable that contains fields for options (see below for more detail)
%
%   T1 based options
%       opt.T1.raw = raw output
%       opt.T1.seg = segmented output
%       opt.T1.lesion = lesion output
%       opt.T1.smoothed = smoothed output
%   fMRI based options
%       opt.fmri.raw = raw output
%       opt.T1.seg = segmented output
%       opt.T1.lesion = lesion output
%       opt.T1.smoothed = smoothed output
%   DTI based options
%       opt.T1.raw = raw output
%       opt.T1.seg = segmented output
%       opt.T1.lesion = lesion output
%       opt.T1.smoothed = smoothed output
%   
%
% Example:
%   matoutput2nifti('/home/bonilha/Downloads/BONPL003.nii','/home/bonilha/Documents/NiftiRequest/BONPL003',opt)

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

    %% T1 output
    if isfield(opt,T1)
        % Save segmented files as nifti
        if seg
            % Save mat as nifti
            for m = 1:2
                wk_save_name = fullfile(output_path,sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},matter{m}));
        
                wk_nifti = wk_ses.(['vbm_',matter{m}]);
                hdr = wk_nifti.hdr;
                hdr.fname = wk_save_name;
                spm_write_vol(hdr,wk_nifti.dat);
            end
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
end
