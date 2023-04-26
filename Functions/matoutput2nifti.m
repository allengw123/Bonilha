function matoutput2nifti(input_mat,output_path,opt)
% Converts allen pipeline matfiles to niftis
%
% Requires SPM toolbox
%
%
%   input_mat = path to matfile
%   output_path = output folder
%   opt = variable that contains fields for options (see below for more
%       detail) --> if missing field is was just default to false
%
%   T1 based options
%       opt.T1.img = true/false --> raw output
%       opt.T1.seg = true/false --> segmented output
%       opt.T1.matter = 'gm'/'wm'/'both'[default] --> matter output
%       opt.T1.lesion = true/false --> lesion output
%   fMRI based options
%       work in progress
%   DTI based options
%       work in process
%   
%
% Example:
%   opt = [];
%   opt.T1.seg = true;
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
    if isfield(opt,'T1')
        if output_log(opt.T1,'matter')
            if strcmp(opt.T1.matter,'both')
                matter = {'gm','wm'};
            elseif strcmp(opt.T1.matter,'gm') || strcmp(opt.T1.matter,'wm')
                matter = {opt.T1.matter};
            else
                error('opt.T1.matter input not recognized [',opt.T1.matter,']')
            end
        else
            matter = {'gm','wm'};
        end

        % Save T1 image as nifti
        if output_log(opt.T1,'img')
            if ~exist(fullfile(output_path,'T1'),'dir');mkdir(fullfile(output_path,'T1'));end
            wk_save_name = fullfile(output_path,'T1',sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},'T1'));
            write_field(wk_ses.T1,'T1_normalized',wk_save_name)
        end

        % Save segmented files as nifti
        if output_log(opt.T1,'seg')
            for m = 1:numel(matter)
                if ~exist(fullfile(output_path,'T1'),'dir');mkdir(fullfile(output_path,'T1'));end

                wk_save_name = fullfile(output_path,'T1',sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},matter{m}));
            if any(contains(fieldnames(wk_ses.matter_maps),'enantimorphic'))
                writefield(wk_ses.matter_maps.lesion_corrected,[matter{m},'_les'],wk_save_name)
            else
                write_field(wk_ses.matter_maps.(matter{m}),['vbm_',matter{m}],wk_save_name)
            end
            end
        end

        % Save lesion file as nifti
        if output_log(opt.T1,'lesion')
            if ~exist(fullfile(output_path,'T1'),'dir');mkdir(fullfile(output_path,'T1'));end
            wk_save_name = fullfile(output_path,'T1',sprintf('%s_%s_%s.nii',wk_sbj_name,fn{f},'lesion'));
            if any(contains(fieldnames(wk_ses),'lesion'))
            write_field(wk_ses.lesion,'lesion',wk_save_name)
            end
        end
    end

end
end

%% Funcitons

function log = output_log(module,output)
if isfield(module,output)
    if module.(output)
        log = true;
        return
    end
end
log = false;
end

function write_field(matfile,field,output_name)
root = fileparts(output_name);
if ~exist('root','dir')
    mkdir(root)
end

if isfield(matfile,field)
    wk_nifti = matfile.(field);
    hdr = wk_nifti.hdr;
    hdr.fname = output_name;
    spm_write_vol(hdr,wk_nifti.dat);
else
    disp([field, ' not found in matfile... skipping'])
end
end