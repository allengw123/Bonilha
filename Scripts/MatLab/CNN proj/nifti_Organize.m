clear
clc

% githubpath='C:\Users\allen\Documents\GitHub\Bonilha';
githubpath = 'C:\Users\bonilha\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

input_folder = 'F:\PatientData\smallSet\Cat12_segmented';

folders = {dir(input_folder).name};
folders(strcmp(folders,'.')|strcmp(folders,'..')) = [];

output = 'F:\PatientData\FINALSET'

matter = {'GM','WM','GMWM'};
%%
parfor i = 1:numel(folders)
    d_output = fullfile(output,folders{i});
    mkdir(d_output)
    
    nifti_dir =  dir(fullfile(input_folder,folders{i},'*','mri','mwp*'));
    nifti_files = {nifti_dir.name};
    nifti_fullpath = [];
    for f = 1:numel(nifti_dir)
        nifti_fullpath = [nifti_fullpath; {fullfile(nifti_dir(f).folder,nifti_dir(f).name)}];
    end
    subject_names = unique(cellfun(@(x) extractAfter(x,4),nifti_files,'UniformOutput',false));

    for s = 1:numel(subject_names)
        s_folder = fullfile(d_output,extractBefore(subject_names{s},'.nii'));
        mkdir(s_folder)

        for m = 1:numel(matter)
            if strcmp(matter{m},'GM')
                prefix = 'mwp1';
            elseif strcmp(matter{m},'WM')
                prefix = 'mwp2';
            elseif strcmp(matter{m},'GMWM')
                prefix = {'mwp1','mwp2'};
            end

            if m ==3
                tmpfiles = nifti_fullpath(cellfun(@(x) endsWith(x,subject_names{s}),nifti_fullpath));
                
                img1 = load_nii(tmpfiles{1});
                img2 = load_nii(tmpfiles{2});

                com_img = img1;
                com_img.img = img1.img + img2.img;

                save_nii(com_img,fullfile(s_folder,['GMWM_',subject_names{s}]))
            else
                input_file = nifti_fullpath{cellfun(@(x) endsWith(x,[prefix,subject_names{s}]),nifti_fullpath)};
                output_file = fullfile(s_folder,[matter{m},'_',subject_names{s}]);
                copyfile(input_file,output_file)
            end
        end
    end
end
            