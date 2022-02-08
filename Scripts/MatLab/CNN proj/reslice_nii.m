%% reslice .nii files
% CODE NOT DONE
clc
clear
data_path_exp ='C:\Users\bonilha\Documents\Project_Eleni\SVM_results_all\SVM_results';

all_files = {dir(fullfile(data_path_exp,'*nii')).name}';

count =1;
for i = 1:numel(all_files)
    file_img = load(fullfile(data_path_exp,num2str(cell2mat(all_files(i)))));

    % Reslice
    matlabbatch{1}.spm.spatial.coreg.write.ref = {'C:\Users\bonilha\Documents\Project_Eleni\mni152.nii,1'};
    matlabbatch{1}.spm.spatial.coreg.write.source = {fullfile(data_path_exp,[temproi_name{ROI},'.nii'])};
    matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 4;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'r'; 
    spm_jobman('run',matlabbatch);
end