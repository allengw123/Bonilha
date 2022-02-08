%% This script is used to run the statistical analysis using smp12 for grey matter.
%--------------------------------------------------------------------------
% Notes: 
% It creates two big lists: one with all the modified smoothed files for controls
% and one with selected a part of the patients' group.
% The purpose of this script is to run an Independent t-test(Two-sample t-test) 
% using the "Specify 2nd-level" of spm12. 
% The files are those: smooth10mwp1name.nii and smooth10mwp2name.nii.
% The code takes into account that under the main patients' folder (with the 
% modified smoothed files there are two subfolders: 1-bi_patients 2-left_patients.
%--------------------------------------------------------------------------
% Inputs: 
% location1 is the location of the folder that contains all the controls
location1 = 'C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_mod_0.5\mod_0.5_smooth10_controls_gm';
% location2 is the location of the folder that contains all the patients
location2 = 'C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_mod_0.5';
% location3 is the location where you want the data to be saved
location3 = 'C:\Users\bonilha\Documents\Project_Eleni\Results_CAT12_0.5\Results_wmp1_only_right';
%--------------------------------------------------------------------------

%% create folders for results 
Parentfolder = 'C:\Users\bonilha\Documents\Project_Eleni\Results_CAT12_0.5';
mkdir(fullfile(Parentfolder,'Results_wmp1_all_files'));
mkdir(fullfile(Parentfolder,'Results_wmp1_only_left'));
mkdir(fullfile(Parentfolder,'Results_wmp1_only_right'));

%% for the controls group - Group 1 (grey matter)
P = location1;

f = dir(P);

%preallocating a cell where the list of files will be saved
big_list_gm_c = cell(300,1);
big_list_gm_c{300,1} = [];
%big_list_gm_c = strings(300,1); NEEDS to be a cell

count = 1;
for i = 1:numel(f)
    if startsWith(f(i).name,'smooth10mwp1')
        big_list_gm_c{count} = [(fullfile(P,f(i).name)) ',1' ];
        count = count + 1;
    end
end
%big_list_gm_c = big_list_gm_c.'!!!use this if you don't wish topreallocate!!!
big_list_gm_c = big_list_gm_c(~all(cellfun(@isempty,big_list_gm_c),2),:); 

%% for the patients group - Group 2 (gray matter)
P = location2;

f = dir(P);

%preallocating a cell where the list of files will be saved
big_list_gm_p = cell(300,1);
big_list_gm_p{300,1} = [];
%big_list_gm_p = strings(300,1); NEEDS to be a cell

count = 1;
for i = 1:numel(f)
    if  startsWith(f(i).name,'mod_0.5_smooth10_patients_right_gm')
        % || startsWith(f(i).name,'mod_0.5_smooth10_patients_right_gm')
        G = dir(fullfile(P,f(i).name));
        for j = 1:numel(G)
            if startsWith(G(j).name,'smooth10mwp1')
                big_list_gm_p{count} = [(fullfile(P,f(i).name,G(j).name)) ',1' ];
                count = count + 1;
            end
        end 
    end
end
%big_list_gm_p = big_list_gm_p.' !!!use this if you don't wish topreallocate!!!
big_list_gm_p = big_list_gm_p(~all(cellfun(@isempty,big_list_gm_p),2),:); 

%% Statistical analysis 
matlabbatch{1}.spm.stats.factorial_design.dir = {location3};
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = big_list_gm_c;
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = big_list_gm_p;
matlabbatch{1}.spm.stats.factorial_design.des.t2.dept = 0;
matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;
matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova = 0;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
spm_jobman('run',matlabbatch);

% in the next steps after getting those data use the following t-contrast:
% 1 -1
% Group 1 needs to be the controls 
% Group 2 needs to be the patients 

 
