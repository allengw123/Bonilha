%% This script modifies the smoothed files generated from the "smooth_mri_files" script.
% It changes the values of all voxels below a threshold to the value of the
% threshold and then saves them in separate folders for controls gm,
% controls wm, patients left gm, patients left wm, patients right gm, and
% patients right wm. 
%--------------------------------------------------------------------------
% Notes: 
% Here, it uses any threshold. 
% The thr will also apear in the name of the output folders.
%
%--------------------------------------------------------------------------
% Inputs: 
% location is the location of the folder that will contain the modified smoothed files
location = '/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/Smoothed_Files_thr_0.2'; % output
% location1 is the location of the folder that contains the smoothed files for controls 
location1 = '/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/Smoothed_Files_original'
% location2 is the location of the folder that contains the smoothed files for the patients 
location2 = '/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/Smoothed_Files_original'
% define the threshold
thr = 0.2;
%--------------------------------------------------------------------------

%% modifying smoothed files for controls
%create new folders 
Parentfolder = location;
mkdir(fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_controls_gm']));
mkdir(fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_controls_wm']));

Mainfolder = location1;

f = dir(Mainfolder);

% for grey matter
for i = 1:numel(f)
    if startsWith(f(i).name,'smooth10_controls_gm')
        G = dir(fullfile(Mainfolder,f(i).name));
        for j = 1:numel(G)
            if ~startsWith(G(j).name,'.')
                I = load_nii(fullfile(Mainfolder,f(i).name,G(j).name));
                A = I.img;
                A(A<thr) = 0;
                I.img = A;
                B = nonzeros(I.img);
                %disp(j)
                %min(B)
                %fprintf('The min is %4f and the j number is %d',min(B),j);
                display(['The min is ',num2str(min(B)),' and the j number is ',num2str(j)])
                save_nii(I,fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_controls_gm'],G(j).name));
            end
        end
    end
end

% for white matter 
for i = 1:numel(f)
    if startsWith(f(i).name,'smooth10_controls_wm')
        G = dir(fullfile(Mainfolder,f(i).name));
        for j = 1:numel(G)
            if ~startsWith(G(j).name,'.')
                I = load_nii(fullfile(Mainfolder,f(i).name,G(j).name));
                A = I.img;
                A(A<thr) = 0;
                I.img = A;
                B = nonzeros(I.img);
                %disp(j)
                %min(B)
                %fprintf('The min is %4f and the j number is %d',min(B),j);
                display(['The min is ',num2str(min(B)),' and the j number is ',num2str(j)])
                save_nii(I,fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_controls_wm'],G(j).name));
            end
        end
    end
end

%% modifying smoothed files for patients (left and right)
%create new folders 
Parentfolder = location;
mkdir(fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_patients_left_gm']));
mkdir(fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_patients_left_wm']));
mkdir(fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_patients_right_gm']));
mkdir(fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_patients_right_wm']));

Mainfolder = location2;

f = dir(Mainfolder);

% for left patients (gm and wm)
for i = 1:numel(f)
    if startsWith(f(i).name,'smooth10_patients_left_gm')
        G = dir(fullfile(Mainfolder,f(i).name));
        for j = 1:numel(G)
            if ~startsWith(G(j).name,'.')
                I = load_nii(fullfile(Mainfolder,f(i).name,G(j).name));
                A = I.img;
                A(A<thr) = 0;
                I.img = A;
                B = nonzeros(I.img);
                %disp(j)
                %min(B)
                %fprintf('The min is %4f and the j number is %d',min(B),j);
                display(['The min is ',num2str(min(B)),' and the j number is ',num2str(j)])
                save_nii(I,fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_patients_left_gm'],G(j).name));
            end
        end
    end
end

for i = 1:numel(f)
    if startsWith(f(i).name,'smooth10_patients_left_wm')
        G = dir(fullfile(Mainfolder,f(i).name));
        for j = 1:numel(G)
            if ~startsWith(G(j).name,'.')
                I = load_nii(fullfile(Mainfolder,f(i).name,G(j).name));
                A = I.img;
                A(A<thr) = 0;
                I.img = A;
                B = nonzeros(I.img);
                %disp(j)
                %min(B)
                %fprintf('The min is %4f and the j number is %d',min(B),j);
                display(['The min is ',num2str(min(B)),' and the j number is ',num2str(j)])
                save_nii(I,fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_patients_left_wm'],G(j).name));
            end
        end
    end
end

% for right patients (gm and wm) 
for i = 1:numel(f)
    if startsWith(f(i).name,'smooth10_patients_right_gm')
        G = dir(fullfile(Mainfolder,f(i).name));
        for j = 1:numel(G)
            if ~startsWith(G(j).name,'.')
                I = load_nii(fullfile(Mainfolder,f(i).name,G(j).name));
                A = I.img;
                A(A<thr) = 0;
                I.img = A;
                B = nonzeros(I.img);
                %disp(j)
                %min(B)
                %fprintf('The min is %4f and the j number is %d',min(B),j);
                display(['The min is ',num2str(min(B)),' and the j number is ',num2str(j)])
                save_nii(I,fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_patients_right_gm'],G(j).name));
            end
        end
    end
end

for i = 1:numel(f)
    if startsWith(f(i).name,'smooth10_patients_right_wm')
        G = dir(fullfile(Mainfolder,f(i).name));
        for j = 1:numel(G)
            if ~startsWith(G(j).name,'.')
                I = load_nii(fullfile(Mainfolder,f(i).name,G(j).name));
                A = I.img;
                A(A<thr) = 0;
                I.img = A;
                B = nonzeros(I.img);
                %disp(j)
                %min(B)
                %fprintf('The min is %4f and the j number is %d',min(B),j);
                display(['The min is ',num2str(min(B)),' and the j number is ',num2str(j)])
                save_nii(I,fullfile(Parentfolder,['mod_',num2str(thr),'_smooth10_patients_right_wm'],G(j).name));
            end
        end
    end
end

%% Notes 
% save_nii(I,'/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/thr.0.5')
% 
% I = load_nii('/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/smooth10mwp1CON002.nii')'
% A = I.img;
% A(A<0.2) = 0;
% I.img = A;
% B = nonzeros(I.img);
% display(['The min is ',num2str(min(B)),' and the j number is ',num2str(j)])
% save_nii(I,'/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/thr_0.2.nii')


