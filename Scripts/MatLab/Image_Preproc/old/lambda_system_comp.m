
GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha'; 

cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

nii_preprocess_subfolders('/media/bonilha/Elements/test')

system76 = load('PITP0058.mat');
lambda = load('lambda_PITP0058.mat');

figure;
subplot(3,1,1)
imagesc(lambda.RestAve.dat(:,:,34))
title('lambda')
subplot(3,1,2)
imagesc(system76.RestAve.dat(:,:,34))
title('System76')
subplot(3,1,3)
imagesc(lambda.RestAve.dat(:,:,34) - system76.RestAve.dat(:,:,34))
title('Difference')
