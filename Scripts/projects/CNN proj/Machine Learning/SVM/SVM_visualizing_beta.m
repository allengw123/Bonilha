%% SVM visualization beta 

A = load_nii('mni152.nii');
image=double(A.img);
imagesize=size(image);
[X,Y,Z]=meshgrid(1:imagesize(1),1:imagesize(2),1:imagesize(3));
zero_idx=image(:)==0;
X(zero_idx)=[];
Y(zero_idx)=[];
Z(zero_idx)=[];


figure;
plot3D = scatter3(X(:),Y(:),Z(:),ones(numel(Z(:)),1)*1000,image(~zero_idx),'filled');
xlim([0 113])
ylim([0 137])
zlim([0 113])

image_data = imread(A.img);

figure
cmp=jet;
volshow(image,'Colormap',cmp)