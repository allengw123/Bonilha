clear
clc

%%%%%%%%%%%%%%%%%%%% ADJUST BELOW %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Enter in Paths
GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
NIFTIFOLDER_PATH='/media/bonilha/Elements/test_data/images';
XML_PATH = '/media/bonilha/Elements/test_data/labels.xlsx';

% Run option
set_ratio = [0.6 0.25 0.15]; % Ratio of Training, Testing, Validation
model = 3; % Either 2 for 2D or 3 for 3D
iterations = 1; % Number of repeated models

% Label XML options
label_column = 2; % Which column of excel file is labels

% 2D Parameters
slices = 28:85; % Indicies of Coronal slice
param.C2.optimizer = 'adam'; % Either adam or sgdm
param.C2.GLR = 0.01; % Global learning rate
param.C2.BS = 128; % Batch size
param.C2.EP = 30; % Epochs
param.C2.showgraph = 'training-progress'; % Either "training-process" or "none"

% 3D Parameters
param.C3.optimizer = 'adam'; % Either adam or sgdm
param.C3.GLR = 0.0001; % Global learning rate
param.C3.BS = 10; % Global learning rate
param.C3.EP = 20; % Global learning rate
param.C3.showgraph = 'training-progress'; % Global learning rate


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% DO NOT ADJUST CODE BELOW %%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Setup
% Add toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

% Load label xml file
label_xml = readtable(XML_PATH);

% Nifti files
nifti_files = dir(NIFTIFOLDER_PATH);

% Unique labels
label_types = unique(label_xml{:,label_column});

% Define CNN specific type variables
switch model
    case 2
        dim = 3;
        perm_order = [1 2 4 3];
    case 3
        perm_order = [1 2 3 5 4];
        dim = 4;
    otherwise
        error('Model option must equal either 2 (2D CNN) or 3 (3D CNN)')
end

% Allocate Variables
train_Dataset = [];
val_Dataset = [];
test_Dataset = [];

train_Label = [];
val_Label =[];
test_Label = [];


for l = 1:numel(label_types)
    if isnumeric(label_types(l))
        wk_group = num2str(label_types(l));
    else
        wk_group = label_types(l);
    end
    group_sbjs = label_xml{label_xml{:,2}== label_types(l),1};

    group_nifti_files_idx = cellfun(@(x) any(contains(x,group_sbjs)),{nifti_files.name});
    group_nifti_files = nifti_files(group_nifti_files_idx);

    disp(['Loading nifti from group ... ',wk_group,' [',num2str(numel(group_nifti_files)),' total files]'])

    group_images = cell(numel(group_nifti_files),1);
    parfor n = 1:numel(group_nifti_files)
        temp_nii = load_nii(fullfile(group_nifti_files(n).folder,group_nifti_files(n).name));
        if model == 2
            temp_nii.img = temp_nii.img(:,:,slices)
        end
        group_images{n} = temp_nii.img;
    end

    group_images = cat(dim,group_images{:});

    % Create set
    [group_train,group_test,group_val] = createset(model,group_images,set_ratio);

    % Combine sets
    train_Dataset = cat(dim,train_Dataset,group_train);
    val_Dataset = cat(dim,val_Dataset,group_val);
    test_Dataset = cat(dim,test_Dataset,group_test);

    train_Label = [train_Label;ones(size(group_train,dim),1)*l];
    val_Label = [val_Label;ones(size(group_val,dim),1)*l];
    test_Label = [test_Label;ones(size(group_test,dim),1)*l];
end

% Delete old variables
clear *group*

% Permute Correct Dimension
train_Dataset = permute(train_Dataset,perm_order);
val_Dataset = permute(val_Dataset,perm_order);
test_Dataset = permute(test_Dataset,perm_order);

% Reset parpool
if ~isempty(gcp('nocreate'))
    pool = gcp('nocreate');
    delete(pool)
end

% Display load complete
disp('Images Loaded')
%% Train the network

for i = 1:iterations
    [net{i},acc{i},confmat{i}] = cnn(model, ...
        train_Dataset, ...
        train_Label, ...
        test_Dataset, ...
        test_Label, ...
        val_Dataset, ...
        val_Label, ...
        param, ...
        label_types);
end

%% Function
function [net,acc,con]=cnn(model,trainData,trainRes,testData,testRes,valData,valRes,param,label_types)

switch model
    case 2
        input_dim = size(trainData,[1 2 3]);
        param = param.C2;

        layers = [
            imageInputLayer(input_dim)

            convolution2dLayer(3,8,'Padding','same')
            batchNormalizationLayer
            reluLayer

            maxPooling2dLayer(2,'Stride',2)

            convolution2dLayer(3,16,'Padding','same')
            batchNormalizationLayer
            reluLayer

            maxPooling2dLayer(2,'Stride',2)

            convolution2dLayer(3,32,'Padding','same')
            batchNormalizationLayer
            reluLayer

            fullyConnectedLayer(numel(unique(trainRes)))
            softmaxLayer
            classificationLayer];
    case 3
        param = param.C3;
        input_dim = size(trainData,[1 2 3 4]);

        layers = [
            image3dInputLayer(input_dim)
            convolution3dLayer(3,8,'Padding','same')
            batchNormalizationLayer()
            reluLayer
            maxPooling3dLayer(2,'Stride',2)

            convolution3dLayer(3,16,'Padding','same')
            batchNormalizationLayer()
            reluLayer
            maxPooling3dLayer(2,'Stride',2)

            convolution3dLayer(3,32,'Padding','same')
            batchNormalizationLayer()
            reluLayer
            fullyConnectedLayer(numel(unique(trainRes)))
            softmaxLayer()
            classificationLayer()];
end

options = trainingOptions(param.optimizer, ...
    'InitialLearnRate',param.GLR, ...
    'MaxEpochs',param.EP, ...
    'ValidationData',{valData,categorical(valRes)}, ...
    'Plots',param.showgraph,...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment','multi-gpu', ...
    'MiniBatchSize',param.BS);

%%%%%%% Train on regular data %%%%%
net=trainNetwork(trainData,categorical(trainRes),layers,options);

[acc,con] = evalCNN(model,net,testData,testRes,label_types);
end


function [acc,con] = evalCNN(model,net,testData,testRes,label_types)

% Test on regular response
YPred_test = [];
if model == 2
    parfor i = 1:size(testData,4)
        YPred_test = [YPred_test; classify(net,testData(:,:,:,i))];
    end
else
    parfor i = 1:size(testData,5)
        YPred_test = [YPred_test; classify(net,testData(:,:,:,:,i))];
    end
end

YTest = categorical(testRes);
acc = sum(YPred_test == YTest)/numel(YTest);

[con.C, con.order]= confusionmat(YTest,YPred_test);
end

function [train_img,test_img,val_img] = createset(model,img,ratio)
if model == 2
    [trainIdx,testIdx,valIdx] = dividerand(size(img,3),ratio(1),ratio(2),ratio(3));

    train_img = img(:,:,trainIdx);
    test_img = img(:,:,testIdx);
    val_img = img(:,:,valIdx);
else
    [trainIdx,testIdx,valIdx] = dividerand(size(img,4),ratio(1),ratio(2),ratio(3));

    train_img = img(:,:,:,trainIdx);
    test_img = img(:,:,:,testIdx);
    val_img = img(:,:,:,valIdx);
end
end