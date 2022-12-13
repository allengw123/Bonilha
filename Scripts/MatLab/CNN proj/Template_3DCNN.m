%3D CNN MODEL
clear
clc

% Enter in Paths
GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
NIFTIFOLDER_PATH='/media/bonilha/Elements/MasterSet_old/mat2nii_savefolder';


%% Setup 

% Add toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')



%% get volume data

% look for Controls nifti files
controlfiles = dir(fullfile(NIFTIFOLDER_PATH,"Controls","**",['*_',matter,'.nii']));
controls = [];
for i = 1:length(controlfiles)
    controls = [controls;{fullfile(controlfiles(i).folder,controlfiles(i).name)}];
end

% look for Patients nifti files
patientfiles = dir(fullfile(NIFTIFOLDER_PATH,"Patients","**",['*_',matter,'.nii']));
patients = [];
for i = 1:length(patientfiles)
    patients = [patients;{fullfile(patientfiles(i).folder,patientfiles(i).name)}];
end

% Load volume
[Patient_GM, Patient_GM_names] = get_volume_data(patients,true);
[Control_GM, Control_GM_names] = get_volume_data(controls,true);

L_idx = strcmp(cellfun(@(x) x(5),Patient_GM_names),"L");
R_idx =  strcmp(cellfun(@(x) x(5),Patient_GM_names),"R");
%% Disease Prediction setup (ALL)

% Create set
[patient_train,patient_test,patient_val] = createset(Patient_GM,[0.6 0.25 0.15]);
[control_train,control_test,control_val] = createset(Control_GM,[0.6 0.25 0.15]);

% Combine set [patient 1, control 2]
trainDataset = cat(4,patient_train,control_train);
trainLabels = [ones(size(patient_train,4),1); ones(size(control_train,4),1)*2];
trainDataset = permute(trainDataset,[1 2 3 5 4]);

testDataset = cat(4,patient_test,control_test);
testLabels = [ones(size(patient_test,4),1); ones(size(control_test,4),1)*2];
testDataset = permute(testDataset,[1 2 3 5 4]);

valDataset = cat(4,patient_val,control_val);
valLabels = [ones(size(patient_val,4),1); ones(size(control_val,4),1)*2];
valDataset = permute(valDataset,[1 2 3 5 4]);

param.GLR = 0.0001;
param.BS = 10;
param.DO = 0.6;
param.EP = 20;
param.showgraph = 'training-progress';

% Train the network
[net,acc,confmat] = runcnnFC_new(trainDataset,trainLabels,testDataset,testLabels,valDataset,valLabels,param);
acc
confmat.C

%% Disease laterality setup
net = [];
acc = [];
confmat = [];

for i = 1
    % Create set
    [patient_train_L,patient_test_L,patient_val_L] = createset(Patient_GM(:,:,:,L_idx),[0.6 0.25 0.15]);
    [patient_train_R,patient_test_R,patient_val_R] = createset(Patient_GM(:,:,:,R_idx),[0.6 0.25 0.15]);
    [control_train,control_test,control_val] = createset(Control_GM,[0.6 0.25 0.15]);

    % Combine set [patient L 1, patient R 2, control 3]
    trainDataset = cat(4,patient_train_L,patient_train_R,control_train);
    trainLabels = [ones(size(patient_train_L,4),1);ones(size(patient_train_R,4),1)*2 ; ones(size(control_train,4),1)*3];
    trainDataset = permute(trainDataset,[1 2 3 5 4]);

    testDataset = cat(4,patient_test_L,patient_test_R,control_test);
    testLabels = [ones(size(patient_test_L,4),1);ones(size(patient_test_R,4),1)*2 ; ones(size(control_test,4),1)*3];
    testDataset = permute(testDataset,[1 2 3 5 4]);

    valDataset = cat(4,patient_val_L,patient_val_R,control_val);
    valLabels = [ones(size(patient_val_L,4),1);ones(size(patient_val_R,4),1)*2 ; ones(size(control_val,4),1)*3];
    valDataset = permute(valDataset,[1 2 3 5 4]);

    % Parameters [0.816455696202532	0.835443037974684	0.740506329113924 0.800632911392405	0.762658227848101	0.746835443037975	0.822784810126582 0.803797468354430	0.784810126582278	0.746835443037975]
    param.GLR = 0.000001;
    param.BS = 20;
    param.DO = 0.4;
    param.EP = 20;
    param.showgraph = 'training-progress';

    % Train the network
    [net{i},acc{i},confmat{i}] = runcnnFC_new(trainDataset,trainLabels,testDataset,testLabels,valDataset,valLabels,param);
    acc{i}
    confmat{i}.C
end

%% Disease laterality setup (Only patient)
net = [];
acc = [];
confmat = [];

for i = 1:10
    % Create set
    [patient_train_L,patient_test_L,patient_val_L] = createset(Patient_GM(:,:,:,L_idx),[0.6 0.25 0.15]);
    [patient_train_R,patient_test_R,patient_val_R] = createset(Patient_GM(:,:,:,R_idx),[0.6 0.25 0.15]);

    % Combine set [patient L 1, patient R 2, control 3]
    trainDataset = cat(4,patient_train_L,patient_train_R);
    trainLabels = [ones(size(patient_train_L,4),1);ones(size(patient_train_R,4),1)*2];
    trainDataset = permute(trainDataset,[1 2 3 5 4]);

    testDataset = cat(4,patient_test_L,patient_test_R);
    testLabels = [ones(size(patient_test_L,4),1);ones(size(patient_test_R,4),1)*2];
    testDataset = permute(testDataset,[1 2 3 5 4]);

    valDataset = cat(4,patient_val_L,patient_val_R);
    valLabels = [ones(size(patient_val_L,4),1);ones(size(patient_val_R,4),1)*2];
    valDataset = permute(valDataset,[1 2 3 5 4]);

    % Parameters
    param.GLR = 0.00001;
    param.BS = 10;
    param.DO = 0.6;
    param.EP = 20;
    param.showgraph = 'none';
    param.verbose = true;

    % Train the network
    [net{i},acc{i},confmat{i}] = runcnnFC_new(trainDataset,trainLabels,testDataset,testLabels,valDataset,valLabels,param);
    acc{i}
    confmat{i}.C
end

temp  =cellfun(@(x) x.C,confmat,'UniformOutput',false);
sum(cat(3,temp{:}),3)
%% Function


function [net,acc,con]=runcnnFC_new(trainData,trainRes,testData,testRes,valData,valRes,param)

input_dim = size(trainData);

layers = [
    image3dInputLayer(input_dim(1:3))
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
    dropoutLayer(param.DO)
    softmaxLayer()
    classificationLayer()];

options = trainingOptions('adam', ...  %stochastic gradient descent with momentum(SGDM) optimizer
    'InitialLearnRate',param.GLR, ...
    'LearnRateSchedule','piecewise', ...
    'MaxEpochs',param.EP, ...  % Default is 30
    'ValidationData',{valData,categorical(valRes)}, ...
    'Verbose',param.verbose, ... %Indicator to display training progress information in the command window
    'Plots',param.showgraph,...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment','multi-gpu', ...
    'MiniBatchSize',param.BS);

%%%%%%% Train on regular data %%%%%
net=trainNetwork(trainData,categorical(trainRes),layers,options);


% Test on regular response
YPred_test = [];
for i = 1:size(testData,5)
    YPred_test = [YPred_test; classify(net,testData(:,:,:,:,i))];
end
YTest = categorical(testRes);
acc = sum(YPred_test == YTest)/numel(YTest);
[con.C, con.order]= confusionmat(YTest,YPred_test);
end

function [X, X_names,N] = get_volume_data(ff,mid_extract)
count = 1;
for i = 1:numel(ff)
    N = load_nii(ff{i});
    if mid_extract
        X(:,:,:,count) = N.img(:,:,28:85);
    else
        X(:,:,:,count) = N.img();
    end
    [~,name,~] = fileparts(ff{i});
    X_names{count,1} =name;
    count = count + 1;
end
end

function [train_img,test_img,val_img] = createset(img,ratio)

[trainIdx,testIdx,valIdx] = dividerand(size(img,4),ratio(1),ratio(2),ratio(3));

train_img = img(:,:,:,trainIdx);
test_img = img(:,:,:,testIdx);
val_img = img(:,:,:,valIdx);

end
