%% CNN MODEL (Confounding Factor)
clear
clc

githubpath = '/home/bonilha/Documents/GitHub/Bonilha';
% githubpath = 'C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress';
cd(PatientData)

% Load net
nets = load('/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress/AgeRegress_GM_ADTLEHC_CNN.mat');

% Load Regress
load('/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress/residual_imgs.mat');

% Load subjects
load('/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress/subjects.mat');

% Load Disease
load('/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress/disease_label.mat');

% Load Side
load('/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress/side_label.mat');

%% Disease Prediction [Healthy (1) vs TLE (2) vs AD (3)]

% Calculate # of groups
d_groups = [1,2,3];

trainDataset = [];
trainLabels = [];
testDataset = [];
testLabels = [];
valDataset = [];
valLabels = [];

for d = 1:numel(d_groups)
    d_idx = disease==d_groups(d);
    d_img = reshaped_residuals(d_idx);

    [trainIdx,testIdx,valIdx] = dividerand(numel(d_img),0.6,0.25,0.15);        

    % Seperate datasets
    trainDataset = [trainDataset;d_img(trainIdx)];
    trainLabels = [trainLabels;ones(numel(trainIdx),1)*d_groups(d)];
    
    testDataset = [testDataset;d_img(testIdx)];
    testLabels = [testLabels;ones(numel(testIdx),1)*d_groups(d)];
    
    valDataset = [valDataset;d_img(valIdx)];
    valLabels = [valLabels;ones(numel(valIdx),1)*d_groups(d)];
end


% Reshape images
trainDataset=permute(cat(4,trainDataset{:}),[1 2 3 5 4]);
testDataset=permute(cat(4,testDataset{:}),[1 2 3 5 4]);
valDataset=permute(cat(4,valDataset{:}),[1 2 3 5 4]);

%% Disease vs Non Disease [Healthy (1) vs TLE (2) vs AD (3)]

% Calculate # of groups
d_groups = [1,2,3];

trainDataset = [];
trainLabels = [];
testDataset = [];
testLabels = [];
valDataset = [];
valLabels = [];


for d = 1:numel(d_groups)
    d_idx = disease==d_groups(d);
    d_img = reshaped_residuals(d_idx);

    if d_groups(d) == 1
        d_groups_label = 1;
    else
        d_groups_label = 2;
    end

    [trainIdx,testIdx,valIdx] = dividerand(numel(d_img),0.6,0.25,0.15);        

    % Seperate datasets
    trainDataset = [trainDataset;d_img(trainIdx)];
    trainLabels = [trainLabels;ones(numel(trainIdx),1)*d_groups_label];
    
    testDataset = [testDataset;d_img(testIdx)];
    testLabels = [testLabels;ones(numel(testIdx),1)*d_groups_label];
    
    valDataset = [valDataset;d_img(valIdx)];
    valLabels = [valLabels;ones(numel(valIdx),1)*d_groups_label];
end



% Reshape images
trainDataset=permute(cat(4,trainDataset{:}),[1 2 3 5 4]);
testDataset=permute(cat(4,testDataset{:}),[1 2 3 5 4]);
valDataset=permute(cat(4,valDataset{:}),[1 2 3 5 4]);
%% TLE side (ALL DISEASE)

% Calculate # of groups
d_groups = [1,2,3];

trainDataset = [];
trainLabels = [];
testDataset = [];
testLabels = [];
valDataset = [];
valLabels = [];
% Side
for d = 1:numel(d_groups)
    d_idx = disease==d_groups(d);
    d_img = reshaped_residuals(d_idx);
    s = side(d_idx);


    [trainIdx,testIdx,valIdx] = dividerand(numel(d_img),0.6,0.25,0.15);        

    % Seperate datasets
    trainDataset = [trainDataset;d_img(trainIdx)];
    trainLabels = [trainLabels;s(trainIdx)];
    
    testDataset = [testDataset;d_img(testIdx)];
    testLabels = [testLabels;s(testIdx)];
    
    valDataset = [valDataset;d_img(valIdx)];
    valLabels = [valLabels;s(valIdx)];
end


% Reshape images
trainDataset=permute(cat(4,trainDataset{:}),[1 2 3 5 4]);
testDataset=permute(cat(4,testDataset{:}),[1 2 3 5 4]);
valDataset=permute(cat(4,valDataset{:}),[1 2 3 5 4]);

%% TLE side (Just TLE)

trainDataset = [];
trainLabels = [];
testDataset = [];
testLabels = [];
valDataset = [];
valLabels = [];

% Side
d_idx = disease==2;
d_img = reshaped_residuals(d_idx);
s = side(d_idx);


[trainIdx,testIdx,valIdx] = dividerand(numel(d_img),0.6,0.25,0.15);        

% Seperate datasets
trainDataset = [trainDataset;d_img(trainIdx)];
trainLabels = [trainLabels;s(trainIdx)];

testDataset = [testDataset;d_img(testIdx)];
testLabels = [testLabels;s(testIdx)];

valDataset = [valDataset;d_img(valIdx)];
valLabels = [valLabels;s(valIdx)];


% Reshape images
trainDataset=permute(cat(4,trainDataset{:}),[1 2 3 5 4]);
testDataset=permute(cat(4,testDataset{:}),[1 2 3 5 4]);
valDataset=permute(cat(4,valDataset{:}),[1 2 3 5 4]);
%% Run CNN

% Parameters
param.GLR = 0.0001;
param.BS = 10;
param.DO = 0.6;
param.EP = 20;

% Train the network
[new_net,new_regacc.reg,new_confmat.reg] = runcnnFC_new(trainDataset,trainLabels,testDataset,testLabels,valDataset,valLabels,param);
new_regacc.reg

[pre_net,pre_regacc.reg,pre_confmat.reg] = runcnnFC_PRE(nets,trainDataset,trainLabels,testDataset,testLabels,valDataset,valLabels,param);
pre_regacc.reg


%% Function


function [net,acc,con]=runcnnFC_new(trainData,trainRes,testData,testRes,valData,valRes,param)

     layers = [
        image3dInputLayer([113,137,58])
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
        'MaxEpochs',param.EP, ...  % Default is 30
        'ValidationData',{valData,categorical(valRes)}, ...
        'Verbose',false, ... %Indicator to display training progress information in the command window
        'Plots','training-progress',...
        'Shuffle','every-epoch', ...
        'ExecutionEnvironment','multi-gpu', ...
        'OutputNetwork','best-validation-loss',...
        'MiniBatchSize',param.BS);
        
    %%%%%%% Train on regular data %%%%%
    net=trainNetwork(trainData,categorical(trainRes),layers,options);

    
    % Test on regular response
    YPred_test = classify(net,testData);
    YTest = categorical(testRes);
    acc = sum(YPred_test == YTest)/numel(YTest);
    [con.C, con.order]= confusionmat(YTest,YPred_test);
end


function [net,acc,con]=runcnnFC_PRE(nets,trainData,trainRes,testData,testRes,valData,valRes,param)
    
    % Load most accuracy net for PreTrain
    [V,I] = max([nets.acc.reg{:}]);
    p_net = nets.net.reg{I};
    
    layers = [
        image3dInputLayer([113,137,58])
        convolution3dLayer(3,8,'Padding','same', ...
            'Weights',permute(repmat(p_net.Layers(2,1).Weights,[1 1 1 1 3]),[1 2 5 3 4]), ...
            'Bias',permute(p_net.Layers(2,1).Bias,[1 2 4 3]))
        batchNormalizationLayer('TrainedMean',permute(p_net.Layers(3,1).TrainedMean,[1 2 4 3]), ...
            'TrainedVariance',permute(p_net.Layers(3,1).TrainedVariance,[1 2 4 3]),...
            'Scale',permute(p_net.Layers(3,1).Scale,[1 2 4 3]),...
            'Offset',permute(p_net.Layers(3,1).Offset,[1 2 4 3]))
        reluLayer
        maxPooling3dLayer(2,'Stride',2)
    
        convolution3dLayer(3,16,'Padding','same', ...
            'Weights',permute(repmat(p_net.Layers(6,1).Weights,[1 1 1 1 3]),[1 2 5 3 4]), ...
            'Bias',permute(p_net.Layers(6,1).Bias,[1 2 4 3]))
        batchNormalizationLayer('TrainedMean',permute(p_net.Layers(7,1).TrainedMean,[1 2 4 3]), ...
            'TrainedVariance',permute(p_net.Layers(7,1).TrainedVariance,[1 2 4 3]),...
            'Scale',permute(p_net.Layers(7,1).Scale,[1 2 4 3]),...
            'Offset',permute(p_net.Layers(7,1).Offset,[1 2 4 3]))
        reluLayer
        maxPooling3dLayer(2,'Stride',2)
    
        convolution3dLayer(3,32,'Padding','same', ...
            'Weights',permute(repmat(p_net.Layers(10,1).Weights,[1 1 1 1 3]),[1 2 5 3 4]), ...
            'Bias',permute(p_net.Layers(10,1).Bias,[1 2 4 3]))
        batchNormalizationLayer('TrainedMean',permute(p_net.Layers(11,1).TrainedMean,[1 2 4 3]), ...
            'TrainedVariance',permute(p_net.Layers(11,1).TrainedVariance,[1 2 4 3]),...
            'Scale',permute(p_net.Layers(11,1).Scale,[1 2 4 3]),...
            'Offset',permute(p_net.Layers(11,1).Offset,[1 2 4 3]))
        reluLayer
        fullyConnectedLayer(numel(unique(trainRes)),'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        dropoutLayer(param.DO)

        softmaxLayer()
        classificationLayer()];



    options = trainingOptions('adam', ...  %stochastic gradient descent with momentum(SGDM) optimizer
        'InitialLearnRate',param.GLR/10, ...
        'MaxEpochs',param.EP, ...  % Default is 30
        'ValidationData',{valData,categorical(valRes)}, ...
        'Verbose',false, ... %Indicator to display training progress information in the command window
        'Plots','training-progress',...
        'Shuffle','every-epoch', ...
        'ExecutionEnvironment','multi-gpu', ...
        'OutputNetwork','best-validation-loss',...
        'MiniBatchSize',param.BS);
        
    %%%%%%% Train on regular data %%%%%
    net=trainNetwork(trainData,categorical(trainRes),layers,options);

    
    % Test on regular response
    YPred_test = classify(net,testData);
    YTest = categorical(testRes);
    acc = sum(YPred_test == YTest)/numel(YTest);
    [con.C, con.order]= confusionmat(YTest,YPred_test);
end