%% CNN MODEL (Confounding Factor)
clear
clc

githubpath = '/home/bonilha/Documents/GitHub/Bonilha';
% githubpath = 'C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

save_path='/media/bonilha/AllenProj/PatientData/smallSet';

%% Setup for CNN model

% Load Residual img
load '/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/AgeRegress/disease_label.mat'
load '/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/AgeRegress/residual_imgs.mat'
load '/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/AgeRegress/side_label.mat'
load '/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/AgeRegress/subjects.mat'


net = [];
acc = [];
confmat = [];

for iter = 1:100
    
    display(['Running iteration ',num2str(iter)])
    
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
        trainLabels = [trainLabels;ones(numel(trainIdx)*58,1)*d_groups(d)];
        
        testDataset = [testDataset;d_img(testIdx)];
        testLabels = [testLabels;ones(numel(testIdx)*58,1)*d_groups(d)];
        
        valDataset = [valDataset;d_img(valIdx)];
        valLabels = [valLabels;ones(numel(valIdx)*58,1)*d_groups(d)];
    end
    
    % Reshape images
    trainDataset=permute(cat(3,trainDataset{:}),[1 2 4 3]);
    testDataset=permute(cat(3,testDataset{:}),[1 2 4 3]);
    valDataset=permute(cat(3,valDataset{:}),[1 2 4 3]);
    

    %%%%%%%%%%%% Train the network
    [net.reg{iter},acc.reg{iter},confmat.reg{iter}] = runcnnFC(trainDataset,trainLabels,testDataset,testLabels,valDataset,valLabels);
    net.sbjs{iter} = [{trainIdx},{testIdx},{valIdx}];
    
    [net.suff{iter},acc.shuff{iter},confmat.shuff{iter}] = runcnnFC(trainDataset,trainLabels(randperm(numel(trainLabels))),testDataset,testLabels(randperm(numel(testLabels))),valDataset,valLabels(randperm(numel(valLabels))));
end

save_path = 'F:\CNN output\2D_CNN\MATLAB\AgeRegress';
save(fullfile(save_path,'AgeRegress_GM_ADTLEHC_CNN.mat'),'net','acc','confmat','-v7.3')


    
%% Funtions
    
function [net,acc,con]=runcnnFC(trainData,trainRes,testData,testRes,valData,valRes)
    
    
    % Parameters for the network
    layers = [
        imageInputLayer([113 137 1])
        
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
    
    options = trainingOptions('sgdm', ...  %stochastic gradient descent with momentum(SGDM) optimizer
        'InitialLearnRate',0.01, ...
        'MaxEpochs',30, ...  % Default is 30
        'ValidationData',{valData,categorical(valRes)}, ...
        'Verbose',false, ... %Indicator to display training progress information in the command window
        'Plots','none',...
        'Shuffle','every-epoch', ...
        'ExecutionEnvironment','multi-gpu');
        
    %%%%%%% Train on regular data %%%%%
    net=trainNetwork(trainData,categorical(trainRes),layers,options);

    
    % Test on regular response
    YPred_test = classify(net,testData);
    YTest = categorical(testRes);
    acc = sum(YPred_test == YTest)/numel(YTest);
    [con.C, con.order]= confusionmat(YTest,YPred_test);
end
   