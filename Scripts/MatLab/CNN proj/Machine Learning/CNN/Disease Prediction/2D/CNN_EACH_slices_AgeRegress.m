%% CNN MODEL (Confounding Factor)
clear
clc

githubpath = '/home/bonilha/Documents/GitHub/Bonilha';
% githubpath = 'C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

%% Setup for CNN model

% Load Residual img
ag_dir = '/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress';
load(fullfile(ag_dir,'disease_label.mat'));
load(fullfile(ag_dir,'residual_imgs.mat'));
load(fullfile(ag_dir,'side_label.mat'));
load(fullfile(ag_dir,'subjects.mat'));
%%
net = [];
acc = [];
confmat = [];

for iter = 1:10
    
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
        trainLabels = [trainLabels;ones(numel(trainIdx),1)*d_groups(d)];
        
        testDataset = [testDataset;d_img(testIdx)];
        testLabels = [testLabels;ones(numel(testIdx),1)*d_groups(d)];
        
        valDataset = [valDataset;d_img(valIdx)];
        valLabels = [valLabels;ones(numel(valIdx),1)*d_groups(d)];
    end
    
    % Reshape images
    cum_trainDataset=permute(cat(4,trainDataset{:}),[1 2 4 3]);
    cum_testDataset=permute(cat(4,testDataset{:}),[1 2 4 3]);
    cum_valDataset=permute(cat(4,valDataset{:}),[1 2 4 3]);
    
    for s = 1:size(cum_trainDataset,4)
        disp(['...sub ',num2str(s)])
        trainDataset = cum_trainDataset(:,:,:,s);
        testDataset = cum_testDataset(:,:,:,s);
        valDataset = cum_valDataset(:,:,:,s);
        %%%%%%%%%%%% Train the network
        [net.reg{iter,s},acc.reg{iter,s},confmat.reg{iter,s}] = runcnnFC(trainDataset,trainLabels,testDataset,testLabels,valDataset,valLabels);
        net.sbjs{iter,s} = [{trainIdx},{testIdx},{valIdx}];
        
        [net.suff{iter,s},acc.shuff{iter,s},confmat.shuff{iter,s}] = runcnnFC(trainDataset,trainLabels(randperm(numel(trainLabels))),testDataset,testLabels(randperm(numel(testLabels))),valDataset,valLabels(randperm(numel(valLabels))));
    end
end

save_path = '/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress';
save(fullfile(save_path,'AgeRegress_GM_ADTLEHC_CNN_EACHSLICE.mat'),'net','acc','confmat','-v7.3')

figure;
plot(mean(cell2mat(acc.reg),1))
    
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
        'ValidationData',{permute(valData,[1 2 4 3]),categorical(valRes)}, ...
        'Verbose',false, ... %Indicator to display training progress information in the command window
        'Plots','none',...
        'Shuffle','every-epoch', ...
        'ExecutionEnvironment','multi-gpu');
        
    %%%%%%% Train on regular data %%%%%
    net=trainNetwork(permute(trainData,[1 2 4 3]),categorical(trainRes),layers,options);

    
    % Test on regular response
    YPred_test = classify(net,permute(testData,[1 2 4 3]));
    YTest = categorical(testRes);
    acc = sum(YPred_test == YTest)/numel(YTest);
    [con.C, con.order]= confusionmat(YTest,YPred_test);
end
   