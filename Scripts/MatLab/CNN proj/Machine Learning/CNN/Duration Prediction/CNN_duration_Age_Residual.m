%% CNN MODEL (Confounding Factor)
clear
clc

githubpath = '/home/bonilha/Documents/GitHub/Bonilha';
% githubpath = 'C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='/media/bonilha/AllenProj/PatientData/disease_dur/age_reg';
cd(PatientData)

% Load net
nets = load('/media/bonilha/AllenProj/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress/AgeRegress_GM_ADTLEHC_CNN.mat');

% Load Regress
regress = load('/media/bonilha/AllenProj/PatientData/disease_dur/age_reg/calculated_reg.mat');

% Load xml
info = readtable('/media/bonilha/AllenProj/PatientData/disease_dur/Patientinfo.xlsx');
%% Setup for CNN model

% Load most accuracy net for PreTrain
[V,I] = max([nets.acc.reg{:}]);
p_net = nets.net.reg{I};

% Classify Epilepsy length class
e_len = [];
rm_idx = [];
for i=1:numel(regress.sbj)
    e_length = info.DurationInYears(strcmp(info.NewID,regress.sbj{i}));
    if isempty(e_length)
        rm_idx = [rm_idx;i];
    end
    e_len = [e_len ;e_length];
end
Q = quantile(e_len,[0.33, 0.66,1]);
e_cat = sum([e_len<=Q(1) e_len<=Q(2) e_len<=Q(3)],2);

% Remove sbjs that don't have e_duration
for i = 1:numel(rm_idx)
    regress.g_age(rm_idx(i))=[];
    regress.reshaped_residuals(rm_idx(i))=[];
    regress.sbj(rm_idx(i))=[];
end

% Modify Pretrain net
lgraph = layerGraph(p_net);
lgraph = removeLayers(lgraph,{'fc','softmax','classoutput'});

layers = [
    fullyConnectedLayer(3,'Name','fc1','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
    dropoutLayer(0.8)
    fullyConnectedLayer(3,'Name','fc2','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
    softmaxLayer()
     classificationLayer()
     ];
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,'relu_3','fc1');


net = [];
acc = [];
confmat = [];

for iter = 1
    
    display(['Running iteration ',num2str(iter)]) 

    imgs = regress.reshaped_residuals;
        
    [trainIdx,testIdx,valIdx] = dividerand(numel(imgs),0.6,0.25,0.15);        

    % Seperate datasets
    trainDataset = imgs(trainIdx);
    trainLabels = repmat(e_cat(trainIdx),[1 58])';
    trainLabels = trainLabels(:);

    
    testDataset = imgs(testIdx);
    testLabels = repmat(e_cat(testIdx),[1 58])';
    testLabels = testLabels(:);
    
    valDataset = imgs(valIdx);
    valLabels = repmat(e_cat(valIdx),[1 58])';
    valLabels = valLabels(:);

    
    % Reshape images
    trainDataset=permute(cat(3,trainDataset{:}),[1 2 4 3]);
    testDataset=permute(cat(3,testDataset{:}),[1 2 4 3]);
    valDataset=permute(cat(3,valDataset{:}),[1 2 4 3]);
    

    %%%%%%%%%%%% Train the network
    [net.reg{iter},acc.reg{iter},confmat.reg{iter}] = runcnnFC(lgraph,trainDataset,trainLabels,testDataset,testLabels,valDataset,valLabels);
    
    %[net.suff{iter},acc.shuff{iter},confmat.shuff{iter}] = runcnnFC(lgraph,trainDataset,trainLabels(randperm(numel(trainLabels))),testDataset,testLabels(randperm(numel(testLabels))),valDataset,valLabels(randperm(numel(valLabels))));
end


    
%% Funtions
    
function [net,acc,con]=runcnnFC(layers,trainData,trainRes,testData,testRes,valData,valRes)
    
    
    options = trainingOptions('sgdm', ...  %stochastic gradient descent with momentum(SGDM) optimizer
        'InitialLearnRate',0.01, ...
        'MaxEpochs',60, ...  % Default is 30
        'ValidationData',{valData,categorical(valRes)}, ...
        'Verbose',true, ... %Indicator to display training progress information in the command window
        'Plots','training-progress',...
        'Shuffle','every-epoch', ...
        'ExecutionEnvironment','multi-gpu', ...
        'MiniBatchSize',128*2 ...
        );
        
    %%%%%%% Train on regular data %%%%%
    net=trainNetwork(trainData,categorical(trainRes),layers,options);

    
    % Test on regular response
    YPred_test = classify(net,testData);
    YTest = categorical(testRes);
    acc = sum(YPred_test == YTest)/numel(YTest);
    [con.C, con.order]= confusionmat(YTest,YPred_test);
end
   