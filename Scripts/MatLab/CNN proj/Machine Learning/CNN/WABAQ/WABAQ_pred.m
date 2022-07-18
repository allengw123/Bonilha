%% CNN MODEL

% githubpath='C:\Users\bonilha\Documents\GitHub\Bonilha';
githubpath='/home/bonilha/Documents/GitHub/Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
input_dir = '/media/bonilha/AllenProj/PatientData/WABAQ';

info_sheet = readtable(fullfile(input_dir,'POLAR_Data_forAllenNoControls.xlsx'));

%% Setup for CNN model

% Load lesion maps
sbjs = {dir(fullfile(input_dir,'M*')).name}';
WABAQ_score = [];
lesions = [];
cd(input_dir)
for s = 1:numel(sbjs)
    temp_lesion = load(sbjs{s},'lesion');
    lesions = cat(4,lesions,temp_lesion.lesion.dat);
    WABAQ_score = [WABAQ_score round(info_sheet.WAB_AQ(info_sheet.ID == str2num(cell2mat(extractBetween(sbjs{s},'M','.mat')))))];
end


[trainIdx,testIdx,valIdx] = dividerand(numel(sbjs),0.6,0.25,0.15);        
    
% Seperate datasets
trainDataset = permute(lesions(:,:,:,trainIdx),[1 2 3 5 4]);
trainLabels = WABAQ_score(trainIdx);

testDataset = permute(lesions(:,:,:,testIdx),[1 2 3 5 4]);
testLabels = WABAQ_score(testIdx);

valDataset = permute(lesions(:,:,:,valIdx),[1 2 3 5 4]);
valLabels = WABAQ_score(valIdx);


% Parameters for the network
imageSize = size(trainDataset,[1 2 3 4]);

layers = [
    image3dInputLayer(imageSize)

    convolution3dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling3dLayer(2,'Stride',2)

    convolution3dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling3dLayer(2,'Stride',2)

    convolution3dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(100)
    fullyConnectedLayer(100)
    fullyConnectedLayer(100)
    fullyConnectedLayer(100)
    fullyConnectedLayer(100)
    fullyConnectedLayer(100)

    fullyConnectedLayer(1)

    regressionLayer];

miniBatchSize = 30;
validationFrequency = 10;

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{valDataset,valLabels'}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',true,...
    'ExecutionEnvironment','multi-gpu');


%%%%%%%%%%%% Train the network
net = trainNetwork(trainDataset,trainLabels',layers,options);
YPredicted = predict(net,testDataset);
predictionError = testLabels' - YPredicted;

thr = 1;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(YValidation);

accuracy = numCorrect/numValidationImages
