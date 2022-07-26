%% CNN MODEL

% githubpath='C:\Users\bonilha\Documents\GitHub\Bonilha';
githubpath='/home/bonilha/Documents/GitHub/Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
input_dir = '/media/bonilha/AllenProj/PatientData/WABAQ';

info_sheet = readtable(fullfile(input_dir,'POLAR_Data_forAllenNoControls.xlsx'));


%% Load Pretrained 
lgraph = resnet18TL3Dfunction();
inputSize = lgraph.Layers(1).InputSize;
lgraph = removeLayers(lgraph,{'fc1000','prob','ClassificationLayer_predictions'});
%%

lgraph = replaceLayer(lgraph,'data',image3dInputLayer([224 224 224 2]));
%% Setup for CNN model

% Load lesion maps
sbjs = {dir(fullfile(input_dir,'M*')).name}';
WABAQ_score = [];
lesions = [];
cd(input_dir)
for s = 1:numel(sbjs)
    temp_lesion = load(sbjs{s},'lesion');
    lesions = cat(4,lesions,imresize3(temp_lesion.lesion.dat,inputSize(1:3)));
    WABAQ_score = [WABAQ_score round(info_sheet.WAB_AQ(info_sheet.ID == str2num(cell2mat(extractBetween(sbjs{s},'M','.mat')))))];
end
%%

[trainIdx,testIdx,valIdx] = dividerand(numel(sbjs),0.6,0.25,0.15);        
    
% Seperate datasets
trainDataset = permute(lesions(:,:,:,trainIdx),[1 2 3 5 4]);
trainLabels = WABAQ_score(trainIdx);

testDataset = permute(lesions(:,:,:,testIdx),[1 2 3 5 4]);
testLabels = WABAQ_score(testIdx);

valDataset = permute(lesions(:,:,:,valIdx),[1 2 3 5 4]);
valLabels = WABAQ_score(valIdx);

%%
miniBatchSize = 3;
validationFrequency = 50;

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{valDataset,valLabels'}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',true,...
    'ExecutionEnvironment','multi-gpu');


%%%%%%%%%%%% Train the network
net = trainNetwork(trainDataset,trainLabels',lgraph,options);

%%
YPredicted = [];
for i=1:size(testDataset,5)
    YPredicted = [YPredicted;predict(net,testDataset(:,:,:,i))];
end

predictionError = testLabels' - YPredicted;

thr = 1;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(YValidation);

accuracy = numCorrect/numValidationImages

%%
function lgraph = resnet18TL3Dfunction()
params = load("~/Downloads/params.mat");
lgraph = layerGraph();

tempLayers = [
    image3dInputLayer([224 224 224 1],"Name","data","Normalization","zscore","Mean",params.data.Mean,"StandardDeviation",params.data.StandardDeviation)
    convolution3dLayer([7 7 7],64,"Name","conv1","BiasLearnRateFactor",0,"Padding",[3 3 3;3 3 3],"Stride",[2 2 2],"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
    batchNormalizationLayer("Name","bn_conv1","Offset",params.bn_conv1.Offset,"Scale",params.bn_conv1.Scale,"TrainedMean",params.bn_conv1.TrainedMean,"TrainedVariance",params.bn_conv1.TrainedVariance)
    reluLayer("Name","conv1_relu")
    maxPooling3dLayer([3 3 3],"Name","pool1","Padding",[1 1 1;1 1 1],"Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([3 3 3],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res2a_branch2a.Bias,"Weights",params.res2a_branch2a.Weights)
    batchNormalizationLayer("Name","bn2a_branch2a","Offset",params.bn2a_branch2a.Offset,"Scale",params.bn2a_branch2a.Scale,"TrainedMean",params.bn2a_branch2a.TrainedMean,"TrainedVariance",params.bn2a_branch2a.TrainedVariance)
    reluLayer("Name","res2a_branch2a_relu")
    convolution3dLayer([3 3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res2a_branch2b.Bias,"Weights",params.res2a_branch2b.Weights)
    batchNormalizationLayer("Name","bn2a_branch2b","Offset",params.bn2a_branch2b.Offset,"Scale",params.bn2a_branch2b.Scale,"TrainedMean",params.bn2a_branch2b.TrainedMean,"TrainedVariance",params.bn2a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","res2a")
    reluLayer("Name","res2a_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([3 3 3],64,"Name","res2b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res2b_branch2a.Bias,"Weights",params.res2b_branch2a.Weights)
    batchNormalizationLayer("Name","bn2b_branch2a","Offset",params.bn2b_branch2a.Offset,"Scale",params.bn2b_branch2a.Scale,"TrainedMean",params.bn2b_branch2a.TrainedMean,"TrainedVariance",params.bn2b_branch2a.TrainedVariance)
    reluLayer("Name","res2b_branch2a_relu")
    convolution3dLayer([3 3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res2b_branch2b.Bias,"Weights",params.res2b_branch2b.Weights)
    batchNormalizationLayer("Name","bn2b_branch2b","Offset",params.bn2b_branch2b.Offset,"Scale",params.bn2b_branch2b.Scale,"TrainedMean",params.bn2b_branch2b.TrainedMean,"TrainedVariance",params.bn2b_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","res2b")
    reluLayer("Name","res2b_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],128,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2 2],"Bias",params.res3a_branch1.Bias,"Weights",params.res3a_branch1.Weights)
    batchNormalizationLayer("Name","bn3a_branch1","Offset",params.bn3a_branch1.Offset,"Scale",params.bn3a_branch1.Scale,"TrainedMean",params.bn3a_branch1.TrainedMean,"TrainedVariance",params.bn3a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([3 3 3],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Stride",[2 2 2],"Bias",params.res3a_branch2a.Bias,"Weights",params.res3a_branch2a.Weights)
    batchNormalizationLayer("Name","bn3a_branch2a","Offset",params.bn3a_branch2a.Offset,"Scale",params.bn3a_branch2a.Scale,"TrainedMean",params.bn3a_branch2a.TrainedMean,"TrainedVariance",params.bn3a_branch2a.TrainedVariance)
    reluLayer("Name","res3a_branch2a_relu")
    convolution3dLayer([3 3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res3a_branch2b.Bias,"Weights",params.res3a_branch2b.Weights)
    batchNormalizationLayer("Name","bn3a_branch2b","Offset",params.bn3a_branch2b.Offset,"Scale",params.bn3a_branch2b.Scale,"TrainedMean",params.bn3a_branch2b.TrainedMean,"TrainedVariance",params.bn3a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","res3a")
    reluLayer("Name","res3a_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([3 3 3],128,"Name","res3b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res3b_branch2a.Bias,"Weights",params.res3b_branch2a.Weights)
    batchNormalizationLayer("Name","bn3b_branch2a","Offset",params.bn3b_branch2a.Offset,"Scale",params.bn3b_branch2a.Scale,"TrainedMean",params.bn3b_branch2a.TrainedMean,"TrainedVariance",params.bn3b_branch2a.TrainedVariance)
    reluLayer("Name","res3b_branch2a_relu")
    convolution3dLayer([3 3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res3b_branch2b.Bias,"Weights",params.res3b_branch2b.Weights)
    batchNormalizationLayer("Name","bn3b_branch2b","Offset",params.bn3b_branch2b.Offset,"Scale",params.bn3b_branch2b.Scale,"TrainedMean",params.bn3b_branch2b.TrainedMean,"TrainedVariance",params.bn3b_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","res3b")
    reluLayer("Name","res3b_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2 2],"Bias",params.res4a_branch1.Bias,"Weights",params.res4a_branch1.Weights)
    batchNormalizationLayer("Name","bn4a_branch1","Offset",params.bn4a_branch1.Offset,"Scale",params.bn4a_branch1.Scale,"TrainedMean",params.bn4a_branch1.TrainedMean,"TrainedVariance",params.bn4a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([3 3 3],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Stride",[2 2 2],"Bias",params.res4a_branch2a.Bias,"Weights",params.res4a_branch2a.Weights)
    batchNormalizationLayer("Name","bn4a_branch2a","Offset",params.bn4a_branch2a.Offset,"Scale",params.bn4a_branch2a.Scale,"TrainedMean",params.bn4a_branch2a.TrainedMean,"TrainedVariance",params.bn4a_branch2a.TrainedVariance)
    reluLayer("Name","res4a_branch2a_relu")
    convolution3dLayer([3 3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res4a_branch2b.Bias,"Weights",params.res4a_branch2b.Weights)
    batchNormalizationLayer("Name","bn4a_branch2b","Offset",params.bn4a_branch2b.Offset,"Scale",params.bn4a_branch2b.Scale,"TrainedMean",params.bn4a_branch2b.TrainedMean,"TrainedVariance",params.bn4a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","res4a")
    reluLayer("Name","res4a_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([3 3 3],256,"Name","res4b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res4b_branch2a.Bias,"Weights",params.res4b_branch2a.Weights)
    batchNormalizationLayer("Name","bn4b_branch2a","Offset",params.bn4b_branch2a.Offset,"Scale",params.bn4b_branch2a.Scale,"TrainedMean",params.bn4b_branch2a.TrainedMean,"TrainedVariance",params.bn4b_branch2a.TrainedVariance)
    reluLayer("Name","res4b_branch2a_relu")
    convolution3dLayer([3 3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res4b_branch2b.Bias,"Weights",params.res4b_branch2b.Weights)
    batchNormalizationLayer("Name","bn4b_branch2b","Offset",params.bn4b_branch2b.Offset,"Scale",params.bn4b_branch2b.Scale,"TrainedMean",params.bn4b_branch2b.TrainedMean,"TrainedVariance",params.bn4b_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","res4b")
    reluLayer("Name","res4b_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],512,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2 2],"Bias",params.res5a_branch1.Bias,"Weights",params.res5a_branch1.Weights)
    batchNormalizationLayer("Name","bn5a_branch1","Offset",params.bn5a_branch1.Offset,"Scale",params.bn5a_branch1.Scale,"TrainedMean",params.bn5a_branch1.TrainedMean,"TrainedVariance",params.bn5a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([3 3 3],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Stride",[2 2 2],"Bias",params.res5a_branch2a.Bias,"Weights",params.res5a_branch2a.Weights)
    batchNormalizationLayer("Name","bn5a_branch2a","Offset",params.bn5a_branch2a.Offset,"Scale",params.bn5a_branch2a.Scale,"TrainedMean",params.bn5a_branch2a.TrainedMean,"TrainedVariance",params.bn5a_branch2a.TrainedVariance)
    reluLayer("Name","res5a_branch2a_relu")
    convolution3dLayer([3 3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res5a_branch2b.Bias,"Weights",params.res5a_branch2b.Weights)
    batchNormalizationLayer("Name","bn5a_branch2b","Offset",params.bn5a_branch2b.Offset,"Scale",params.bn5a_branch2b.Scale,"TrainedMean",params.bn5a_branch2b.TrainedMean,"TrainedVariance",params.bn5a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","res5a")
    reluLayer("Name","res5a_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([3 3 3],512,"Name","res5b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res5b_branch2a.Bias,"Weights",params.res5b_branch2a.Weights)
    batchNormalizationLayer("Name","bn5b_branch2a","Offset",params.bn5b_branch2a.Offset,"Scale",params.bn5b_branch2a.Scale,"TrainedMean",params.bn5b_branch2a.TrainedMean,"TrainedVariance",params.bn5b_branch2a.TrainedVariance)
    reluLayer("Name","res5b_branch2a_relu")
    convolution3dLayer([3 3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1;1 1 1],"Bias",params.res5b_branch2b.Bias,"Weights",params.res5b_branch2b.Weights)
    batchNormalizationLayer("Name","bn5b_branch2b","Offset",params.bn5b_branch2b.Offset,"Scale",params.bn5b_branch2b.Scale,"TrainedMean",params.bn5b_branch2b.TrainedMean,"TrainedVariance",params.bn5b_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","res5b")
    reluLayer("Name","res5b_relu")
    globalAveragePooling3dLayer("Name","pool5")
    fullyConnectedLayer(1000,"Name","fc1000","Bias",params.fc1000.Bias,"Weights",params.fc1000.Weights)
    softmaxLayer("Name","prob")
    classificationLayer("Name","ClassificationLayer_predictions","Classes",params.ClassificationLayer_predictions.Classes)];
lgraph = addLayers(lgraph,tempLayers);
% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"pool1","res2a_branch2a");
lgraph = connectLayers(lgraph,"pool1","res2a/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2b","res2a/in1");
lgraph = connectLayers(lgraph,"res2a_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"res2a_relu","res2b/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2b","res2b/in1");
lgraph = connectLayers(lgraph,"res2b_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"res2b_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"bn3a_branch1","res3a/in2");
lgraph = connectLayers(lgraph,"bn3a_branch2b","res3a/in1");
lgraph = connectLayers(lgraph,"res3a_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"res3a_relu","res3b/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2b","res3b/in1");
lgraph = connectLayers(lgraph,"res3b_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"res3b_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"bn4a_branch1","res4a/in2");
lgraph = connectLayers(lgraph,"bn4a_branch2b","res4a/in1");
lgraph = connectLayers(lgraph,"res4a_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"res4a_relu","res4b/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2b","res4b/in1");
lgraph = connectLayers(lgraph,"res4b_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"res4b_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"bn5a_branch2b","res5a/in1");
lgraph = connectLayers(lgraph,"bn5a_branch1","res5a/in2");
lgraph = connectLayers(lgraph,"res5a_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"res5a_relu","res5b/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2b","res5b/in1");
end
