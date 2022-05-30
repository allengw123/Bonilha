%% CNN MODEL (Confounding Factor)
clear
clc

%githubpath = 'C:\Users\bonilha\Documents\GitHub\Bonilha';
githubpath = 'C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='F:\PatientData\smallSet';
cd(PatientData)

save_path='F:\PatientData\smallSet';

SmoothThres=fullfile(PatientData,'smooth');
addpath(genpath(SmoothThres));

%% Calculate Residuals

% Read excel files
ep_tle_info=readtable(fullfile(PatientData,'ep_TLE_info.xlsx'));
ep_controls_info=readtable(fullfile(PatientData,'ep_controls_info.xlsx'));
ADNI_CN_info=readtable(fullfile(PatientData,'ADNI_CN_info.csv'));
ADNI_Alz_info=readtable(fullfile(PatientData,'ADNI_Alz_info.csv'));

% look for Alz nifti files
Alzfiles={dir(fullfile(SmoothThres,'Alz\ADNI_Alz_nifti','*','*.nii')).name}';

% look for TLE nifti files
tlefiles={dir(fullfile(SmoothThres,'TLE','*','*','*.nii')).name}';

% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Controls','*','*','*GM*.nii')).name}';


disp('Loading tle control subjects and extracting 50 slices')
count1=0;
for con=1:numel(controlfiles)
    
    % Find image ID
    if isempty(regexp(controlfiles{con},'_ADNI_','match'))
        tempIN=extractBetween(controlfiles{con},'GM_','.nii');
        if any(strcmp(ep_controls_info.Participant,tempIN))
            tempage=ep_controls_info.Age(strcmp(ep_controls_info.Participant,tempIN));
            if isnan(tempage)
                disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
                continue
            elseif tempage<18
                disp(sprintf('subject %s below 18 yr old',tempIN{:}))
                continue
            end
            count1=count1+1;
        else
            disp(sprintf('Cannot find age for subject:%s',tempIN{:}))
            continue
        end
    else
        tempIN=extractBetween(controlfiles{con},'_I','.nii');
        % Find subject age
        if any(strcmp(extractAfter(ADNI_CN_info.ImageDataID,'I'),tempIN))
            tempage=ADNI_CN_info.Age(strcmp(extractAfter(ADNI_CN_info.ImageDataID,'I'),tempIN));
            if isnan(tempage)
                disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
                continue
            elseif tempage<18
                disp(sprintf('subject %s below 18 yr old',tempIN{:}))
                continue
            end
            count1=count1+1;
        else
            disp(sprintf('Cannot find age for subject:%s',tempIN{:}))
            continue
        end
    end
    
    % Load image
    temp=load_nii(controlfiles{con});
    control_img{count1,1}=temp.img(:,:,28:85);
    control_age{count1,1}=tempage;
end


tempdata=Alzfiles(strcmp(extractBetween(Alzfiles,'smooth10_','_ADNI'),'GM'));
alz_img=[];
alz_age=[];
count1=0;
disp('Loading adni alz subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image number
    tempIN=extractBetween(tempdata{con},'_I','.nii');
    
    % Find subject age
    if any(strcmp(extractAfter(ADNI_Alz_info.ImageDataID,'I'),tempIN))
        tempage=ADNI_Alz_info.Age(strcmp(extractAfter(ADNI_Alz_info.ImageDataID,'I'),tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find subject:%s',tempIN{:}))
        continue
    end
    
    % Load image
    temp=load_nii(tempdata{con});
    alz_img{count1,1}=temp.img(:,:,28:85);
    alz_age{count1,1}=tempage;
end

%%%%%%%%%%%%%% Load ep TLE %%%%%%%%%%%%%%%%%%
tempdata=tlefiles(strcmp(extractBetween(tlefiles,'smooth10_','_'),'GM'));
tle_img=[];
tle_age=[];
count1=0;
disp('Loading tle subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image ID
    tempIN=extractBetween(tempdata{con},['GM','_'],'.nii');
    
    % Find subject age
    if any(strcmp(ep_tle_info.ID,tempIN))
        tempage=ep_tle_info.Age(strcmp(ep_tle_info.ID,tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find subject:%s',tempIN{:}))
        continue
    end
    
    % Load image
    temp=load_nii(tempdata{con});
    tle_img{count1,1}=temp.img(:,:,28:85);
    tle_age{count1,1}=tempage;
end

warning('off','all')
lin_relationship = [];
template = [];
residual_imgs = cell(numel([tle_age;alz_age;control_age]),1);
tic
for vox = 1:897898
    
    intensities = [];
    age = [];
    disease = [];
    for sub = 1:numel(control_img)
        intensities = [intensities;control_img{sub}(vox)];
        age = [age;control_age{sub}];
        disease = [disease;1];
    end
    
    for sub = 1:numel(tle_img)
        intensities = [intensities;tle_img{sub}(vox)];
        age = [age;tle_age{sub}];
        disease = [disease;2];
    end
    
    for sub = 1:numel(alz_img)
        intensities = [intensities;alz_img{sub}(vox)];
        age = [age;alz_age{sub}];
        disease = [disease;3];
    end
    
    if any(intensities)
        mdl=LinearModel.fit(age,intensities);
        residuals = mdl.Residuals.('Raw');
        for s = 1:numel(residuals)
            residual_imgs{s} = [residual_imgs{s},residuals(s)];
        end
    else
        for s = 1:numel(residuals)
            residual_imgs{s} = [residual_imgs{s},0];
        end
    end
    
    if rem(vox,10000)==0
        disp([num2str(vox/897898*100),'% complete elapsed time(s)...',num2str(toc)])
    end
end

reshaped_residuals = cellfun(@(x) reshape(x,113,137,58),residual_imgs,'UniformOutput',false);
%% Setup for CNN model

net = [];
acc = [];
confmat = [];

for iter = 1:100
    
    display(['Running iteration ',num2str(iter)])
    
    % Calculate # of groups
    d_groups = unique(disease);
    
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
   