%% CNN MODEL (Confounding Factor)
clear
clc

githubpath='C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='F:\PatientData';
cd(PatientData)

save_path='F:\CNN output';

SmoothThres=fullfile(PatientData,'smooth');
addpath(genpath(SmoothThres));

matter={'GM','WM'};
previous=[];
controlsplit=false;
cont=false;

% adni_control,ep_control,alz,tle
compGroup{1}={'ep_control','tle';1,2};
compGroup{2}={'ep_control','adni_control','tle','alz';1,1,2,3};

% Setup local cluster
myCluster = parcluster('local');
delete(myCluster.Jobs);
%% Setup for CNN model


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
controlfiles={dir(fullfile(SmoothThres,'Controls','*','*','*.nii')).name}';
controlfiles_adni=controlfiles(contains(controlfiles,'ADNI'));
controlfiles_ep=controlfiles(~contains(controlfiles,'ADNI'));


for m=1:numel(matter)
    disp(['Running ',matter{m}])
    
    %%%%%%%%%%%%%% Load adni control %%%%%%%%%%%%%%%%%%
    tempdata=controlfiles_adni(strcmp(extractBetween(controlfiles_adni,'smooth10_','_ADNI'),matter{m}));
    adni_control_img=[];
    adni_control_age=[];
    count1=0;
    disp('Loading adni control subjects and extracting 50 slices')
    for con=1:numel(tempdata)
        
        % Find image ID
        tempIN=extractBetween(tempdata{con},'_I','.nii');
        
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
        
        % Load Image
        temp=load_nii(tempdata{con});
        count2=1;
        for i=28:85
            temp_img{count2,1}=temp.img(:,:,i);
            count2=count2+1;
        end
        adni_control_img{count1,1}=temp_img;
        adni_control_age{count1,1}=tempage;
    end
    
    %%%%%%%%%%%%%% Load ep control %%%%%%%%%%%%%%%%%%
    tempdata=controlfiles_ep(strcmp(extractBetween(controlfiles_ep,'smooth10_','_'),matter{m}));
    ep_control_img=[];
    ep_control_age=[];
    count1=0;
    disp('Loading tle control subjects and extracting 50 slices')
    for con=1:numel(tempdata)
        
        % Find image ID
        tempIN=extractBetween(tempdata{con},[matter{m},'_'],'.nii');
        
        % Find subject age
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
        
        % Load image
        temp=load_nii(tempdata{con});
        count2=1;
        for i=28:85
            temp_img{count2,1}=temp.img(:,:,i);
            count2=count2+1;
        end
        ep_control_img{count1,1}=temp_img;
        ep_control_age{count1,1}=tempage;
    end
    
    %%%%%%%%%%%%%% Load adni Alz %%%%%%%%%%%%%%%%%%
    tempdata=Alzfiles(strcmp(extractBetween(Alzfiles,'smooth10_','_ADNI'),matter{m}));
    adni_alz_img=[];
    adni_alz_age=[];
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
        count2=1;
        for i=28:85
            temp_img{count2,1}=temp.img(:,:,i);
            count2=count2+1;
        end
        adni_alz_img{count1,1}=temp_img;
        adni_alz_age{count1,1}=tempage;
    end
    
    %%%%%%%%%%%%%% Load ep TLE %%%%%%%%%%%%%%%%%%
    tempdata=tlefiles(strcmp(extractBetween(tlefiles,'smooth10_','_'),matter{m}));
    ep_tle_img=[];
    ep_tle_age=[];
    count1=0;
    disp('Loading tle subjects and extracting 50 slices')
    for con=1:numel(tempdata)
        
        % Find image ID
        tempIN=extractBetween(tempdata{con},[matter{m},'_'],'.nii');
        
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
        
        count2=1;
        for i=28:85
            temp_img{count2,1}=temp.img(:,:,i);
            count2=count2+1;
        end
        ep_tle_img{count1,1}=temp_img;
        ep_tle_age{count1,1}=tempage;
    end
    %%
    for comp=1:numel(compGroup)
        while cont==false
            
            %%%%% Permute
            for iter=1:100
                while cont==false
                    
                    display(['Running iteration ',num2str(iter)])
                    trainPercent,testPercent,valPercent,inputage)
                    % Permute datasets
                    adni_control = orgCNNinput(adni_control_img,0.6,0.25,0.15,adni_control_age);
                    ep_control = orgCNNinput(ep_control_img,0.6,0.25,0.15,ep_control_age);
                    alz = orgCNNinput(adni_alz_img,0.6,0.25,0.15,adni_alz_age);
                    tle = orgCNNinput(ep_tle_img,0.6,0.25,0.15,ep_tle_age);
                    
                    
                    if controlsplit
                        error('Not WORKING fix before run');
                        response_train = categorical([ones(numel(adni_control_permtrain),1);ones(numel(ep_control_permtrain),1)*2;ones(numel(adni_alz_permtrain),1)*3;ones(numel(ep_tle_permtrain),1)*4]);
                        response_test = categorical([ones(numel(adni_control_permtest),1);ones(numel(ep_control_permtest),1)*2;ones(numel(adni_alz_permtest),1)*3;ones(numel(ep_tle_permtest),1)*4]);
                        response_val = categorical([ones(numel(adni_control_permval),1);ones(numel(ep_control_permval),1)*2;ones(numel(adni_alz_permval),1)*3;ones(numel(ep_tle_permval),1)*4]);
                        
                        quan = quantile([adni_control_age_mat;ep_control_age_mat;adni_alz_age_mat;ep_tle_age_mat],[0 0.25 0.5 0.75],'all');
                        
                        response.trainAge=[adni_control_age_mat(adni_control_permtrain);ep_control_age_mat(ep_control_permtrain);adni_alz_age_mat(adni_alz_permtrain);ep_tle_age_mat(ep_tle_permtrain)];
                        response.trainAge=categorical(sum([response.trainAge>=quan(1) response.trainAge>=quan(2) response.trainAge>=quan(3) response.trainAge>=quan(4)],2));
                        response.testAge=[adni_control_age_mat(adni_control_permtest);ep_control_age_mat(ep_control_permtest);adni_alz_age_mat(adni_alz_permtest);ep_tle_age_mat(ep_tle_permtest)];
                        response.testAge=categorical(sum([response.testAge>=quan(1) response.testAge>=quan(2) response.testAge>=quan(3) response.testAge>=quan(4)],2));
                        response.valAge=[adni_control_age_mat(adni_control_permval);ep_control_age_mat(ep_control_permval);adni_alz_age_mat(adni_alz_permval);ep_tle_age_mat(ep_tle_permval)];
                        response.valAge=categorical(sum([response.valAge>=quan(1) response.valAge>=quan(2) response.valAge>=quan(3) response.valAge>=quan(4)],2));
                    else
                        tempComp=compGroup{comp};
                        
                        total_img_train = [];
                        total_img_test = [];
                        total_img_val = [];
                        
                        response_train = [];
                        response_test = [];
                        response_val = [];
                        
                        response.trainAge=[];
                        response.testAge=[];
                        response.valAge=[];
                        
                        compName=[];
                        
                        % Concatinate comparison groups
                        for g=1:size(tempComp,2)
                            total_img_train = cat(4,total_img_train,eval([tempComp{1,g},'.trainDataset']));
                            total_img_test = cat(4,total_img_test,eval([tempComp{1,g},'.testDataset']));
                            total_img_val = cat(4,total_img_val,eval([tempComp{1,g},'.valDataset']));
                            
                            response_train = [response_train;ones(size(eval([tempComp{1,g},'.trainDataset']),4),1)*tempComp{2,g}];
                            response_test = [response_test;ones(size(eval([tempComp{1,g},'.testDataset']),4),1)*tempComp{2,g}];
                            response_val = [response_val;ones(size(eval([tempComp{1,g},'.valDataset']),4),1)*tempComp{2,g}];
                            
                            
                            response.trainAge=[response.trainAge;eval([tempComp{1,g},'.trainAge'])];
                            response.testAge=[response.testAge;eval([tempComp{1,g},'.testAge'])];
                            response.valAge=[response.valAge;eval([tempComp{1,g},'.valAge'])];
                            
                            compName=[compName sprintf('%s(%d) ',tempComp{1,g},tempComp{2,g})];
                        end
                        
                        if exist(fullfile(save_path,[compName,'-',matter{m},'-CNN.mat']),'file')~=0
                            previous=vertcat(previous,fullfile(save_path,[compName,'-',matter{m},'-CNN.mat']))
                            cont=true;
                            continue
                        end
                        
                        % Calculate CF Quantiles
                        quan_percent=1/size(tempComp,2);
                        quan_val=[];
                        for i=1:size(tempComp,2)
                            quan_val=[quan_val quan_percent*(i-1)];
                        end
                        quan = quantile([response.trainAge;response.testAge;response.valAge],quan_val,'all');
                        
                        response.trainAge_categ=[];
                        response.testAge_categ=[];
                        response.valAge_categ=[];
                        for i=1:numel(quan)
                            response.trainAge_categ(:,i)=response.trainAge>=quan(i);
                            response.testAge_categ(:,i)=response.testAge>=quan(i);
                            response.valAge_categ(:,i)=response.valAge>=quan(i);
                        end
                        
                        response.trainAge_categ=categorical(sum(response.trainAge_categ,2));
                        response.testAge_categ=categorical(sum(response.testAge_categ,2));
                        response.valAge_categ=categorical(sum(response.valAge_categ,2));
                    end
                    
                    %%%%%%%%%%%% Train the network
                    [net.reg{iter},acc.reg{iter},confmat.reg{iter},acc_CF.reg{iter},confmat_CF.reg{iter}]=runcnnFC(total_img_train,response_train,total_img_val,response_val,response.trainAge_categ,response.valAge_categ,total_img_test,response_test,response.testAge_categ);
                    [net.suff{iter},acc.shuff{iter},confmat.shuff{iter},acc_CF.shuff{iter},confmat_CF.shuff{iter}]=runcnnFC(total_img_train,response_train(randperm(numel(response_train),numel(response_train))),total_img_val,response_val(randperm(numel(response_val),numel(response_val))),response.trainAge_categ(randperm(numel(response.trainAge),numel(response.trainAge))),response.valAge_categ(randperm(numel(response.valAge),numel(response.valAge))),total_img_test,response_test,response.testAge_categ);
                    
                    
                    
                    save(fullfile(save_path,[compName,'-',matter{m},'-CNN.mat']),'net','acc','confmat','acc_CF','confmat_CF','-v7.3')
                end
            end
        end
        cont=false;
    end
end
%% Funtions

function [net,acc,con,acc_CF,con_CF]=runcnnFC(trainData,trainResponse,valData,valResponse,CFtrainResponse,CFvalResponse,testDat,testRes,CFtestRes)


%% Parameters for the network

% Layers
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
    
    fullyConnectedLayer(numel(unique(trainResponse)))
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...  %stochastic gradient descent with momentum(SGDM) optimizer
    'InitialLearnRate',0.01, ...
    'MaxEpochs',30, ...  % Default is 30
    'ValidationData',{valData,categorical(valResponse)}, ...
    'Verbose',false, ... %Indicator to display training progress information in the command window
    'Plots','none',...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment','multi-gpu');

optionsCF= trainingOptions('sgdm', ...  %stochastic gradient descent with momentum(SGDM) optimizer
    'InitialLearnRate',0.01, ...
    'MaxEpochs',30, ...  % Default is 30
    'ValidationData',{valData,categorical(CFvalResponse)}, ...
    'Verbose',false, ... %Indicator to display training progress information in the command window
    'Plots','none',...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment','multi-gpu');

% options_single = trainingOptions('sgdm', ...  %stochastic gradient descent with momentum(SGDM) optimizer
%     'InitialLearnRate',0.01, ...
%     'MaxEpochs',1, ...  % Default is 30
%     'ValidationData',{valData,valResponse}, ...
%     'Verbose',false, ... %Indicator to display training progress information in the command window
%     'Plots','none',...
%     'Shuffle','every-epoch', ...
%     'ExecutionEnvironment','multi-gpu');
%
% optionsCF_sing = trainingOptions('sgdm', ...  %stochastic gradient descent with momentum(SGDM) optimizer
%     'InitialLearnRate',0.01, ...
%     'MaxEpochs',1, ...  % Default is 30
%     'ValidationData',{valData,CFvalResponse}, ...
%     'Verbose',false, ... %Indicator to display training progress information in the command window
%     'Plots','none',...
%     'Shuffle','every-epoch', ...
%     'ExecutionEnvironment','multi-gpu');

%%%%%%% % Parse the CF data %%%%%%

% figure
% spot=[1:2:30 2:2:30];
% ax=[];
% for i=1:30
%     if i==1
%         net=trainNetwork(trainData,trainResponse,layers,options_single);
%         subplot(15,2,spot(i))
%         tempax=imagesc(net.Layers(13).Weights);
%         tempax.Parent.XTick=[];
%         YPred_test = classify(net,testDat);
%         YTest = testRes;
%         accuracy_test = sum(YPred_test == YTest)/numel(YTest);
%         title(sprintf('NN(%f)',accuracy_test),'Rotation',0);
%         ax=[ax tempax];
%         yticks([1 2 3 4]);
%     elseif i>1 && i<=15
%         net=trainNetwork(trainData,trainResponse,net.Layers,options_single);
%         tempax=subplot(15,2,spot(i))
%         imagesc(net.Layers(13).Weights);
%         tempax.Parent.XTick=[];
%         YPred_test = classify(net,testDat);
%         YTest = testRes;
%         accuracy_test = sum(YPred_test == YTest)/numel(YTest);
%         title(sprintf('NN(%f)',accuracy_test));
%         ax=[ax tempax];
%     else
%         net=trainNetwork(trainData,CFtrainResponse,net.Layers,optionsCF_sing);
%         tempax=subplot(15,2,spot(i))
%         imagesc(net.Layers(13).Weights);
%         tempax.Parent.XTick=[];
%         YPred_test = classify(net,testDat);
%         YTest = CFtestRes;
%         accuracy_test = sum(YPred_test == YTest)/numel(YTest);
%         title(sprintf('CF(%f)',accuracy_test));
%         ax=[ax tempax];
%     end
% end
% linkaxes(ax)
% cb=cbar;


%% Train networks

%%%%%%% Train on regular data %%%%%
net=trainNetwork(trainData,categorical(trainResponse),layers,options);

% Test on regular response
YPred_test = classify(net,testDat);
YTest = categorical(testRes);
acc = sum(YPred_test == YTest)/numel(YTest);
[con.C, con.order]= confusionmat(YTest,YPred_test);


% Test on CF response
groups=unique(CFtestRes);
per=double(perms(groups));
for p=1:size(per,1)
    
    YPred_test = classify(net,testDat);
    
    % Change labels based permutations
    YTest = double(CFtestRes);
    YTest(double(CFtestRes)==1)=per(p,1);
    YTest(double(CFtestRes)==2)=per(p,2);
    
    try
        YTest(double(CFtestRes)==3)=per(p,3);
    catch
    end
    
    try
        YTest(double(CFtestRes)==4)=per(p,4);
    catch
    end
    
    YTest = categorical(YTest);
    
    
    acc_CF(p) = sum(YPred_test == YTest)/numel(YTest);
    [con_CF.C{p},con_CF.order{p}]=confusionmat(YTest,YPred_test);
    con_CF.perm{p}=[1 per(p,1);2 per(p,2)];
    try
        con_CF.perm{p}=[con_CF.perm{p};3 per(p,3)];
    catch
    end
end
end

function output=orgCNNinput(input,trainPercent,testPercent,valPercent,inputage)

% Seperate test permutations
testidx=randperm(numel(input),floor(testPercent*numel(input)));
testDataset=input(testidx);

testAge=inputage(testidx);

% Seperate Train/Val permutations
trainvalDataset=input;
trainvalDataset(testidx)=[];

trainvalAge=inputage;
trainvalAge(testidx)=[];

% Seperate Val permutations
validx=randperm(numel(trainvalDataset),floor((valPercent/trainPercent)*numel(trainvalDataset)));
valDataset=trainvalDataset(validx);

valAge=trainvalAge(validx);

% Seperate Test permutations
trainDataset=trainvalDataset;
trainDataset(validx)=[];

trainAge=trainvalAge;
trainAge(validx)=[];

% Reshape images
output.testDataset=reshape(cell2mat(cat(1,testDataset{:})'),113,137,1,[]);
output.trainDataset=reshape(cell2mat(cat(1,trainDataset{:})'),113,137,1,[]);
output.valDataset=reshape(cell2mat(cat(1,valDataset{:})'),113,137,1,[]);

% Reshape age
output.testAge=repmat(testAge',58,1);
output.testAge=vertcat(output.testAge{:});

output.valAge=repmat(valAge',58,1);
output.valAge=vertcat(output.valAge{:});

output.trainAge=repmat(trainAge',58,1);
output.trainAge=vertcat(output.trainAge{:});
end





