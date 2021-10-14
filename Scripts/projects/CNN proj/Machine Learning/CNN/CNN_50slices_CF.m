%% CNN MODEL (Confounding Factor)
clear
clc

githubpath='C:\Users\bonilha\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='F:\PatientData';
cd(PatientData)

save_path='F:\CNN output';

SmoothThres=fullfile(PatientData,'smooth');
addpath(genpath(SmoothThres));
cnn_output = 'F:\CNN output';

matter={'GM','WM'};

controlbrain_smooth=fullfile(githubpath,'Toolbox','imaging','standards','spm152','threshold','mod_0.2smooth10_gm_spm152.nii');
controlbrain_gm=fullfile(githubpath,'Toolbox','imaging','standards','spm152','cat12_seg','mri','mwp1spm152_GM.nii');
controlsplit=false;

% adni_control,ep_control,alz,tle
compGroup{1}={'ep_control','tle';1,2};
compGroup{2}={'ep_control','adni_control','tle','alz';1,1,2,3};
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


for comp=1:numel(compGroup)
    for m=1
    % for m=1:numel(matter)

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

        % Reshape image matrix
        adni_control_img=cat(1,adni_control_img{:});
        adni_control_img=cell2mat(adni_control_img');
        adni_control_img_reshape=reshape(adni_control_img,113,137,1,[]);


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

        % Reshape image matrix
        ep_control_img=cat(1,ep_control_img{:});
        ep_control_img=cell2mat(ep_control_img');
        ep_control_img_reshape=reshape(ep_control_img,113,137,1,[]);

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

        % Reshape image matrix
        adni_alz_img=cat(1,adni_alz_img{:});
        adni_alz_img=cell2mat(adni_alz_img');
        adni_alz_img_reshape=reshape(adni_alz_img,113,137,1,[]);

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

        % Reshape image matrix
        ep_tle_img=cat(1,ep_tle_img{:});
        ep_tle_img=cell2mat(ep_tle_img');
        ep_tle_img_reshape=reshape(ep_tle_img,113,137,1,[]);

        % Age matrix match
        adni_control_age_mat=repmat(adni_control_age',58,1);
        adni_control_age_mat=vertcat(adni_control_age_mat{:});

        ep_control_age_mat=repmat(ep_control_age',58,1);
        ep_control_age_mat=vertcat(ep_control_age_mat{:});

        adni_alz_age_mat=repmat(adni_alz_age',58,1);
        adni_alz_age_mat=vertcat(adni_alz_age_mat{:});

        ep_tle_age_mat=repmat(ep_tle_age',58,1);
        ep_tle_age_mat=vertcat(ep_tle_age_mat{:});

        %%
        % Permute
        for iter=1:100

            display(['Running iteration ',num2str(iter)])

            % Permute testing/Validation data
            % 40% for test/val
            adni_control_permtestval = randperm(size(adni_control_img_reshape,4),floor(size(adni_control_img_reshape,4)*0.40));
            ep_control_permtestval = randperm(size(ep_control_img_reshape,4),floor(size(ep_control_img_reshape,4)*0.40));
            adni_alz_permtestval = randperm(size(adni_alz_img_reshape,4),floor(size(adni_alz_img_reshape,4)*0.40));
            ep_tle_permtestval = randperm(size(ep_tle_img_reshape,4),floor(size(ep_tle_img_reshape,4)*0.40));

            % 60% of test/val for test
            adni_control_permtest = adni_control_permtestval (1:floor(0.6*numel(adni_control_permtestval)));
            adni_control_permval = adni_control_permtestval (floor(0.6*numel(adni_control_permtestval))+1:end);

            ep_control_permtest = ep_control_permtestval (1:floor(0.6*numel(ep_control_permtestval)));
            ep_control_permval = ep_control_permtestval (floor(0.6*numel(ep_control_permtestval))+1:end);

            adni_alz_permtest = adni_alz_permtestval(1:floor(0.6*numel(adni_alz_permtestval)));
            adni_alz_permval = adni_alz_permtestval(floor(0.6*numel(adni_alz_permtestval))+1:end);

            ep_tle_permtest = ep_tle_permtestval(1:floor(0.6*numel(ep_tle_permtestval)));
            ep_tle_permval = ep_tle_permtestval(floor(0.6*numel(ep_tle_permtestval))+1:end);


            % Permute training data
            adni_control_permtrain=1:size(adni_control_img_reshape,4);
            adni_control_permtrain(adni_control_permtestval)=[];

            ep_control_permtrain=1:size(ep_control_img_reshape,4);
            ep_control_permtrain(ep_control_permtestval)=[];

            adni_alz_permtrain=1:size(adni_alz_img_reshape,4);
            adni_alz_permtrain(adni_alz_permval) = [];

            ep_tle_permtrain=1:size(ep_tle_img_reshape,4);
            ep_tle_permtrain(ep_tle_permtestval) = [];


            % Select training/testing data
            adni_control_data_train= adni_control_img_reshape(:,:,:,adni_control_permtrain);
            adni_control_data_test= adni_control_img_reshape(:,:,:,adni_control_permtest);
            adni_control_data_val=adni_control_img_reshape(:,:,:,adni_control_permval);

            ep_control_data_train= ep_control_img_reshape(:,:,:,ep_control_permtrain);
            ep_control_data_test= ep_control_img_reshape(:,:,:,ep_control_permtest);
            ep_control_data_val=ep_control_img_reshape(:,:,:,ep_control_permval);

            alz_data_train= adni_alz_img_reshape(:,:,:,adni_alz_permtrain);
            alz_data_test= adni_alz_img_reshape(:,:,:,adni_alz_permtest);
            alz_data_val= adni_alz_img_reshape(:,:,:,adni_alz_permval);

            tle_data_train= ep_tle_img_reshape(:,:,:,ep_tle_permtrain);
            tle_data_test= ep_tle_img_reshape(:,:,:,ep_tle_permtest);
            tle_data_val= ep_tle_img_reshape(:,:,:,ep_tle_permval);

            % Select CF training/testing data
            ep_control_CF_train = ep_control_age_mat(ep_control_permtrain);
            adni_control_CF_train = adni_control_age_mat(adni_control_permtrain);
            tle_CF_train = ep_tle_age_mat(ep_tle_permtrain);
            alz_CF_train = adni_alz_age_mat(adni_alz_permtrain);
            
            ep_control_CF_test = ep_control_age_mat(ep_control_permtest);
            adni_control_CF_test = adni_control_age_mat(adni_control_permtest);
            tle_CF_test = ep_tle_age_mat(ep_tle_permtest);
            alz_CF_test = adni_alz_age_mat(adni_alz_permtest);
            
            ep_control_CF_val = ep_control_age_mat(ep_control_permval);
            adni_control_CF_val = adni_control_age_mat(adni_control_permval);
            tle_CF_val = ep_tle_age_mat(ep_tle_permval);
            alz_CF_val = adni_alz_age_mat(adni_alz_permval); 
            
           
            
            if controlsplit
                error('Not WORKING fix before run');
                response_train = categorical([ones(numel(adni_control_permtrain),1);ones(numel(ep_control_permtrain),1)*2;ones(numel(adni_alz_permtrain),1)*3;ones(numel(ep_tle_permtrain),1)*4]);
                response_test = categorical([ones(numel(adni_control_permtest),1);ones(numel(ep_control_permtest),1)*2;ones(numel(adni_alz_permtest),1)*3;ones(numel(ep_tle_permtest),1)*4]);
                response_val = categorical([ones(numel(adni_control_permval),1);ones(numel(ep_control_permval),1)*2;ones(numel(adni_alz_permval),1)*3;ones(numel(ep_tle_permval),1)*4]);

                quan = quantile([adni_control_age_mat;ep_control_age_mat;adni_alz_age_mat;ep_tle_age_mat],[0 0.25 0.5 0.75],'all');

                response_CF_train=[adni_control_age_mat(adni_control_permtrain);ep_control_age_mat(ep_control_permtrain);adni_alz_age_mat(adni_alz_permtrain);ep_tle_age_mat(ep_tle_permtrain)];
                response_CF_train=categorical(sum([response_CF_train>=quan(1) response_CF_train>=quan(2) response_CF_train>=quan(3) response_CF_train>=quan(4)],2));
                response_CF_test=[adni_control_age_mat(adni_control_permtest);ep_control_age_mat(ep_control_permtest);adni_alz_age_mat(adni_alz_permtest);ep_tle_age_mat(ep_tle_permtest)];
                response_CF_test=categorical(sum([response_CF_test>=quan(1) response_CF_test>=quan(2) response_CF_test>=quan(3) response_CF_test>=quan(4)],2));
                response_CF_val=[adni_control_age_mat(adni_control_permval);ep_control_age_mat(ep_control_permval);adni_alz_age_mat(adni_alz_permval);ep_tle_age_mat(ep_tle_permval)];
                response_CF_val=categorical(sum([response_CF_val>=quan(1) response_CF_val>=quan(2) response_CF_val>=quan(3) response_CF_val>=quan(4)],2));
            else
                tempComp=compGroup{comp};
                
                total_img_train = [];
                total_img_test = [];
                total_img_val = [];
                
                response_train = [];
                response_test = [];
                response_val = [];
                
                response_CF_train=[];
                response_CF_test=[];
                response_CF_val=[];
                
                compName=[];
                
                % Concatinate comparison groups
                for g=1:size(tempComp,2)
                    total_img_train = cat(4,total_img_train,eval([tempComp{1,g},'_data_train']));
                    total_img_test = cat(4,total_img_test,eval([tempComp{1,g},'_data_test']));
                    total_img_val = cat(4,total_img_val,eval([tempComp{1,g},'_data_val']));

                    response_train = [response_train;ones(size(eval([tempComp{1,g},'_data_train']),4),1)*tempComp{2,g}];
                    response_test = [response_test;ones(size(eval([tempComp{1,g},'_data_test']),4),1)*tempComp{2,g}];
                    response_val = [response_val;ones(size(eval([tempComp{1,g},'_data_val']),4),1)*tempComp{2,g}];


                    response_CF_train=[response_CF_train;eval([tempComp{1,g},'_CF_train'])];
                    response_CF_test=[response_CF_test;eval([tempComp{1,g},'_CF_test'])];
                    response_CF_val=[response_CF_val;eval([tempComp{1,g},'_CF_val'])];
                    
                    compName=[compName sprintf('%s(%d) ',tempComp{1,g},tempComp{2,g})];
                end
                
                % Calculate CF Quantiles
                quan_percent=1/size(tempComp,2);
                quan_val=[];
                for i=1:size(tempComp,2)
                    quan_val=[quan_val quan_percent*(i-1)];
                end
                quan = quantile([response_CF_train;response_CF_test;response_CF_val],quan_val,'all');

                response_CF_train_categ=[];
                response_CF_test_categ=[];
                response_CF_val_categ=[];
                for i=1:numel(quan)
                    response_CF_train_categ(:,i)=response_CF_train>=quan(i);
                    response_CF_test_categ(:,i)=response_CF_test>=quan(i);
                    response_CF_val_categ(:,i)=response_CF_val>=quan(i);
                end

                response_CF_train_categ=categorical(sum(response_CF_train_categ,2));
                response_CF_test_categ=categorical(sum(response_CF_test_categ,2));
                response_CF_val_categ=categorical(sum(response_CF_val_categ,2));
            end

            %%%%%%%%%%%% Train the network
            [net.reg{iter},acc.reg{iter},confmat.reg{iter},acc_CF.reg{iter},confmat_CF.reg{iter}]=runcnnFC(total_img_train,response_train,total_img_val,response_val,response_CF_train_categ,response_CF_val_categ,total_img_test,response_test,response_CF_test_categ);
            [net.suff{iter},acc.shuff{iter},confmat.shuff{iter},acc_CF.shuff{iter},confmat_CF.shuff{iter}]=runcnnFC(total_img_train,response_train(randperm(numel(response_train),numel(response_train))),total_img_val,response_val(randperm(numel(response_val),numel(response_val))),response_CF_train_categ(randperm(numel(response_CF_train),numel(response_CF_train))),response_CF_val_categ(randperm(numel(response_CF_val),numel(response_CF_val))),total_img_test,response_test,response_CF_test_categ);
        end
        
        
        save(fullfile(save_path,[compName,'CNN.mat']),'net','acc','confmat','acc_CF','confmat_CF','-v7.3')
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





