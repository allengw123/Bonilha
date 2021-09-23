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

SmoothThres=fullfile(PatientData,'thres');
addpath(genpath(SmoothThres));
cnn_output = 'F:\CNN output';

matter={'GM','WM'};

tle_GM=fullfile('F:\PatientData\thres\TLE\EP_RTLE_nifti\sub_6800','smooth10_GM_sub_6800.nii');
tle_WM=fullfile('F:\PatientData\thres\TLE\EP_RTLE_nifti\sub_6800','smooth10_WM_sub_6800.nii');

control_GM=fullfile('F:\PatientData\thres\Control\EP_CN_nifti\sub_7129','smooth10_GM_sub_7129.nii');
control_WM=fullfile('F:\PatientData\thres\Control\EP_CN_nifti\sub_7129','smooth10_WM_sub_7129.nii');

alz_GM=fullfile('F:\PatientData\thres\Alz\ADNI_Alz_nifti\ADNI_153_S_4172_MR_MPRAGE_br_raw_20110817113413375_135_S119196_I251187','smooth10_GM_ADNI_153_S_4172_MR_MPRAGE_br_raw_20110817113413375_135_S119196_I251187.nii');
alz_WM=fullfile('F:\PatientData\thres\Alz\ADNI_Alz_nifti\ADNI_153_S_4172_MR_MPRAGE_br_raw_20110817113413375_135_S119196_I251187','smooth10_WM_ADNI_153_S_4172_MR_MPRAGE_br_raw_20110817113413375_135_S119196_I251187.nii');
%% Setup for CNN model

% Read excel files
ep_tle_info=readtable(fullfile(PatientData,'ep_TLE_info.xlsx'));
ep_controls_info=readtable(fullfile(PatientData,'ep_controls_info.xlsx'));
ADNI_CN_info=readtable(fullfile(PatientData,'ADNI_CN_info.csv'));
ADNI_Alz_info=readtable(fullfile(PatientData,'ADNI_Alz_info.csv'));

% look for Alz nifti files
Alzfiles={dir(fullfile(SmoothThres,'Alz\ADNI_Alz_nifti','*','*.nii')).name};

% look for TLE nifti files
tlefiles={dir(fullfile(SmoothThres,'TLE','*','*','*.nii')).name};


% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Control','*','*','*.nii')).name}';
controlfiles_adni=controlfiles(contains(controlfiles,'ADNI'));
controlfiles_ep=controlfiles(~contains(controlfiles,'ADNI'));



for m=1:2
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
   
    
    % Permute
    for iter=1:100
        
        display(['Running iteration ',num2str(iter)])

        % Permute datasets
        adniControl = orgCNNinput(adni_control_img,0.6,0.25,0.15,adni_control_age);
        epControl = orgCNNinput(ep_control_img,0.6,0.25,0.15,ep_control_age);
        adniAlz = orgCNNinput(adni_alz_img,0.6,0.25,0.15,adni_alz_age);
        epTLE = orgCNNinput(ep_tle_img,0.6,0.25,0.15,ep_tle_age);
        
        % Concatinate dataset into total
        total_img_train = cat(4,adniControl.trainDataset,epControl.trainDataset,adniAlz.trainDataset,epTLE.trainDataset);
        total_img_test =  cat(4,adniControl.testDataset,epControl.testDataset,adniAlz.testDataset,epTLE.testDataset);
        total_img_val = cat(4,adniControl.valDataset,epControl.valDataset,adniAlz.valDataset,epTLE.valDataset);
        
        % Obtain response labels
        response_train = categorical([ones(size(adniControl.trainDataset,4),1);ones(size(epControl.trainDataset,4),1);ones(size(adniAlz.trainDataset,4),1)*2;ones(size(epTLE.trainDataset,4),1)*3]);
        response_test =  categorical([ones(size(adniControl.testDataset,4),1);ones(size(epControl.testDataset,4),1);ones(size(adniAlz.testDataset,4),1)*2;ones(size(epTLE.testDataset,4),1)*3]);
        response_val =  categorical([ones(size(adniControl.valDataset,4),1);ones(size(epControl.valDataset,4),1);ones(size(adniAlz.valDataset,4),1)*2;ones(size(epTLE.valDataset,4),1)*3]);
        
        % Obtain age labels
        quan = floor(quantile(cell2mat([adni_control_age;ep_control_age;adni_alz_age;ep_tle_age]),[0 0.33 0.66],'all'));
        
        response_CF_train=[adniControl.trainAge;epControl.trainAge;adniAlz.trainAge;epTLE.trainAge];
        response_CF_train=categorical(sum([response_CF_train>=quan(1) response_CF_train>=quan(2) response_CF_train>=quan(3)],2));
        
        response_CF_test=[adniControl.testAge;epControl.testAge;adniAlz.testAge;epTLE.testAge];
        response_CF_test=categorical(sum([response_CF_test>=quan(1) response_CF_test>=quan(2) response_CF_test>=quan(3)],2));

        response_CF_val=[adniControl.valAge;epControl.valAge;adniAlz.valAge;epTLE.valAge];
        response_CF_val=categorical(sum([response_CF_val>=quan(1) response_CF_val>=quan(2) response_CF_val>=quan(3)],2));

        
        %%%%%%%%%%%% Train the network
        [net.reg{iter},acc.reg{iter},confmat.reg{iter},acc_CF.reg{iter},confmat_CF.reg{iter}]=runcnnFC(total_img_train,response_train,total_img_val,response_val,response_CF_train,response_CF_val,total_img_test,response_test,response_CF_test);
        [net.suff{iter},acc.shuff{iter},confmat.shuff{iter},acc_CF.shuff{iter},confmat_CF.shuff{iter}]=runcnnFC(total_img_train,response_train(randperm(numel(response_train),numel(response_train))),total_img_val,response_val(randperm(numel(response_val),numel(response_val))),response_CF_train(randperm(numel(response_CF_train),numel(response_CF_train))),response_CF_val(randperm(numel(response_CF_val),numel(response_CF_val))),total_img_test,response_test,response_CF_test);
    end
    
    save(fullfile(save_path,['CNN_',matter{m},'.mat']),'net','acc','confmat','acc_CF','confmat_CF','-v7.3')
end
%% Analyze network


analyzeNetwork(net.reg{1})

for m=1:numel(matter)
    
    % Load network
    load(fullfile(cnn_output,['CNN_',matter{m}]))
    
    %%%% Historgram of accuracy
    figure('WindowState','maximized');
    figtitle=['CNN - middle 50 percent slices - Axial',' ',matter{m}];
    sgtitle(figtitle)

    subplot(2,4,1)
    histogram(cell2mat(acc.reg),'BinWidth',0.05);
    xlim([0 1.2])
    xlabel('Accuracy')
    ylabel('# of models')
    title('CNN Reg Label')

    subplot(2,4,5)
    hold on
    histogram(cellfun(@(x) x(1,1)/sum(x(1,:),'all'),confmat.reg),'BinWidth',0.05);
    histogram(cellfun(@(x) x(2,2)/sum(x(2,:),'all'),confmat.reg),'BinWidth',0.05);
    histogram(cellfun(@(x) x(3,3)/sum(x(3,:),'all'),confmat.reg),'BinWidth',0.05);
    legend('control','alz','tle')
    xlim([0 1.2])
    xlabel('Accuracy')
    ylabel('# of models')

    subplot(2,4,2)
    histogram(cell2mat(acc.shuff),'BinWidth',0.05);
    xlim([0 1.2])
    xlabel('Accuracy')
    ylabel('# of models')
    title('CNN Shuffle Label')

    subplot(2,4,6)
    hold on
    histogram(cellfun(@(x) x(1,1)/sum(x(1,:),'all'),confmat.shuff),'BinWidth',0.05);
    histogram(cellfun(@(x) x(2,2)/sum(x(2,:),'all'),confmat.shuff),'BinWidth',0.05);
    histogram(cellfun(@(x) x(3,3)/sum(x(3,:),'all'),confmat.shuff),'BinWidth',0.05);
    legend('control','alz','tle')
    xlabel('Accuracy')
    ylabel('# of models')
    xlim([0 1.2])


    subplot(2,4,3)
    for i=1:numel(acc_CF.reg)
        [CF_max(i),CF_idx(i)]=max(acc_CF.reg{i});
    end
    histogram(CF_max,'BinWidth',0.05);
    xlim([0 1.2])
    xlabel('Accuracy')
    ylabel('# of models')
    title('CF Reg Label')

    subplot(2,4,7)
    for i=1:numel(confmat_CF.reg)
       temp=cellfun(@(y) y(1,1)/sum(y(1,:),'all'),confmat_CF.reg{i});
       control_temp(i)=temp(CF_idx(i));
       
       temp=cellfun(@(y) y(2,2)/sum(y(2,:),'all'),confmat_CF.reg{i});
       alz_temp(i)=temp(CF_idx(i));
       
       temp=cellfun(@(y) y(3,3)/sum(y(3,:),'all'),confmat_CF.reg{i});
       tle_temp(i)=temp(CF_idx(i));
    end
    hold on
    histogram(control_temp,'BinWidth',0.05);
    histogram(alz_temp,'BinWidth',0.05);
    histogram(tle_temp,'BinWidth',0.05);
    legend('control','alz','tle')
    xlabel('Accuracy')
    ylabel('# of models')
    xlim([0 1.2])

    subplot(2,4,4)
    histogram(cellfun(@(x) mean(x),acc_CF.shuff),'BinWidth',0.05);
    xlim([0 1.2])
    xlabel('Accuracy')
    ylabel('# of models')
    title('CF Shuffle Label')

    subplot(2,4,8)
    hold on
    histogram(cellfun(@(x) mean(cellfun(@(y) y(1,1)/sum(y(1,:),'all'),x),'all'),confmat_CF.shuff),'BinWidth',0.05);
    histogram(cellfun(@(x) mean(cellfun(@(y) y(2,2)/sum(y(2,:),'all'),x),'all'),confmat_CF.shuff),'BinWidth',0.05);
    histogram(cellfun(@(x) mean(cellfun(@(y) y(3,3)/sum(y(3,:),'all'),x),'all'),confmat_CF.shuff),'BinWidth',0.05);
    legend('control','alz','tle')
    xlabel('Accuracy')
    ylabel('# of models')
    xlim([0 1.2])

    saveas(gcf,fullfile(save_path,figtitle));
    close all
    clc
    
    
    %%%%% Feature visualization
    % Load template imgs
    alzimg=load_nii(eval(['alz_',matter{m}]));
    controlimg=load_nii(eval(['control_',matter{m}]));
    tleimg=load_nii(eval(['tle_',matter{m}]));
    
    % Define image size
    imgSize = size(controlimg.img);
    imgSize = imgSize(1:2);


    layers_num=[1 4 8 12];
    layers_label={'Input','ReLu 1','ReLu 2','ReLu 3'};
    groups={'Control','TLE','Alz'};

    actimages={controlimg,tleimg,alzimg};

    % Video
    vw=VideoWriter(fullfile(save_path,['CNN activation_',matter{m}]),'MPEG-4');
    vw.FrameRate=1;
    open(vw)
    fh=figure('WindowState','maximized');
    for s=28:85
        sgtitle(sprintf('Slice #%u',s))
        for l=1:numel(layers_num)
            for g=1:numel(groups)
                act_cum_mean=[];
                act_cum_std=[];

                for n=1:numel(net.reg)
                    tempimg=actimages{g}.img;
                    tempnet=net.reg{n};  
                    if acc.reg{n}<0.70
                        continue
                    end
                    act = activations(tempnet,tempimg(:,:,s),layers_num(l));

                    act_cum_mean(:,:,n)=mean(act,3);
                    act_cum_std(:,:,n)=std(act,[],3);
                end

                subplot(numel(layers_num),numel(groups)*2,g+(l-1)*numel(groups)*2)
                imagesc(mean(act_cum_mean,3))
                if l==1
                    title(sprintf('Mean activation - %s',groups{g}));
                end
                ylabel(layers_label{l})
                colormap jet

                subplot(numel(layers_num),numel(groups)*2,g+3+(l-1)*numel(groups)*2)
                imagesc(std(act_cum_std,[],3))
                if l==1
                    title(sprintf('STD activation - %s',groups{g}));
                end
                ylabel(layers_label{l})
                colormap jet
            end
        end
        drawnow
        writeVideo(vw,getframe(fh));
    end
    close(vw)
    
    disease={'control','alz','tle'};
    for a=1:numel(actimages)
        figure;
        for slice=28:5:85
            tempimg=actimages{a}.img;
            for n=1:numel(net.reg)
                tempnet=net.reg{n};  
                if acc.reg{n}<0.70
                    continue
                end
                act(:,:,n) = mean(activations(tempnet,tempimg(:,:,slice),12),3);
            end
            act_cum_mean=mean(act,3);
            nexttile
            imagesc(act_cum_mean)
            title(['Slice #',num2str(slice)])
        end
        sgtitle(disease{a})
   end

end
                    


%     if s==1
%         
%         con_h=nexttile;
%         imshow(controlimg.img(:,:,s),'InitialMagnification','fit','Parent',con_h)
%         title(con_h,'Original image')
%         for a=1:numel(act)
%             img = imtile(mat2gray(act{a}),'GridSize',[6 6]);
%             h(a)=nexttile;
%             imshow(img,'InitialMagnification','fit','Parent',h(a));
%             title(h(a),['Net ',num2str(a),' - Accuracy ',num2str(accuracy_val(a))])
%         end
%     else
%         imshow(controlimg.img(:,:,s),'InitialMagnification','fit','Parent',con_h)
%         title(con_h,'Original image')
%         for a=1:numel(act)
%             img = imtile(mat2gray(act{a}),'GridSize',[6 6]);
%             imshow(img,'InitialMagnification','fit','Parent',h(a));
%             title(h(a),['Net ',num2str(a),' - Accuracy ',num2str(accuracy_val(a))])
%         end
%     end
% end


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
        'ValidationData',{valData,valResponse}, ...
        'Verbose',false, ... %Indicator to display training progress information in the command window
        'Plots','none',...
        'Shuffle','every-epoch', ...
        'ExecutionEnvironment','multi-gpu');

    optionsCF= trainingOptions('sgdm', ...  %stochastic gradient descent with momentum(SGDM) optimizer
        'InitialLearnRate',0.01, ...
        'MaxEpochs',30, ...  % Default is 30
        'ValidationData',{valData,CFvalResponse}, ...
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
    net=trainNetwork(trainData,trainResponse,layers,options);

    % Test on regular response
    YPred_test = classify(net,testDat);
    YTest = testRes;
    acc = sum(YPred_test == YTest)/numel(YTest);
    con = confusionmat(YTest,YPred_test);
    % figure;confusionchart(con,{'Adni_control','Tle_control','Alz','TLE'})
    % title(num2str(acc))


    % Test on CF response
    % figure
    groups=unique(CFtestRes);
    per=double(perms(groups));
    for p=1:size(per,1)

        YPred_test = classify(net,testDat);

        % Change labels based permutations
        YTest = double(CFtestRes);
        YTest(double(CFtestRes)==1)=per(p,1);
        YTest(double(CFtestRes)==2)=per(p,2);
        YTest(double(CFtestRes)==3)=per(p,3);

        try
        YTest(double(CFtestRes)==4)=per(p,4);
        catch
        end

        YTest = categorical(YTest);


        acc_CF(p) = sum(YPred_test == YTest)/numel(YTest);
        con_CF{p}=confusionmat(YTest,YPred_test);
    %     nexttile
    %     confusionchart(con_CF,{'Adni_control','Tle_control','Alz','TLE'})
    %     title(num2str(acc_CF(p)))
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


