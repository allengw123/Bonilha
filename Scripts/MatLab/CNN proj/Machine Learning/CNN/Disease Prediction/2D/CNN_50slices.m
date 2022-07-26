%% CNN MODEL

githubpath='C:\Users\bonilha\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
PatientData='F:\PatientData\smooth_thr02';
addpath(genpath(PatientData));
cnn_output = 'F:\CNN output';

matter={'GM','WM'};


%% Setup for CNN model

% look for Alz nifti files
Alzfiles={dir(fullfile('F:\PatientData\smooth_thr02\Alz\ADNI_Alz_nifti','*','*.nii')).name};

% look for TLE nifti files
tlefiles={dir(fullfile('F:\PatientData\smooth_thr02\TLE','*','*','*.nii')).name};

% look for control nifti files
controlfiles={dir(fullfile('F:\PatientData\smooth_thr02\Control','*','*','*.nii')).name}';
controlfiles_adni=controlfiles(contains(controlfiles,'ADNI'));
controlfile_tle=controlfiles(~contains(controlfiles,'ADNI'));

for m=1:numel(matter)

    disp(['Running ',matter{m}])

    % Load adni control imgs
    disp('Loading adni control subjects and extracting 50 slices')
    tempdata=controlfiles_adni(strcmp(extractBetween(controlfiles_adni,'smooth','02_'),matter{m}));
    for con=1:numel(tempdata)
        temp=load_nii(tempdata{con});
        count=1;
        for i=28:85
            temp_img{count,1}=temp.img(:,:,i);
            count=count+1;
        end
        adni_control_img{con,1}=temp_img;
    end
    adni_control_img=cat(1,adni_control_img{:});
    adni_control_img=cell2mat(adni_control_img');
    adni_control_img_reshape=reshape(adni_control_img,113,137,1,[]);
    
    % Load tle control imgs
    disp('Loading tle control subjects and extracting 50 slices')
    tempdata=controlfile_tle(strcmp(extractBetween(controlfile_tle,'smooth','02_'),matter{m}));
    for con=1:numel(tempdata)
        temp=load_nii(tempdata{con});
        count=1;
        for i=28:85
            temp_img{count,1}=temp.img(:,:,i);
            count=count+1;
        end
        tle_control_img{con,1}=temp_img;
    end
    tle_control_img=cat(1,tle_control_img{:});
    tle_control_img=cell2mat(tle_control_img');
    tle_control_img_reshape=reshape(tle_control_img,113,137,1,[]);
    
    % Load Alz imgs
    disp('Loading Alz subjects and extracting 50 slices')
    tempdata=Alzfiles(strcmp(extractBetween(Alzfiles,'smooth','02_'),matter{m}));
    for Alz=1:numel(tempdata)
        temp=load_nii(tempdata{Alz});
        count=1;
        for i=28:85
            temp_img{count,1}=temp.img(:,:,i);
            count=count+1;
        end
        Alz_img{Alz,1}=temp_img;
    end
    Alz_img=cat(1,Alz_img{:});
    Alz_img=cell2mat(Alz_img');
    Alz_img_reshape=reshape(Alz_img,113,137,1,[]);

    % Load tle imgs
    disp('Loading TLE subjects and extracting 50 slices')
    tempdata=tlefiles(strcmp(extractBetween(tlefiles,'smooth','02_'),matter{m}));
    for tle=1:numel(tempdata)
        temp=load_nii(tempdata{tle});
        count=1;
        for i=28:85
            temp_img{count,1}=temp.img(:,:,i);
            count=count+1;
        end
        tle_img{tle,1}=temp_img;
    end
    tle_img=cat(1,tle_img{:});
    tle_img=cell2mat(tle_img');
    tle_img_reshape=reshape(tle_img,113,137,1,[]);


    % Permute
    for iter=1:5
        display(['Running iteration ',num2str(iter)])

        % Permute testing/Validation data
        % 40% for test/val
        adni_control_permtestval = randperm(size(adni_control_img_reshape,4),floor(size(adni_control_img_reshape,4)*0.40));
        tle_control_permtestval = randperm(size(tle_control_img_reshape,4),floor(size(tle_control_img_reshape,4)*0.40));
        Alz_permtestval = randperm(size(Alz_img_reshape,4),floor(size(Alz_img_reshape,4)*0.40));
        tle_permtestval = randperm(size(tle_img_reshape,4),floor(size(tle_img_reshape,4)*0.40));

        % 60% of test/val for test
        adni_control_permtest = adni_control_permtestval (1:floor(0.6*numel(adni_control_permtestval)));
        adni_control_permval = adni_control_permtestval (floor(0.6*numel(adni_control_permtestval))+1:end);

        tle_control_permtest = tle_control_permtestval (1:floor(0.6*numel(tle_control_permtestval)));
        tle_control_permval = tle_control_permtestval (floor(0.6*numel(tle_control_permtestval))+1:end);
        
        Alz_permtest = Alz_permtestval(1:floor(0.6*numel(Alz_permtestval)));
        Alz_permval = Alz_permtestval(floor(0.6*numel(Alz_permtestval))+1:end);

        tle_permtest = tle_permtestval(1:floor(0.6*numel(tle_permtestval)));
        tle_permval = tle_permtestval(floor(0.6*numel(tle_permtestval))+1:end);



        % Permute training data
        adnicontrol_permtrain=1:size(adni_control_img_reshape,4);
        adnicontrol_permtrain(adni_control_permtestval)=[];
        
        tlecontrol_permtrain=1:size(tle_control_img_reshape,4);
        tlecontrol_permtrain(tle_control_permtestval)=[];

        Alz_permtrain=1:size(Alz_img_reshape,4);
        Alz_permtrain(Alz_permtestval) = [];

        tle_permtrain=1:size(tle_img_reshape,4);
        tle_permtrain(tle_permtestval) = [];


        % Select training/testing data
        adni_control_data_train= adni_control_img_reshape(:,:,:,adnicontrol_permtrain);
        adni_control_data_test= adni_control_img_reshape(:,:,:,adni_control_permtest);
        adni_control_data_val=adni_control_img_reshape(:,:,:,adni_control_permval);

        tle_control_data_train= tle_control_img_reshape(:,:,:,tlecontrol_permtrain);
        tle_control_data_test= tle_control_img_reshape(:,:,:,tle_control_permtest);
        tle_control_data_val=tle_control_img_reshape(:,:,:,tle_control_permval);
        
        Alz_data_train= Alz_img_reshape(:,:,:,Alz_permtrain);
        Alz_data_test= Alz_img_reshape(:,:,:,Alz_permtest);
        Alz_data_val= Alz_img_reshape(:,:,:,Alz_permval);

        tle_data_train= tle_img_reshape(:,:,:,tle_permtrain);
        tle_data_test= tle_img_reshape(:,:,:,tle_permtest);
        tle_data_val= tle_img_reshape(:,:,:,tle_permval);

        % Concatinate 
        total_img_train = cat(4,adni_control_data_train,tle_control_data_train,Alz_data_train,tle_data_train);
        total_img_test = cat(4,adni_control_data_test,tle_control_data_test,Alz_data_test,tle_data_test);
        total_img_val = cat(4,adni_control_data_val,tle_control_data_val,Alz_data_val,tle_data_val);

        response_train = categorical([ones(numel(adnicontrol_permtrain),1);ones(numel(tlecontrol_permtrain),1)*2;ones(numel(Alz_permtrain),1)*3;ones(numel(tle_permtrain),1)*4]);
        response_test = categorical([ones(numel(adni_control_permtest),1);ones(numel(tle_control_permtest),1)*2;ones(numel(Alz_permtest),1)*3;ones(numel(tle_permtest),1)*4]);
        response_val = categorical([ones(numel(adni_control_permval),1);ones(numel(tle_control_permval),1)*2;ones(numel(Alz_permval),1)*3;ones(numel(tle_permval),1)*4]);



        % Parameters for the network
        imageSize = [113 137 1];

        layers = [
            imageInputLayer(imageSize)

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

            fullyConnectedLayer(4)
            softmaxLayer
            classificationLayer];

        options = trainingOptions('sgdm', ...  %stochastic gradient descent with momentum(SGDM) optimizer
            'InitialLearnRate',0.01, ...
            'MaxEpochs',30, ...  % Default is 30
            'Shuffle','every-epoch', ...
            'ValidationData',{total_img_val,response_val}, ...
            'Verbose',false, ... %Indicator to display training progress information in the command window
            'Plots','none',...
            'ExecutionEnvironment','multi-gpu');


        %%%%%%%%%%%% Train the network
        tic
        net{iter,1} = trainNetwork(total_img_train,response_train,layers,options);
        toc

        % Accuracies
        % Validation
        YPred_val = classify(net{iter,1},total_img_val);
        YValidation = response_val;
        accuracy_val(iter,1) = sum(YPred_val == YValidation)/numel(YValidation);
        C=confusionmat(YValidation,YPred_val);
        figure;confusionchart(C,{'Adni_control','Tle_control','Alz','TLE'})

        % Test
        YPred_test = classify(net{iter,1},total_img_test);
        Ytest = response_test;
        accuracy_test(iter,1) = sum(YPred_test == Ytest)/numel(Ytest);
        C=confusionmat(Ytest,YPred_test);
        confusionchart(C,{'Adni_control','Tle_control','Alz','TLE'})

        %%%%%%%%%%% Train on shuffled labels
        tic
        net_shuff{iter,1} = trainNetwork(total_img_train,response_train(randperm(numel(response_train),numel(response_train))),layers,options);
        toc

        % Accuracies
        % Validation
        YPred_val = classify(net_shuff{iter,1},total_img_val);
        YValidation = response_val;
        accuracy_val_shuff(iter,1) = sum(YPred_val == YValidation)/numel(YValidation);
        C=confusionmat(YValidation,YPred_val);
        confusionchart(C,{'Adni_control','Tle_control','Alz','TLE'})

        % Test
        YPred_test = classify(net_shuff{iter,1},total_img_test);
        Ytest = response_test;
        accuracy_test_shuff(iter,1) = sum(YPred_test == Ytest)/numel(Ytest);
        C=confusionmat(Ytest,YPred_test);
        confusionchart(C,{'Adni_control','Tle_control','Alz','TLE'})
    end

    % Historgram of accuracy
    figure;
    hold on
    histogram(accuracy_test,'BinWidth',0.01);
    histogram(accuracy_val,'BinWidth',0.01);
    xlim([.5 1])
    legend({'Testing','Training'})
    figtitle=['CNN - middle 50 percent slices - Axial',' ',matter{m},' ',patient_side{p}];
    title(figtitle)
    xlabel('Accuracy')
    ylabel('# of models')
    saveas(gcf,fullfile(save_path,figtitle));
    save(fullfile(save_path,figtitle),'accuracy_test','accuracy_val');
    close all
    clc
end

 

%% Analyze network
analyzeNetwork(net{1})

controlimg=load_nii(controlbrain);
imgSize = size(controlimg);
imgSize = imgSize(1:2);

l=12 % ReLU

hFig = figure('Toolbar', 'none', 'Menu', 'none', 'WindowState', 'maximized'); 
for s=1:size(controlimg.img,3)
    sgtitle(['Slice # ',num2str(s)])
    pause(0.25)
    for n=1:size(net,1)
        act{n} = activations(net{n},controlimg.img(:,:,s),l);
    end

    
    if s==1
        
        con_h=nexttile;
        imshow(controlimg.img(:,:,s),'InitialMagnification','fit','Parent',con_h)
        title(con_h,'Original image')
        for a=1:numel(act)
            img = imtile(mat2gray(act{a}),'GridSize',[6 6]);
            h(a)=nexttile;
            imshow(img,'InitialMagnification','fit','Parent',h(a));
            title(h(a),['Net ',num2str(a),' - Accuracy ',num2str(accuracy_val(a))])
        end
    else
        imshow(controlimg.img(:,:,s),'InitialMagnification','fit','Parent',con_h)
        title(con_h,'Original image')
        for a=1:numel(act)
            img = imtile(mat2gray(act{a}),'GridSize',[6 6]);
            imshow(img,'InitialMagnification','fit','Parent',h(a));
            title(h(a),['Net ',num2str(a),' - Accuracy ',num2str(accuracy_val(a))])
        end
    end
end

for l=1:numel(net.Layers)
    figure('Name',net.Layers(l).Name);
    I = imtile(inputimg(l,:));
    imshow(I)
end
