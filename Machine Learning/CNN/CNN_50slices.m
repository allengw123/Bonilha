%% CNN MODEL

% % Inputs:
controlbrain='C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_all_thr\Smoothed_Files_thr_0.2\mod_0.2_smooth10_controls_gm\smooth10mwp1sub_7129.nii';
save_path = 'C:\Users\bonilha\Documents\Project_Eleni\CNN_results_all\CNN_results_50slices';
mkdir(save_path)

datapath = 'C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_all_thr\Smoothed_Files_thr_0.2';

matter={'gm','wm'};
patient_side={'left','right'};

for m=1:numel(matter)
    for p = 1:numel(patient_side)
        disp(['Running ',matter{m},' on the ',patient_side{p}, ' side'])
        controlpath = fullfile(datapath,['mod_0.2_smooth10_controls_',matter{m}]);
        patientpath = fullfile(datapath,['mod_0.2_smooth10_patients_',patient_side{p},'_',matter{m}]);
        control_nii = {dir(fullfile(controlpath,'*.nii')).name}';
        patient_nii = {dir(fullfile(patientpath,'*.nii')).name}';

        % Load subjects imgs
        disp('Loading subjects')
        for con=1:numel(control_nii)
            temp=load_nii(control_nii{con});
            count=1;
            for i=28:85
                temp_img{count,1}=temp.img(:,:,i);
                count=count+1;
            end
            control_img{con,1}=temp_img;
        end
        control_img=cat(1,control_img{:});
        control_img=cell2mat(control_img');
        control_img_reshape=reshape(control_img,113,137,1,[]);

        disp('Extracting 50 slices')
        for pat=1:numel(patient_nii)
            temp=load_nii(patient_nii{pat});
            count=1;
            for i=28:85
                temp_img{count,1}=temp.img(:,:,i);
                count=count+1;
            end
            patient_img{pat,1}=temp_img;
        end
        patient_img=cat(1,patient_img{:});
        patient_img=cell2mat(patient_img');
        patient_img_reshape=reshape(patient_img,113,137,1,[]);

        % Permute
        for iter=1:10
            display(['Running iteration ',num2str(iter)])
            
            % Permute testing/Validation data
            permcontroltestval = randperm(size(control_img_reshape,4),floor(size(control_img_reshape,4)*0.40));
            permpatienttestval = randperm(size(patient_img_reshape,4),floor(size(patient_img_reshape,4)*0.40));

            permcontroltest = permcontroltestval (1:floor(0.6*numel(permcontroltestval)));
            permcontrolval = permcontroltestval (floor(0.6*numel(permcontroltestval))+1:end);
            
            permpatienttest = permpatienttestval(1:floor(0.6*numel(permpatienttestval)));
            permpatientval = permpatienttestval(floor(0.6*numel(permpatienttestval))+1:end);



            % Permute training data
            permcontroltrain=1:size(control_img_reshape,4);
            permcontroltrain(permcontroltestval)=[];

            permpatienttrain=1:size(patient_img_reshape,4);
            permpatienttrain(permpatienttestval) = [];



            % Select training/testing data
            control_data_test= control_img_reshape(:,:,:,permcontroltest);
            control_data_train= control_img_reshape(:,:,:,permcontroltrain);
            control_data_val=control_img_reshape(:,:,:,permcontrolval);

            patient_data_test= patient_img_reshape(:,:,:,permpatienttest);
            patient_data_train= patient_img_reshape(:,:,:,permpatienttrain);
            patient_data_val= patient_img_reshape(:,:,:,permpatientval);

            % Concatinate 
            total_img_test = cat(4,control_data_test,patient_data_test);
            total_img_train = cat(4,control_data_train,patient_data_train);
            total_img_val = cat(4,control_data_val,patient_data_val);

            response_train = categorical([ones(numel(permcontroltrain),1);ones(numel(permpatienttrain),1)*2]);
            response_val = categorical([ones(numel(permcontrolval),1);ones(numel(permpatientval),1)*2]);
            response_test = categorical([ones(numel(permcontroltest),1);ones(numel(permpatienttest),1)*2]);


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



                fullyConnectedLayer(2)
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


            % Train the network
            tic
            net{iter,1} = trainNetwork(total_img_train,response_train,layers,options);
            toc

            % Accuracies
            YPred_val = classify(net{iter,1},total_img_val);
            YValidation = response_val;
            accuracy_val(iter,1) = sum(YPred_val == YValidation)/numel(YValidation);

            YPred_test = classify(net{iter,1},total_img_test);
            Ytest = response_test;
            accuracy_test(iter,1) = sum(YPred_test == Ytest)/numel(Ytest);
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
