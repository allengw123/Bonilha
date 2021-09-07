%% CNN MODEL
% Inputs:
 
save_path = 'C:\Users\bonilha\Documents\Project_Eleni\CNN_results_all\CNN_results_50slices_wrong_labels';
mkdir(save_path)

datapath = 'C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_all_thr\Smoothed_Files_thr_0.2';

matter={'gm','wm'};
patient_side={'left','right'};

for m=1:numel(matter)
    for p = 1:numel(patient_side)
        controlpath = fullfile(datapath,['mod_0.2_smooth10_controls_',matter{m}]);
        patientpath = fullfile(datapath,['mod_0.2_smooth10_patients_',patient_side{p},'_',matter{m}]);
        control_nii = {dir(fullfile(controlpath,'*.nii')).name}';
        patient_nii = {dir(fullfile(patientpath,'*.nii')).name}';
        
        % Load subjects imgs
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

        %% Permute

        for iter=1:100
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


            %Concatinate
            total_img_test = cat(4,control_data_test,patient_data_test);
            total_img_train = cat(4,control_data_train,patient_data_train);
            total_img_val = cat(4,control_data_val,patient_data_val);


            response_train = categorical([ones(numel(permcontroltrain),1);ones(numel(permpatienttrain),1)*2]);
            response_val = categorical([ones(numel(permcontrolval),1);ones(numel(permpatientval),1)*2]);
            response_test = categorical([ones(numel(permcontroltest),1);ones(numel(permpatienttest),1)*2]);

            ROI_traindata_wrong_labels= categorical(response_train(randperm(length(response_train))));
            ROI_valdata_wrong_labels= categorical(response_val(randperm(length(response_val))));


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
                'ValidationData',{total_img_val,ROI_valdata_wrong_labels}, ...
                'Verbose',false, ... %Indicator to display training progress information in the command window
                'Plots','none',...
                'ExecutionEnvironment','multi-gpu');

            % Train the network
            net = trainNetwork(total_img_train,ROI_traindata_wrong_labels,layers,options);

            % Accuracies

            YPred_val = classify(net,total_img_val);
            YValidation = ROI_valdata_wrong_labels;
            accuracy_val(iter,1) = sum(YPred_val == YValidation)/numel(YValidation);


            YPred_test = classify(net,total_img_test);
            Ytest = response_test;
            accuracy_test(iter,1) = sum(YPred_test == Ytest)/numel(Ytest);
    
        end

        %% Historgram of accuracy
        figure;
        hold on
        histogram(accuracy_test,'BinWidth',0.01);
        histogram(accuracy_val,'BinWidth',0.01);
        xlim([.5 1])
        legend({'Testing','Training'})
        figtitle=['CNN(wrongLabels) - middle 50 percent slices - Axial',' ',matter{m},' ',patient_side{p}];
        title(figtitle)
        xlabel('Accuracy')
        ylabel('# of models')
        saveas(gcf,fullfile(save_path,figtitle));
        save(fullfile(save_path,figtitle),'accuracy_test','accuracy_val');
        close all
        clc
    end 
end

% %% Analyze network
% analyzeNetwork(net)
% 
% im=control_data_val(:,:,:,1);
% imgSize = size(im);
% imgSize = imgSize(1:2);
% 
% layer=[2 3 4 5 6 7 8 9 10 11 12 13 14];
% for l=layer
%     name=net.Layers(l).Name;
%     act1 = activations(net,im,l);
% 
%     figure
%     [maxValue,maxValueIndex] = max(max(max(act1)));
%     act1chMax = act1(:,:,maxValueIndex);
%     act1chMax = mat2gray(act1chMax);
%     act1chMax = imresize(act1chMax,imgSize);
%     I = imtile({im,act1chMax});
%     imshow(I)
%     title(name)
% end