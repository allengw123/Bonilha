%% CNN MODEL

% % Inputs:

% https://www.mathworks.com/help/matlab/ref/matlab.io.datastore.imagedatastore.html

% https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html

% https://www.mathworks.com/help/deeplearning/ref/augmentedimagedatastore.html  

% https://www.mathworks.com/help/deeplearning/ref/augmentedimagedatastore.html

 

%augmenter = imageDataAugmenter();

controlpath = 'C:\Users\allen\Box Sync\Eleni\Smoothed_Files_thr_0.2\mod_0.2_smooth10_controls_gm';
patientpath = 'C:\Users\allen\Box Sync\Eleni\Smoothed_Files_thr_0.2\mod_0.2_smooth10_patients_left_gm';

 

control_nii = {dir(fullfile(controlpath,'*.nii')).name}';
patient_nii = {dir(fullfile(patientpath,'*.nii')).name}';

 

%% Load subjects imgs

for con=1:numel(control_nii)
    temp=load_nii(control_nii{con});
    control_img_1(:,:,1,con)=temp.img(:,:,30);
    control_img_2(:,:,1,con)=temp.img(:,:,31);
    control_img_3(:,:,1,con)=temp.img(:,:,43);
end

control_img=cat(4,control_img_1,control_img_2,control_img_3);

 

for pat=1:numel(patient_nii)
    temp=load_nii(patient_nii{pat});
    patient_img_1(:,:,1,pat)=temp.img(:,:,30);
    patient_img_2(:,:,1,pat)=temp.img(:,:,31);
    patient_img_3(:,:,1,pat)=temp.img(:,:,43);
end

patient_img=cat(4,patient_img_1,patient_img_2,patient_img_3);

%% Permute

% Permute testing/Validation data
permcontroltestval = randperm(numel(control_nii)*3,floor(numel(control_nii)*3*0.40));
permpatienttestval = randperm(numel(patient_nii)*3,floor(numel(patient_nii)*3*0.40));

 

permcontroltest = permcontroltestval (1:floor(0.6*numel(permcontroltestval)));
permcontrolval = permcontroltestval (floor(0.6*numel(permcontroltestval))+1:end);

 

permpatienttest = permpatienttestval(1:floor(0.6*numel(permpatienttestval)));
permpatientval = permpatienttestval(floor(0.6*numel(permpatienttestval))+1:end);

 

% Permute training data
permcontroltrain=1:numel(control_nii)*3;
permcontroltrain(permcontroltestval)=[];

 

permpatienttrain=1:numel(patient_nii)*3;
permpatienttrain(permpatienttestval) = [];

 

%% Select training/testing data

control_data_test= control_img(:,:,:,permcontroltest);
control_data_train= control_img(:,:,:,permcontroltrain);
control_data_val=control_img(:,:,:,permcontrolval);

 

patient_data_test= patient_img(:,:,:,permpatienttest);
patient_data_train= patient_img(:,:,:,permpatienttrain);
patient_data_val= patient_img(:,:,:,permpatientval);

 

%%

total_img_test = cat(4,control_data_test,patient_data_test);
total_img_train = cat(4,control_data_train,patient_data_train);
total_img_val = cat(4,control_data_val,patient_data_val);

 

response_train = categorical([ones(numel(permcontroltrain),1);ones(numel(permpatienttrain),1)*2]);
response_val = categorical([ones(numel(permcontrolval),1);ones(numel(permpatientval),1)*2]);
response_test = categorical([ones(numel(permcontroltest),1);ones(numel(permpatienttest),1)*2]);

 



%% Parameters for the network 
imageSize = [113 137 1];

%datastore = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);

 
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

   

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

 

options = trainingOptions('sgdm', ... %stochastic gradient descent with momentum(SGDM) optimizer
    'InitialLearnRate',0.01, ...
    'MaxEpochs',30, ... % Default is 30
    'Shuffle','every-epoch', ...
    'ValidationData',{total_img_val,response_val}, ...
    'Verbose',true, ... %Indicator to display training progress information in the command window
    'Plots','training-progress');

 

%% Train the network 

net = trainNetwork(total_img_train,response_train,layers,options);

%analyzeNetwork(net)

 

%% Accuracies 

YPred_val = classify(net,total_img_val);
YValidation = response_val;
accuracy_val = sum(YPred_val == YValidation)/numel(YValidation)

 

YPred_test = classify(net,total_img_test);
Ytest = response_test;
accuracy_test = sum(YPred_test == Ytest)/numel(Ytest)