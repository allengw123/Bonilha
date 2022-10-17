clear all
clc

GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
IMAGE_DATABASE = '/media/bonilha/Elements/Image_database';

HIPP = true;
%%
% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

patients = dir(fullfile(IMAGE_DATABASE,'*','post_qc','Patients','*.mat'));
controls = dir(fullfile(IMAGE_DATABASE,'*','post_qc','Controls','*.mat'));

jhu_path = fullfile(GITHUB_PATH,'Toolbox','imaging','Atlas','jhu','Resliced Atlas','rJHU','r75_Hippo_L.nii');
jhu = load_nii(jhu_path);
hipp_log = jhu.img~=0;
template = load_nii('/media/bonilha/Elements/Image_database/MasterSet/harvestOutput/Patients/BONPL0103/pos/mri/mwp1eT1_BONPL0103_pos.nii');

% Patients img
patient_img = cell(numel(patients),1);
parfor pat=1:numel(patients)
    temp=load(fullfile(patients(pat).folder,patients(pat).name));
    if isfield(temp,'pre')
        side = extractAfter(patients(pat).name,'P');
        if strcmp(side(1),'R')
            flipped = flip(temp.pre.smooth_vbm_gm.dat,1);
            temp_img = flipped(:);
        else
            temp_img = temp.pre.smooth_vbm_gm.dat(:);
        end
        patient_img{pat}= temp_img';
    end
end
patient_img = cat(1,patient_img{:});

% Control img
control_img = cell(numel(controls),1);
parfor con=1:numel(controls)
    temp=load(fullfile(controls(con).folder,controls(con).name));
    if isfield(temp,'session')
        temp_img = temp.session.smooth_vbm_gm.dat(:);
        control_img{con} = temp_img';
    end
end
control_img = cat(1,control_img{:});

%%
% Permute/Run SVM
for iter=1:100
    display(['Running iteration ',num2str(iter)])

    % Permute testing
    permcontroltest = randperm(size(control_img,1),floor(size(control_img,1)*0.25));
    permpatienttest = randperm(size(patient_img,1),floor(size(patient_img,1)*0.25));


    % Permute training data
    permcontroltrain=1:size(control_img,1);
    permcontroltrain(permcontroltest)=[];

    permpatienttrain=1:size(patient_img,1);
    permpatienttrain(permpatienttest) = [];



    % Select training/testing data
    if HIPP
        log = hipp_log(:);
    else
        log = 1:size(control_img,2);
    end

    control_data_test= control_img(permcontroltest,log);
    control_data_train= control_img(permcontroltrain,log);

    patient_data_test= patient_img(permpatienttest,log);
    patient_data_train= patient_img(permpatienttrain,log);

    % Concatinate
    total_img_test = cat(1,control_data_test,patient_data_test);
    total_img_train = cat(1,control_data_train,patient_data_train);

    response_train = [ones(numel(permcontroltrain),1);ones(numel(permpatienttrain),1)*2];
    response_test = [ones(numel(permcontroltest),1);ones(numel(permpatienttest),1)*2];


    % Train SVM
    SVMModel= fitcsvm(total_img_train, response_train,'KernelFunction','linear','KFold',5);

    conmat{iter}=confusionmat(SVMModel.Y,kfoldPredict(SVMModel));
    accuracytraining(iter,1)=1-kfoldLoss(SVMModel);

    % Test SVM
    trainedModel=SVMModel;
    testingdataset=total_img_test;

    output=predict(trainedModel.Trained{1},testingdataset);
    accuracytesting(iter,1)=1-sum(output~=response_test)/numel(output);

    % Beta Weights
    for tm=1:numel(SVMModel.Trained)
        betaweights{iter,tm}=SVMModel.Trained{tm}.Beta;
    end
end

betaweights_reshape=reshape(betaweights,[],1);
betaweights_reshape_sum=sum([betaweights_reshape{:}]',1);

betaweights_recon = zeros(113,137,113);

if HIPP
    betaweights_recon(hipp_log)=betaweights_reshape_sum;
else
    betaweights_recon=betaweights_reshape_sum;
end

temp_nii = jhu; 
temp_nii.img = betaweights_recon;
temp_nii.hdr.dime.datatype = 16;
temp_nii.hdr.dime.bitpix = 16;
save_nii(temp_nii,'/home/bonilha/Downloads/betaweights.nii')

% Distribution of accuracy
figure;
hold on
histogram(accuracytesting,'BinWidth',0.01)
histogram(accuracytraining,'BinWidth',0.01)
xlim([.7 1])
legend({'Testing','Training'})
figtitle=[patient_side{p},'_    ',matter{m}];
title(figtitle)
xlabel('Accuracy')
ylabel('# of models')
saveas(gcf,fullfile(save_path,figtitle));
save(fullfile(save_path,figtitle),'conmat','accuracytraining','accuracytesting');
close all
clc
