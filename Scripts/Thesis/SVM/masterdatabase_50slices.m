clear all
clc

GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
IMAGE_DATABASE = '/media/bonilha/Elements/Image_database';

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

patients = dir(fullfile(IMAGE_DATABASE,'*','post_qc','Patients','*.mat'));
controls = dir(fullfile(IMAGE_DATABASE,'*','post_qc','Controls','*.mat'));

ROI_name.gm.left={'parahippocampal gyrus left','amygdala left','thalamus left','hippocampus left'};
ROI_idx.gm.left={45,73,83,75};

jhu_path = fullfile(GITHUB_PATH,'Toolbox','imaging','Atlas','jhu','jhu.nii');
jhu = load_nii(jhu_path);
thal_log = jhu.img == 75;

template = load_nii('/media/bonilha/Elements/Image_database/MasterSet/harvestOutput/Patients/BONPL0103/pos/mri/mwp1eT1_BONPL0103_pos.nii');


% Patients img
patient_img = cell(numel(patients),1);
parfor pat=1:numel(patients)
    temp=load(fullfile(patients(pat).folder,patients(pat).name));
    if isfield(temp,'pre')
        side = extractAfter(patients(pat).name,'P');
        if strcmp(side(1),'R')
            flipped = flip(temp.pre.vbm_gm.dat,1);
            temp_img = flipped(thal_log);
        else
            temp_img = temp.pre.vbm_gm.dat(thal_log);
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
        temp_img = temp.session.vbm_gm.dat(thal_log);
        control_img{con} = temp_img';
    end
end
control_img = cat(1,control_img{:});


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
    control_data_test= control_img(permcontroltest,:);
    control_data_train= control_img(permcontroltrain,:);

    patient_data_test= patient_img(permpatienttest,:);
    patient_data_train= patient_img(permpatienttrain,:);

    % Concatinate
    total_img_test = cat(1,control_data_test,patient_data_test);
    total_img_train = cat(1,control_data_train,patient_data_train);

    response_train = [ones(numel(permcontroltrain),1);ones(numel(permpatienttrain),1)*2];
    response_test = [ones(numel(permcontroltest),1);ones(numel(permpatienttest),1)*2];


    % Train SVM
    SVMModel= fitcsvm(array2table(total_img_train), response_train,'KernelFunction','linear','KFold',5);

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
betaweights_recon(thal_log)=betaweights_reshape_sum*1000;

betaweights_nii = jhu;
betaweights_nii.img = betaweights_recon;
save_nii(betaweights_nii,'/home/bonilha/Downloads/betaweights.nii')


thal = template;
thal.img = thal_log;
save_nii(thal,'/home/bonilha/Downloads/thal.nii')

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
