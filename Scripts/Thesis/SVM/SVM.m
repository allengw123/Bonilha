clear all
clc

GITHUB_PATH = '/home/bonilha/Documents/GitHub/Bonilha';
nifti_files = '/media/bonilha/AllenProj/Thesis/niftifiles';
SVM_folder = '/media/bonilha/AllenProj/Thesis/SVM';

FLIP_log = true;
%% Laod niftis

% Setup correct toolboxes
cd(GITHUB_PATH)
allengit_genpath(GITHUB_PATH,'imaging')

SVM_output = fullfile(SVM_folder,'ROI_maps');
patients = dir(fullfile(nifti_files,'Patients','*pre*smooth*gm*'));
patients_files = [];
for p = 1:length(patients)
    patients_files = [patients_files;{fullfile(patients(p).folder,patients(p).name)}];
end

controls = dir(fullfile(nifti_files,'Controls','*session*smooth*gm*'));
controls_files = [];
for p = 1:length(controls)
    controls_files = [controls_files;{fullfile(controls(p).folder,controls(p).name)}];
end

[img.TLE_GM, img.TLE_GM_names] = get_volume_data(patients_files);
[img.Control_GM, img.Control_GM_names] = get_volume_data(controls_files);


temp = cell(size(img.TLE_GM,4),1);
for i = 1:size(img.TLE_GM,4)
    wk_sbj = img.TLE_GM(:,:,:,i);
    if FLIP_log
        side = extractAfter(img.TLE_GM_names{i},'P');
        side = side(1);
        if side == 'R'
            wk_sbj = flip(wk_sbj,1);
        end
    end
    temp{i} = wk_sbj(:)';
end
img.TLE_GM = cat(1,temp{:});


temp = cell(size(img.Control_GM,4),1);
for i = 1:size(img.Control_GM,4)
    wk_sbj = img.Control_GM(:,:,:,i);
    temp{i} = wk_sbj(:)';
end
img.Control_GM = cat(1,temp{:});

jhu_path = fullfile(GITHUB_PATH,'Toolbox','imaging','Atlas','jhu','Resliced_Atlas','rJHU');
ROI = {dir(fullfile(jhu_path,'*.nii')).name};

%%
% Permute/Run SVM
for r = 1:numel(ROI)
    for iter=1:100
        display(['Running iteration ',num2str(iter)])

        % Permute testing
        permcontroltest = randperm(size(img.Control_GM,1),floor(size(img.Control_GM,1)*0.25));
        permpatienttest = randperm(size(img.TLE_GM,1),floor(size(img.TLE_GM,1)*0.25));


        % Permute training data
        permcontroltrain=1:size(img.Control_GM,1);
        permcontroltrain(permcontroltest)=[];

        permpatienttrain=1:size(img.TLE_GM,1);
        permpatienttrain(permpatienttest) = [];



        % Select training/testing data
        jhu = load_nii(fullfile(jhu_path,ROI{r}));
        ROI_og = jhu.img~=0;
        log = ROI_og;


        control_data_test= img.Control_GM(permcontroltest,log);
        control_data_train= img.Control_GM(permcontroltrain,log);

        patient_data_test= img.TLE_GM(permpatienttest,log);
        patient_data_train= img.TLE_GM(permpatienttrain,log);

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

    betaweights_recon(ROI_og)=betaweights_reshape_sum;
    

    % Distribution of accuracy
    figure;
    hold on
    histogram(accuracytesting,'BinWidth',0.01)
    histogram(accuracytraining,'BinWidth',0.01)
    xlim([.7 1])
    legend({'Testing','Training'})
    figtitle=[extractBefore(ROI{r},'.nii')];
    title(figtitle,'Interpreter','none')
    xlabel('Accuracy')
    ylabel('# of models')
    saveas(gcf,fullfile(SVM_output,figtitle));
    save(fullfile(SVM_output,figtitle),'conmat','accuracytraining','accuracytesting');

    temp_nii = jhu;
    temp_nii.img = betaweights_recon;
    save_nii_float(temp_nii,fullfile(SVM_output,[figtitle,'betaweights.nii']))

    close all
    clc
end

%% Analysis

%%%% Accuracy map

% Find files
ROI_mats = dir(fullfile(SVM_output,'*.mat'));
ROI_nii = dir(fullfile(SVM_output,'*.nii'));

% Create Mask
mask = fullfile(ROI_nii(1).folder,ROI_nii(1).name);
mask = load_nii(mask);
mask.img = zeros(size(mask.img));

acc_mask = mask;
beta_mask = mask;



for r = 1:numel(ROI_mats)

    % Load ROI SVM results
    ROI = load(fullfile(ROI_mats(r).folder,ROI_mats(r).name));
    acc = mean(ROI.accuracytesting);
    
    % Load ROI beta weights
    ROI_nii = load_nii(strrep(fullfile(ROI_mats(r).folder,ROI_mats(r).name),'.mat','betaweights.nii'));
    
    % Find ROI logic mask
    ROI_log = ROI_nii.img ~= 0;

    % Save weighted beta mask
    beta_mask.img = double(beta_mask.img) + double((ROI_nii.img*acc));

    % Save accuracy mask
    acc_mask.img(ROI_log) = acc;
end
save_nii(acc_mask,fullfile(SVM_folder,'ROI_acc.nii'));
save_nii_float(beta_mask,fullfile(SVM_folder,'ROI_betaweights.nii'));


%%

function [X, X_names,N] = get_volume_data(ff)
count = 1;
for i = 1:numel(ff)
    N = load_nii(ff{i});
    X(:,:,:,count) = N.img;
    [~,sbjname,~] = fileparts(ff{i});
    X_names{count,1} =sbjname;
    count = count + 1;
end
end
