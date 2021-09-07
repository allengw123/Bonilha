% Linear SVM
%--------------------------------------------------------------------------
% Notes:
% In this script, JHU atlas is used and the hippocampus area is selected.
%--------------------------------------------------------------------------
% Inputs:
data_path='/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/Smoothed_Files_thr_0.2';
save_path='/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/SVM_resutls_wrong_labels';

patient_name={'Left TLE','Right TLE'};
patient_side={'left','right'};

whitematteratlas='/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/Atlases/rJHU-WhiteMatter-labels-1mm.nii';
greymatteratlas='/Users/elenibougioukli/Downloads/MUSC/epilepsy_project/Atlases/rJHU_altas/rjhu.nii';

ROI_name.gm.left={'parahippocampal gyrus left','amygdala left','thalamus left','hippocampus left'};
ROI_idx.gm.left={45,73,83,75};

ROI_name.gm.right={'parahippocampal gyrus right','amygdala right','thalamus right','hippocampus right'};
ROI_idx.gm.right={46,74,84,76};


ROI_name.wm.left={'corpus callosum(splenum)','fornix(column and body)','posterior limb of the internal capsule (left)','superior longitudinal fasciculus (left)'};
ROI_idx.wm.left={5,6,20,42};

ROI_name.wm.right={'corpus callosum(splenum)','fornix(column and body)','posterior limb of the internal capsule (right)','superior longitudinal fasciculus (right)'};
ROI_idx.wm.right={5,6,19,41};

brain_matter={'gm','wm'};

%--------------------------------------------------------------------------

for TLE_side=1:numel(patient_name)    
    for matter=1:numel(brain_matter)
        if matter==1
            atlas=load_nii(greymatteratlas);
        else
            atlas=load_nii(whitematteratlas);
        end
       
        if TLE_side==1
            temproi_name=ROI_name.(brain_matter{matter}).left;
            temproi_idx=ROI_idx.(brain_matter{matter}).left;
        else
            temproi_name=ROI_name.(brain_matter{matter}).right;
            temproi_idx=ROI_idx.(brain_matter{matter}).right;
        end
       
        for ROI=1:numel(temproi_name)
            ROI_nii_log=atlas.img==temproi_idx{ROI};
           
            control_path=fullfile(data_path,['mod_0.2_smooth10_controls_',brain_matter{matter}]);
            patient_path=fullfile(data_path,['mod_0.2_smooth10_patients_',patient_side{TLE_side},'_',brain_matter{matter}]);
           %%
            [SVMModel,accuracytraining,accuracytesting]=runsvm(ROI_nii_log,control_path,patient_path,temproi_name{ROI},patient_name{TLE_side},save_path);
           
        end
    end
end

%% Functions
function [SVMModel,accuracytraining,accuracytesting]=runsvm(ROI_nii_log,control_path,patient_path,ROI_name,patient_name,save_path)
   
    % Load subjects
    control_nii = {dir(fullfile(control_path,'*.nii')).name}';
    patient_nii = {dir(fullfile(patient_path,'*.nii')).name}';

    % Control
    for c = 1:numel(control_nii)
        tempnii = load_nii(control_nii{c});
        temproinii = tempnii.img(ROI_nii_log);

        control_roi_data(c,:) = temproinii';
    end


    % Patients
    for p = 1:numel(patient_nii)
        tempnii = load_nii(patient_nii{p});
        temproinii = tempnii.img(ROI_nii_log);

        patient_roi_data(p,:) = temproinii';
    end

    % Run SVM

    for svm_count=1:1000
        display(['Running model ',num2str(svm_count)]);
        % Permute testing data
        permcontroltest = randperm(numel(control_nii),floor(numel(control_nii)*0.25));
        permpatienttest = randperm(numel(patient_nii),floor(numel(patient_nii)*0.25));

        % Permute training data
        permcontroltrain=1:numel(control_nii);
        permcontroltrain(permcontroltest)=[];

        permpatienttrain=1:numel(patient_nii);
        permpatienttrain(permpatienttest) = [];


        % Organize train data
        ROI_traindata.label = ROI_name;
        ROI_traindata.all = [control_roi_data(permcontroltrain,:);patient_roi_data(permpatienttrain,:)];
        ROI_traindata.ident = [ones(numel(permcontroltrain),1);ones(numel(permpatienttrain),1)*2];
        ROI_traindata_wrong_labels=ROI_traindata.ident(randperm(length(ROI_traindata.ident)));

        % Organize test data
        ROI_testdata.label = ROI_name;
        ROI_testdata.all =  [control_roi_data(permcontroltest,:);patient_roi_data(permpatienttest,:)];
        ROI_testdata.ident = [ones(numel(permcontroltest),1);ones(numel(permpatienttest),1)*2];

        % Train SVM
        SVMModel= fitcsvm(array2table(ROI_traindata.all), ROI_traindata_wrong_labels,'KernelFunction','linear','KFold',5);
       
        conmat{svm_count}=confusionmat(SVMModel.Y,kfoldPredict(SVMModel));
        accuracytraining(svm_count,1)=1-kfoldLoss(SVMModel);

        % Test SVM
        trainedModel=SVMModel;
        testingdataset=ROI_testdata.all;

        output=predict(trainedModel.Trained{1},testingdataset);
        accuracytesting(svm_count,1)=1-sum(output~=ROI_testdata.ident)/numel(output);
    end

    % Distribution of accuracy
    figure;
    hold on
    histogram(accuracytesting,'BinWidth',0.01)
    histogram(accuracytraining,'BinWidth',0.01)
    xlim([.3 .9])
    legend({'Testing','Training'})
    figtitle=[ROI_name,'-',patient_name];
    title(figtitle)
    xlabel('Accuracy')
    ylabel('# of models')
    saveas(gcf,fullfile(save_path,figtitle));
    save(fullfile(save_path,figtitle),'conmat','accuracytraining','accuracytesting');
end