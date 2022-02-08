%% SVM MODEL

% % Inputs:

standard_brain_gm = 'C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_all_thr\Smoothed_Files_thr_0.2\mod_0.2_smooth10_controls_gm\smooth10mwp1CON001.nii';
standard_brain_wm = '"C:\Users\bonilha\Documents\Project_Eleni\Smoothed_Files_all_thr\Smoothed_Files_thr_0.2\mod_0.2_smooth10_controls_wm\smooth10mwp2CON001.nii"';

save_path = 'C:\Users\bonilha\Documents\Project_Eleni\SVM_results_all\SVM_results_50slices';
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
        control_img=[];
        for con=1:numel(control_nii)
            temp=load_nii(control_nii{con});
            count=1;
            temp_img=[];
            for i=28:85
                temp_img{count,1}=reshape(temp.img(:,:,i),1,[]);
                count=count+1;
            end
            control_img{con,1}=temp_img;
        end
        control_img=cat(1,control_img{:});
        control_img=cell2mat(control_img);
        
        patient_img=[];
        for pat=1:numel(patient_nii)
            temp=load_nii(patient_nii{pat});
            count=1;
            temp_img =[];
            for i=28:85
                temp_img{count,1}=reshape(temp.img(:,:,i),1,[]);
                count=count+1;
            end
            patient_img{pat,1}=temp_img;
        end
        patient_img=cat(1,patient_img{:});
        patient_img=cell2mat(patient_img);

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

        betaweights_2D=reshape(betaweights_reshape_sum,113,[]);
        
        if m==1
            st_brain = load_nii(standard_brain_gm);
            for i=28:85
                st_brain.img(:,:,i)= (st_brain.img(:,:,i) .* betaweights_2D);
            end
        elseif m==2
            st_brain = load_nii(standard_brain_wm);
            for i=28:85
                st_brain.img(:,:,i)= (st_brain.img(:,:,i) .* betaweights_2D);
            end
        end

        % Use save_nii function
        figure;
        imagesc(betaweights_2D);
        saveas(gcf,fullfile(save_path,['Beta_brain_slice ',patient_side{p},'_',matter{m},'.fig']));
   
        temp_nii = st_brain; 
        temp_nii.hdr.dime.datatype = 16;
        temp_nii.hdr.dime.bitpix = 16;
        save_nii(temp_nii,fullfile(save_path,[patient_side{p},'_',matter{m},'.nii']));
        
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
    end 
end 