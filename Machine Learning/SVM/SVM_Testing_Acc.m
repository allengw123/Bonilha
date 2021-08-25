datapath='C:\Users\bonilha\Documents\Project_Eleni\SVM_results_all\SVM_results';

ROI_mat_names.left={dir(fullfile(datapath,'*Left TLE.mat')).name};
ROI_mat_names.right={dir(fullfile(datapath,'*Right TLE.mat')).name};

figure;
hold on
xlabel('Test Accuracy','Fontsize',20)
ylabel('# of models','Fontsize',20)
title('Left TLE - SVM Accuracies for each ROI','Fontsize',25)
for r=1:numel(ROI_mat_names.left)
    tempdat=load(fullfile(datapath,ROI_mat_names.left{r}));
    tempaccuracytest=tempdat.accuracytesting;
    histogram(tempaccuracytest,'BinWidth',0.01)
end
legend(extractBefore(ROI_mat_names.left,'.mat'))
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)

figure;
hold on
xlabel('Test Accuracy','Fontsize',20)
ylabel('# of models','Fontsize',20)
title('Right TLE - SVM Accuracies for each ROI','Fontsize',25)
for r=1:numel(ROI_mat_names.right)
    tempdat=load(fullfile(datapath,ROI_mat_names.right{r}));
    tempaccuracytest=tempdat.accuracytesting;
    histogram(tempaccuracytest,'BinWidth',0.01)
end
legend(extractBefore(ROI_mat_names.right,'.mat'))
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)