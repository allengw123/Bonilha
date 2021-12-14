%% Network Visualization
clear
clc

githubpath='C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
CNNoutput='C:\Users\allen\Box Sync\Desktop\Bonilha\Projects\ep_imaging_AI\CNN output';
cd(CNNoutput)

savepath=fullfile(CNNoutput,'Figures');mkdir(savepath);

TCmat=load(fullfile(CNNoutput,'ep_control(1) tle(2) -GM-CNN.mat'));
TCAmat=load(fullfile(CNNoutput,'ep_control(1) adni_control(1) tle(2) alz(3) -GM-CNN.mat'));

%% Disease accuracy

%%%% Historgram of accuracy (TLE v Control)
figure('WindowState','maximized');
figtitle='TLE vs Healthy (100 models) - Disease prediction';
sgtitle(figtitle)

subplot(2,6,[1:6])
histogram(cell2mat(TCmat.acc.reg),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Reg Label')

subplot(2,6,7)
hold on
conf_stat=conf_analysis(TCmat.confmat.reg);
histogram(conf_stat.Precision{1},'BinWidth',0.025);
histogram(conf_stat.Precision{2},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Precision')
ylabel('# of models')
legend('Control','TLE')

subplot(2,6,8)
hold on
conf_stat=conf_analysis(TCmat.confmat.reg);
histogram(conf_stat.NPR{1},'BinWidth',0.025);
histogram(conf_stat.NPR{2},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Negative Predictive Rate')
ylabel('# of models')
legend('Control','TLE')

subplot(2,6,9)
hold on
conf_stat=conf_analysis(TCmat.confmat.reg);
histogram(conf_stat.Recall{1},'BinWidth',0.025);
histogram(conf_stat.Recall{2},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Recall')
ylabel('# of models')
legend('Control','TLE')

subplot(2,6,10)
hold on
conf_stat=conf_analysis(TCmat.confmat.reg);
histogram(conf_stat.Specificity{1},'BinWidth',0.025);
histogram(conf_stat.Specificity{2},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Specificity')
ylabel('# of models')
legend('Control','TLE')

subplot(2,6,11)
hold on
conf_stat=conf_analysis(TCmat.confmat.reg);
histogram(conf_stat.Acc{1},'BinWidth',0.025);
histogram(conf_stat.Acc{2},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
legend('Control','TLE')

subplot(2,6,12)
hold on
conf_stat=conf_analysis(TCmat.confmat.reg);
histogram(conf_stat.F1{1},'BinWidth',0.025);
histogram(conf_stat.F1{2},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('F1 Score')
ylabel('# of models')
legend('Control','TLE')


subplot(3,2,4)
hold on
histogram(cellfun(@(x) x.C(1,1)/sum(x.C(1,:),'all'),TCmat.confmat.shuff),'BinWidth',0.025);
histogram(cellfun(@(x) x.C(2,2)/sum(x.C(2,:),'all'),TCmat.confmat.shuff),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
legend('Control','TLE')
savefig(gcf,fullfile(savepath,figtitle));


%%
%%%% Historgram of accuracy (TLE v Control v Alz)
figure('WindowState','maximized');
set(gcf,'color','w');

hold on
histogram(cell2mat(TCAmat.acc.reg),'BinWidth',0.025);
histogram(cell2mat(TCAmat.acc.shuff),'BinWidth',0.025,'FaceColor','#631CB3');
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
xticks([0:0.2:1])
% title('CNN Reg Label')
axis square
pbaspect([1 2 1]) 
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)

%%
conf_stat=conf_analysis(TCAmat.confmat.reg);

%%
figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histogram(conf_stat.Precision{1},'BinWidth',0.025,'FaceColor','g');
histogram(conf_stat.Precision{2},'BinWidth',0.025);
histogram(conf_stat.Precision{3},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Precision')
ylabel('# of models')
xticks([0:0.2:1])
axis square
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
% legend('Control','TLE','Alz')


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histogram(conf_stat.Recall{1},'BinWidth',0.025,'FaceColor','g');
histogram(conf_stat.Recall{2},'BinWidth',0.025);
histogram(conf_stat.Recall{3},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Recall')
ylabel('# of models')
xticks([0:0.2:1])
axis square
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
legend('Control','TLE','Alz','Orientation','horizontal')


subplot(2,2,3)
axis square
hold on
h=histogram(cellfun(@(x) x.C(1,1)/sum(x.C(1,:),'all'),TCAmat.confmat.reg),'BinWidth',0.025,'FaceColor','g');
histogram(cellfun(@(x) x.C(2,2)/sum(x.C(2,:),'all'),TCAmat.confmat.reg),'BinWidth',0.025);
histogram(cellfun(@(x) x.C(3,3)/sum(x.C(3,:),'all'),TCAmat.confmat.reg),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
legend('Control','TLE','Alz')

subplot(2,2,2)
histogram(cell2mat(TCAmat.acc.shuff),'BinWidth',0.025);
axis square
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Shuffled Label')

subplot(2,2,4)
axis square
hold on
histogram(cellfun(@(x) x.C(1,1)/sum(x.C(1,:),'all'),TCAmat.confmat.shuff),'BinWidth',0.025,'FaceColor','g');
histogram(cellfun(@(x) x.C(2,2)/sum(x.C(2,:),'all'),TCAmat.confmat.shuff),'BinWidth',0.025);
histogram(cellfun(@(x) x.C(3,3)/sum(x.C(3,:),'all'),TCAmat.confmat.shuff),'BinWidth',0.025);

xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
legend('Control','TLE','Alz')
saveas(gcf,fullfile(savepath,figtitle),'epsc');


%% Confounding factor

%%%% Historgram of accuracy (TLE v Control)
figure('WindowState','maximized');
set(gcf,'color','w');
figtitle='TLE vs Healthy (100 models) - Age prediction';
sgtitle(figtitle)

subplot(1,2,1)
[maxAcc,maxIdx]=cellfun(@(x) max(x),TCmat.acc_CF.reg);
histogram(maxAcc,'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Reg Label')

subplot(1,2,2)
[maxAcc,maxIdx]=cellfun(@(x) max(x),TCmat.acc_CF.shuff);
histogram(maxAcc,'BinWidth',0.025);xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Shuffled Label')
savefig(gcf,fullfile(savepath,figtitle));



%%%% Historgram of accuracy (TLE v Control v Alz)
figure('WindowState','maximized');
set(gcf,'color','w');

figtitle='TLE vs Healthy vs Alz (100 models) - Age prediction';
sgtitle(figtitle)



hold on
[maxAcc,maxIdx]=cellfun(@(x) max(x),TCAmat.acc_CF.reg);
histogram(maxAcc,'BinWidth',0.025);
[maxAcc,maxIdx]=cellfun(@(x) max(x),TCAmat.acc_CF.shuff);
histogram(maxAcc,'BinWidth',0.025,'FaceColor','#631CB3');

xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
axis square
pbaspect([1 2 1]) 
xticks([0:0.2:1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)

subplot(1,2,2)
[maxAcc,maxIdx]=cellfun(@(x) max(x),TCAmat.acc_CF.shuff);
histogram(maxAcc,'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Shuffled Label')

savefig(gcf,fullfile(savepath,figtitle));




figure('WindowState','maximized');
set(gcf,'color','w');
hold on
[maxAcc,maxIdx]=cellfun(@(x) max(x),TCAmat.acc_CF.reg);
tempconf=TCAmat.confmat_CF.reg;  
for i=1:numel(TCAmat.confmat_CF.reg)
    tempconf{i}.C=tempconf{i}.C{maxIdx(i)}
end
conf_stat=conf_analysis(TCAmat.confmat.reg);
histogram(conf_stat.Precision{1},'BinWidth',0.025,'FaceColor','g');
histogram(conf_stat.Precision{2},'BinWidth',0.025);
histogram(conf_stat.Precision{3},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Precision')
ylabel('# of models')
xticks([0:0.2:1])
axis square
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
% legend('Control','TLE','Alz')


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histogram(conf_stat.Recall{1},'BinWidth',0.025,'FaceColor','g');
histogram(conf_stat.Recall{2},'BinWidth',0.025);
histogram(conf_stat.Recall{3},'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Recall')
ylabel('# of models')
xticks([0:0.2:1])
axis square
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
legend('Control','TLE','Alz','Orientation','horizontal')


%%

% look for Alz nifti files
Alzfiles={dir(fullfile(SmoothThres,'Alz\ADNI_Alz_nifti','*','*.nii')).name}';

% look for TLE nifti files
tlefiles={dir(fullfile(SmoothThres,'TLE','*','*','*.nii')).name}';


% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Controls','*','*','*.nii')).name}';
controlfiles_adni=controlfiles(contains(controlfiles,'ADNI'));
controlfiles_ep=controlfiles(~contains(controlfiles,'ADNI'));

%%%%%%%%%%%%%% Load adni control %%%%%%%%%%%%%%%%%%
tempdata=controlfiles_adni(strcmp(extractBetween(controlfiles_adni,'smooth10_','_ADNI'),matter{m}));
adni_control_img=[];
adni_control_age=[];
count1=0;
disp('Loading adni control subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image ID
    tempIN=extractBetween(tempdata{con},'_I','.nii');
    
    % Load Image
    temp=load_nii(tempdata{con});
    count2=1;
    for i=28:85
        temp_img{count2,1}=temp.img(:,:,i);
        count2=count2+1;
    end
    adni_control_img{count1,1}=temp_img;
    adni_control_age{count1,1}=tempage;
end

%%%%%%%%%%%%%% Load ep control %%%%%%%%%%%%%%%%%%
tempdata=controlfiles_ep(strcmp(extractBetween(controlfiles_ep,'smooth10_','_'),matter{m}));
ep_control_img=[];
ep_control_age=[];
count1=0;
disp('Loading tle control subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image ID
    tempIN=extractBetween(tempdata{con},[matter{m},'_'],'.nii');
    
    % Load image
    temp=load_nii(tempdata{con});
    count2=1;
    for i=28:85
        temp_img{count2,1}=temp.img(:,:,i);
        count2=count2+1;
    end
    ep_control_img{count1,1}=temp_img;
    ep_control_age{count1,1}=tempage;
end

%%%%%%%%%%%%%% Load adni Alz %%%%%%%%%%%%%%%%%%
tempdata=Alzfiles(strcmp(extractBetween(Alzfiles,'smooth10_','_ADNI'),matter{m}));
adni_alz_img=[];
adni_alz_age=[];
count1=0;
disp('Loading adni alz subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image number
    tempIN=extractBetween(tempdata{con},'_I','.nii');
    
    % Load image
    temp=load_nii(tempdata{con});
    count2=1;
    for i=28:85
        temp_img{count2,1}=temp.img(:,:,i);
        count2=count2+1;
    end
    adni_alz_img{count1,1}=temp_img;
    adni_alz_age{count1,1}=tempage;
end


analyzeNetwork(TCAmat.net.reg{1})

act = activations(tempnet,controlimg_smooth.img(:,:,s),2);

%%
function [output]=conf_analysis(conf)

nGroups=numel(conf{1}.order);

for g=1:nGroups
    for i=1:numel(conf)
        tempconf=conf{i}.C;
        TP{g,1}(i,1)=tempconf(g,g);
        FP{g,1}(i,1)=sum(tempconf(:,g))-TP{g,1}(i,1);
        FN{g,1}(i,1)=sum(tempconf(g,:))-TP{g,1}(i,1);
        TN{g,1}(i,1)=sum(sum(tempconf))-TP{g,1}(i,1)-FP{g,1}(i,1)-FN{g,1}(i,1);
        
        
        Precision{g,1}(i,1)=TP{g,1}(i,1)/(TP{g,1}(i,1)+FP{g,1}(i,1));
        NPR{g,1}(i,1)=TN{g,1}(i,1)/(TN{g,1}(i,1)+FN{g,1}(i,1));
        Recall{g,1}(i,1)=TP{g,1}(i,1)/(TP{g,1}(i,1)+FN{g,1}(i,1));
        Specificity{g,1}(i,1)=TN{g,1}(i,1)/(TN{g,1}(i,1)+FP{g,1}(i,1));
        Acc{g,1}(i,1)=(TP{g,1}(i,1)+TN{g,1}(i,1))/sum(sum(tempconf));
        F1{g,1}(i,1)=2*((Precision{g,1}(i,1)*Recall{g,1}(i,1))/(Precision{g,1}(i,1)+Recall{g,1}(i,1)));
    end
end


output.TP=TP;
output.FP=FP;
output.FN=FN;
output.TN=TN;
output.Precision=Precision;
output.NPR=NPR;
output.Recall=Recall;
output.Specificity=Specificity;
output.Acc=Acc;
output.F1=F1;


end

