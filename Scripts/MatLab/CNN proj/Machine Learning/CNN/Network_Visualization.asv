%% Network Visualization
clear
clc

% githubpath='C:\Users\allen\Documents\GitHub\Bonilha';
githubpath='C:\Users\bonilha\Documents\GitHub\Bonilha';

cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
% CNNoutput='C:\Users\allen\Box Sync\Desktop\Bonilha\Projects\ep_imaging_AI\CNN output';
CNNoutput='F:\CNN output';

cd(CNNoutput)

savepath=fullfile(CNNoutput,'Figures');mkdir(savepath);

TCmat=load(fullfile(CNNoutput,'ep_control(1) tle(2) -GM-CNN.mat'));
TCAmat=load(fullfile(CNNoutput,'ep_control(1) adni_control(1) tle(2) alz(3) -GM-CNN.mat'));

%% Disease accuracy

%%%% Historgram of accuracy (TLE v Control v Alz)

conf_stat_reg=conf_analysis(TCAmat.confmat.reg);
conf_stat_shuff=conf_analysis(TCAmat.confmat.shuff);
%%

figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100,10);
mean(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)
std(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)
histfit(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100,5);
mean(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100)
std(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100)
xlim([40 101]) 
ylim([0 90])
xlabel('Accuracy')
ylabel('# of models')
xticks([40:20:100])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)


fdic=sum((mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)>(mean(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100)))/100



figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'),10);
std(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'))
mean(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'))
histfit(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'),15);
std(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'))
mean(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'))
xlim([.10 1.01])
ylim([0 90])
xlabel('Precision')
ylabel('# of models')
xticks([0.10:0.20:1.00])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(mean(cell2mat(conf_stat_reg.Recall'),2,'omitnan'),15);
std(mean(cell2mat(conf_stat_reg.Recall'),2,'omitnan'))
mean(mean(cell2mat(conf_stat_reg.Recall'),2,'omitnan'))
histfit(mean(cell2mat(conf_stat_shuff.Recall'),2,'omitnan'),5);
std(mean(cell2mat(conf_stat_shuff.Recall'),2,'omitnan'))
mean(mean(cell2mat(conf_stat_shuff.Recall'),2,'omitnan'))
xlim([.10 1.01])
ylim([0 90])
xlabel('Recall')
ylabel('# of models')
xticks([0.10:0.20:1.00])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)

%%
figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(conf_stat_reg.Acc{1}*100);
mean(conf_stat_reg.Acc{1}*100)
std(conf_stat_reg.Acc{1}*100)
histfit(conf_stat_reg.Acc{2}*100);
mean(conf_stat_reg.Acc{2}*100)
std(conf_stat_reg.Acc{2}*100)
histfit(conf_stat_reg.Acc{3}*100);
mean(conf_stat_reg.Acc{3}*100)
std(conf_stat_reg.Acc{3}*100)
xlim([40 101]) 
ylim([0 70])
xlabel('Accuracy')
ylabel('# of models')
xticks([40:20:100])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(conf_stat_reg.Precision{1});
histfit(conf_stat_reg.Precision{2});
histfit(conf_stat_reg.Precision{3});
xlim([40 101])  
ylim([0 70]) 
xlabel('Precision')
ylabel('# of models')
xticks([0:0.2:1])
yticks([0:20:60])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
% legend('Control','TLE','Alz')

figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(conf_stat_reg.Recall{1});
histfit(conf_stat_reg.Recall{2});
histfit(conf_stat_reg.Recall{3});
xlim([40 101])  
ylim([0 70])
xlabel('Recall')
ylabel('# of models')
xticks([0:0.2:1])
yticks([0:20:60])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
% legend('Control','TLE','Alz')

%% Confounding factor

[maxAcc,maxIdx]=cellfun(@(x) max(x),TCAmat.acc_CF.reg);
tempconf=TCAmat.confmat_CF.reg;
for i=1:numel(TCAmat.confmat_CF.reg)
    tempconf{i}.C=tempconf{i}.C{maxIdx(i)};
    tempconf{i}.order=tempconf{i}.order{maxIdx(i)};
    tempconf{i}.perm=tempconf{i}.perm{maxIdx(i)};
end
conf_stat_reg=conf_analysis(tempconf);

[maxAcc,maxIdx]=cellfun(@(x) max(x),TCAmat.acc_CF.shuff);
tempconf=TCAmat.confmat_CF.shuff;
for i=1:numel(TCAmat.confmat_CF.reg)
    tempconf{i}.C=tempconf{i}.C{maxIdx(i)};
    tempconf{i}.order=tempconf{i}.order{maxIdx(i)};
    tempconf{i}.perm=tempconf{i}.perm{maxIdx(i)};
end
conf_stat_shuff=conf_analysis(tempconf);
%%
%%%% Historgram of accuracy (TLE v Control v Alz)



figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100);
mean(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)
std(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)
histfit(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100,7);
mean(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100)
std(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100)
xlim([40 101]) 
ylim([0 90])
xlabel('Accuracy')
ylabel('# of models')
xticks([40:20:100])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'),10);
mean(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'))
std(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'))
histfit(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'),15);
mean(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'))
std(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'))
xlim([10 101])
ylim([0 90])
xlabel('Precision')
ylabel('# of models')
xticks([10:20:100])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(mean(cell2mat(conf_stat_reg.Recall'),2,'omitnan'),15);
mean(mean(cell2mat(conf_stat_reg.Recall'),2,'omitnan'))
std(mean(cell2mat(conf_stat_reg.Recall'),2,'omitnan'))
histfit(mean(cell2mat(conf_stat_shuff.Recall'),2,'omitnan'),5);
mean(mean(cell2mat(conf_stat_shuff.Recall'),2,'omitnan'))
std(mean(cell2mat(conf_stat_shuff.Recall'),2,'omitnan'))
xlim([10 101])
ylim([0 90])
xlabel('Recall')
ylabel('# of models')
xticks([10:20:100])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)



%%
figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(conf_stat_reg.Acc{1}*100);
mean(conf_stat_reg.Acc{1}*100)
std(conf_stat_reg.Acc{1}*100)
histfit(conf_stat_reg.Acc{2}*100);
mean(conf_stat_reg.Acc{2}*100)
std(conf_stat_reg.Acc{2}*100)
histfit(conf_stat_reg.Acc{3}*100);
mean(conf_stat_reg.Acc{3}*100)
std(conf_stat_reg.Acc{3}*100)
xlim([40 101]) 
ylim([0 70])
xlabel('Accuracy')
ylabel('# of models')
xticks([40:20:100])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',18)


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(conf_stat_reg.Precision{1});
histfit(conf_stat_reg.Precision{2});
histfit(conf_stat_reg.Precision{3});
xlim([40 101])  
ylim([0 70]) 
xlabel('Precision')
ylabel('# of models')
xticks([0:0.2:1])
yticks([0:20:60])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
% legend('Control','TLE','Alz')

figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(conf_stat_reg.Recall{1});
histfit(conf_stat_reg.Recall{2});
histfit(conf_stat_reg.Recall{3});
xlim([40 101])  
ylim([0 70])
xlabel('Recall')
ylabel('# of models')
xticks([0:0.2:1])
yticks([0:20:60])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
% legend('Control','TLE','Alz')


%% Feature visualization (ReLU3)

PatientData='F:\PatientData';
SmoothThres=fullfile(PatientData,'smooth');
addpath(genpath(SmoothThres));

groups={'adni_alz','ep_tle',{'adni_control','ep_control'}};

% Read excel files
ep_tle_info=readtable(fullfile(PatientData,'ep_TLE_info.xlsx'));
ep_controls_info=readtable(fullfile(PatientData,'ep_controls_info.xlsx'));
ADNI_CN_info=readtable(fullfile(PatientData,'ADNI_CN_info.csv'));
ADNI_Alz_info=readtable(fullfile(PatientData,'ADNI_Alz_info.csv'));

% look for Alz nifti files
Alzfiles={dir(fullfile(SmoothThres,'Alz\ADNI_Alz_nifti','*','*.nii')).name}';

% look for TLE nifti files
tlefiles={dir(fullfile(SmoothThres,'TLE','*','*','*.nii')).name}';


% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Controls','*','*','*.nii')).name}';
controlfiles_adni=controlfiles(contains(controlfiles,'ADNI'));
controlfiles_ep=controlfiles(~contains(controlfiles,'ADNI'));

%%%%%%%%%%%%%% Load adni control %%%%%%%%%%%%%%%%%%
tempdata=controlfiles_adni(strcmp(extractBetween(controlfiles_adni,'smooth10_','_ADNI'),'GM'));
adni_control_img=[];
adni_control_age=[];
count1=0;
disp('Loading adni control subjects and extracting 50 slices')

for con=1:numel(tempdata)
    
    % Find image ID
    tempIN=extractBetween(tempdata{con},'_I','.nii');
    
    % Find subject age
    if any(strcmp(extractAfter(ADNI_CN_info.ImageDataID,'I'),tempIN))
        tempage=ADNI_CN_info.Age(strcmp(extractAfter(ADNI_CN_info.ImageDataID,'I'),tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find age for subject:%s',tempIN{:}))
        continue
    end
    
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
tempdata=controlfiles_ep(strcmp(extractBetween(controlfiles_ep,'smooth10_','_'),'GM'));
ep_control_img=[];
ep_control_age=[];
count1=0;
disp('Loading tle control subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image ID
    tempIN=extractBetween(tempdata{con},['GM','_'],'.nii');
    
    % Find subject age
    if any(strcmp(ep_controls_info.Participant,tempIN))
        tempage=ep_controls_info.Age(strcmp(ep_controls_info.Participant,tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find age for subject:%s',tempIN{:}))
        continue
    end
    
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
tempdata=Alzfiles(strcmp(extractBetween(Alzfiles,'smooth10_','_ADNI'),'GM'));
adni_alz_img=[];
adni_alz_age=[];
count1=0;
disp('Loading adni alz subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image number
    tempIN=extractBetween(tempdata{con},'_I','.nii');
    
    % Find subject age
    if any(strcmp(extractAfter(ADNI_Alz_info.ImageDataID,'I'),tempIN))
        tempage=ADNI_Alz_info.Age(strcmp(extractAfter(ADNI_Alz_info.ImageDataID,'I'),tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find subject:%s',tempIN{:}))
        continue
    end
    
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

%%%%%%%%%%%%%% Load ep TLE %%%%%%%%%%%%%%%%%%
tempdata=tlefiles(strcmp(extractBetween(tlefiles,'smooth10_','_'),'GM'));
ep_tle_img=[];
ep_tle_age=[];
count1=0;
disp('Loading tle subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image ID
    tempIN=extractBetween(tempdata{con},['GM','_'],'.nii');
    
    % Find subject age
    if any(strcmp(ep_tle_info.ID,tempIN))
        tempage=ep_tle_info.Age(strcmp(ep_tle_info.ID,tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find subject:%s',tempIN{:}))
        continue
    end
    
    % Load image
    temp=load_nii(tempdata{con});
    
    count2=1;
    for i=28:85
        temp_img{count2,1}=temp.img(:,:,i);
        count2=count2+1;
    end
    ep_tle_img{count1,1}=temp_img;
    ep_tle_age{count1,1}=tempage;
end

net=TCAmat.net.reg;
% analyzeNetwork(net{1})
for g=2:numel(groups)
    template=temp;
    template.img=zeros(112,136,113);
    if iscell(groups{g})
        tempdata=eval([groups{g}{1},'_img']);
        tempdata=[tempdata;eval([groups{g}{2},'_img'])];
        savename='controls';
    else
        tempdata=eval([groups{g},'_img']);
        savename=groups{g};
    end
    
    clear image
    clear standardDev
    for t=1:numel(tempdata)
        wkimg=tempdata{t};
        disp([num2str(t),'/',num2str(numel(tempdata))])
        
        parfor layer=1:numel(wkimg)
            inputimg=wkimg{layer};
            tempimage=[];
            for n=1:numel(net)
                act = activations(net{n},inputimg,12);
                tempimage(:,:,n)=mat2gray(double(imresize(sum(act,3),4)));
            end
            image(:,:,layer,t)=mean(tempimage,3);
            standardDev(:,:,layer,t)=std(tempimage,0,3);
        end
    end
    
    tot_image=mean(image,4);
    std_image=mean(standardDev,4);
    
    weight_img=template;
    weight_img.img(:,:,28:85)=tot_image;
    niftiwrite(weight_img.img,fullfile(savepath,[savename,'_Act.nii']));
    
    weight_img=template;
    weight_img.img(:,:,28:85)=std_image;
    niftiwrite(weight_img.img,fullfile(savepath,[savename,'_STD.nii']));
end
%% Feature visualization (occlusionSensitivity)

PatientData='F:\PatientData';
SmoothThres=fullfile(PatientData,'smooth');
addpath(genpath(SmoothThres));

groups={'adni_alz','ep_tle',{'adni_control','ep_control'}};

% Read excel files
ep_tle_info=readtable(fullfile(PatientData,'ep_TLE_info.xlsx'));
ep_controls_info=readtable(fullfile(PatientData,'ep_controls_info.xlsx'));
ADNI_CN_info=readtable(fullfile(PatientData,'ADNI_CN_info.csv'));
ADNI_Alz_info=readtable(fullfile(PatientData,'ADNI_Alz_info.csv'));

% look for Alz nifti files
Alzfiles={dir(fullfile(SmoothThres,'Alz\ADNI_Alz_nifti','*','*.nii')).name}';

% look for TLE nifti files
tlefiles={dir(fullfile(SmoothThres,'TLE','*','*','*.nii')).name}';


% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Controls','*','*','*.nii')).name}';
controlfiles_adni=controlfiles(contains(controlfiles,'ADNI'));
controlfiles_ep=controlfiles(~contains(controlfiles,'ADNI'));

%%%%%%%%%%%%%% Load adni control %%%%%%%%%%%%%%%%%%
tempdata=controlfiles_adni(strcmp(extractBetween(controlfiles_adni,'smooth10_','_ADNI'),'GM'));
adni_control_img=[];
adni_control_age=[];
count1=0;
disp('Loading adni control subjects and extracting 50 slices')

for con=1:numel(tempdata)
    
    % Find image ID
    tempIN=extractBetween(tempdata{con},'_I','.nii');
    
    % Find subject age
    if any(strcmp(extractAfter(ADNI_CN_info.ImageDataID,'I'),tempIN))
        tempage=ADNI_CN_info.Age(strcmp(extractAfter(ADNI_CN_info.ImageDataID,'I'),tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find age for subject:%s',tempIN{:}))
        continue
    end
    
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
tempdata=controlfiles_ep(strcmp(extractBetween(controlfiles_ep,'smooth10_','_'),'GM'));
ep_control_img=[];
ep_control_age=[];
count1=0;
disp('Loading tle control subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image ID
    tempIN=extractBetween(tempdata{con},['GM','_'],'.nii');
    
    % Find subject age
    if any(strcmp(ep_controls_info.Participant,tempIN))
        tempage=ep_controls_info.Age(strcmp(ep_controls_info.Participant,tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find age for subject:%s',tempIN{:}))
        continue
    end
    
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
tempdata=Alzfiles(strcmp(extractBetween(Alzfiles,'smooth10_','_ADNI'),'GM'));
adni_alz_img=[];
adni_alz_age=[];
count1=0;
disp('Loading adni alz subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image number
    tempIN=extractBetween(tempdata{con},'_I','.nii');
    
    % Find subject age
    if any(strcmp(extractAfter(ADNI_Alz_info.ImageDataID,'I'),tempIN))
        tempage=ADNI_Alz_info.Age(strcmp(extractAfter(ADNI_Alz_info.ImageDataID,'I'),tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find subject:%s',tempIN{:}))
        continue
    end
    
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

%%%%%%%%%%%%%% Load ep TLE %%%%%%%%%%%%%%%%%%
tempdata=tlefiles(strcmp(extractBetween(tlefiles,'smooth10_','_'),'GM'));
ep_tle_img=[];
ep_tle_age=[];
count1=0;
disp('Loading tle subjects and extracting 50 slices')
for con=1:numel(tempdata)
    
    % Find image ID
    tempIN=extractBetween(tempdata{con},['GM','_'],'.nii');
    
    % Find subject age
    if any(strcmp(ep_tle_info.ID,tempIN))
        tempage=ep_tle_info.Age(strcmp(ep_tle_info.ID,tempIN));
        if isnan(tempage)
            disp(sprintf('Missing age entry for subject:%s',tempIN{:}))
            continue
        elseif tempage<18
            disp(sprintf('subject %s below 18 yr old',tempIN{:}))
            continue
        end
        count1=count1+1;
    else
        disp(sprintf('Cannot find subject:%s',tempIN{:}))
        continue
    end
    
    % Load image
    temp=load_nii(tempdata{con});
    
    count2=1;
    for i=28:85
        temp_img{count2,1}=temp.img(:,:,i);
        count2=count2+1;
    end
    ep_tle_img{count1,1}=temp_img;
    ep_tle_age{count1,1}=tempage;
end

net=TCAmat.net.reg;
for g=1:numel(groups)
    template=temp;
    template.img=zeros(113,137,113);
    if iscell(groups{g})
        tempdata=eval([groups{g}{1},'_img']);
        tempdata=[tempdata;eval([groups{g}{2},'_img'])];
        savename='controls';
    else
        tempdata=eval([groups{g},'_img']);
        savename=groups{g};
    end
    
    clear image standardDev
    for t=1:numel(tempdata)
        wkimg=tempdata{t};
        disp([num2str(t),'/',num2str(numel(tempdata))])
        switch savename
            case 'controls'
                label = 1;
            case 'ep_tle'
                label = 2;
            case 'adni_alz'
                label = 3;
        end
        
        parfor layer=1:numel(wkimg)
            inputimg=wkimg{layer};
            tempimage=[];
            for n=1:numel(net)
                act = occlusionSensitivity(net{n},inputimg,categorical(label), "Stride", 10, "MaskSize", 15);
%                 displayOcc(inputimg,act)
                tempimage(:,:,n)=mat2gray(act);
            end
            image(:,:,layer,t)=mean(tempimage,3);
            standardDev(:,:,layer,t)=std(tempimage,0,3);
        end
    end
    
    tot_image=mean(image,4);
    std_image=mean(standardDev,4);
    
    weight_img=template;
    weight_img.img(:,:,28:85)=tot_image;
    niftiwrite(weight_img.img,fullfile(savepath,[savename,'_Occlusion_Act.nii']));
    
    weight_img=template;
    weight_img.img(:,:,28:85)=std_image;
    niftiwrite(weight_img.img,fullfile(savepath,[savename,'_Occlusion_STD.nii']));
end
%% Functions

function [output]=conf_analysis(conf)

nGroups=numel(conf{1}.order);

for g=1:nGroups
    for i=1:numel(conf)
        tempconf=conf{i}.C;
        try
            tempperm=conf{i}.perm(:,end);
        catch
            tempperm=conf{i}.order;
        end
        TP{tempperm(g),1}(i,1)=tempconf(g,g);
        FP{tempperm(g),1}(i,1)=sum(tempconf(:,g))-TP{tempperm(g),1}(i,1);
        FN{tempperm(g),1}(i,1)=sum(tempconf(g,:))-TP{tempperm(g),1}(i,1);
        TN{tempperm(g),1}(i,1)=sum(sum(tempconf))-TP{tempperm(g),1}(i,1)-FP{tempperm(g),1}(i,1)-FN{tempperm(g),1}(i,1);
        
        
        Precision{tempperm(g),1}(i,1)=TP{tempperm(g),1}(i,1)/(TP{tempperm(g),1}(i,1)+FP{tempperm(g),1}(i,1));
        NPR{tempperm(g),1}(i,1)=TN{tempperm(g),1}(i,1)/(TN{tempperm(g),1}(i,1)+FN{tempperm(g),1}(i,1));
        Recall{tempperm(g),1}(i,1)=TP{tempperm(g),1}(i,1)/(TP{tempperm(g),1}(i,1)+FN{tempperm(g),1}(i,1));
        Specificity{tempperm(g),1}(i,1)=TN{tempperm(g),1}(i,1)/(TN{tempperm(g),1}(i,1)+FP{tempperm(g),1}(i,1));
        AccT{tempperm(g),1}(i,1)=(TP{tempperm(g),1}(i,1)+TN{tempperm(g),1}(i,1))/sum(sum(tempconf));
        F1{tempperm(g),1}(i,1)=2*((Precision{tempperm(g),1}(i,1)*Recall{tempperm(g),1}(i,1))/(Precision{tempperm(g),1}(i,1)+Recall{tempperm(g),1}(i,1)));
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

function displayOcc(input_image,scoreMap)
figure
ax1 = axes;
ax2 = axes;
%%Create two axes
imagesc(ax1,input_image)
imagesc(ax2,scoreMap,'AlphaData',0.4);
%%Link them together
linkaxes([ax1,ax2])
%%Hide the top axes
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
%%Give each one its own colormap
colormap(ax1,'gray')
colormap(ax2,'jet')

%%Then add colorbars and get everything lined up
set([ax1,ax2],'Position',[.17 .11 .685 .815]);
cb1 = colorbar(ax1,'Position',[.05 .11 .0675 .815]);
cb2 = colorbar(ax2,'Position',[.88 .11 .0675 .815]);

end