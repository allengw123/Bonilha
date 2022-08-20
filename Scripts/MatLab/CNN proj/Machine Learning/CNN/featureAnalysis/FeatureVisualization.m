%% Network Visualization
clear
clc

githubpath='C:\Users\allen\Documents\GitHub\Bonilha';
% githubpath='C:\Users\bonilha\Documents\GitHub\Bonilha';

cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
% CNNoutput='C:\Users\allen\Box Sync\Desktop\Bonilha\Projects\ep_imaging_AI\CNN output';
% CNNoutput='F:\CNN output';
CNNoutput='F:\CNN output\2D_CNN\MATLAB\AgeRegress';
% CNNoutput='F:\CNN output';

cd(CNNoutput)

savepath=fullfile(CNNoutput,'Figures');mkdir(savepath);

% TCmat=load(fullfile(CNNoutput,'ep_control(1) tle(2) -GM-CNN.mat'));
% TCAmat=load(fullfile(CNNoutput,'ep_control(1) adni_control(1) tle(2) alz(3) -GM-CNN.mat'));
TCAmat=load(fullfile(CNNoutput,'AgeRegress_GM_ADTLEHC_CNN.mat'));

%% Feature visualization (ReLU3)

FLIP = true;

PatientData='F:\PatientData\smallSet';
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
tlefiles_R = {dir(fullfile(SmoothThres,'TLE','EP_RTLE_nifti','*','*.nii')).name}';
tlefiles_L = {dir(fullfile(SmoothThres,'TLE','EP_LTLE_nifti','*','*.nii')).name}';
tlefiles = [tlefiles_R;tlefiles_L];
tleside = [ones(numel(tlefiles_R),1);ones(numel(tlefiles_L),1)*2];

% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Controls','*','*','*.nii')).name}';
controlfiles_adni=controlfiles(contains(controlfiles,'ADNI'));
controlfiles_ep=controlfiles(~contains(controlfiles,'ADNI'));

% Dedicate Side Var
side = [];

%%%%%%%%%%%%%% Load adni control %%%%%%%%%%%%%%%%%%
tempdata=controlfiles_adni(strcmp(extractBetween(controlfiles_adni,'smooth10_','_ADNI'),'GM'));
adni_control.img=[];
adni_control.age=[];
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
    adni_control.img{count1,1}=temp_img;
    adni_control.age{count1,1}=tempage;
    adni_control.side{count1,1} = 0;
end

%%%%%%%%%%%%%% Load ep control %%%%%%%%%%%%%%%%%%
tempdata=controlfiles_ep(strcmp(extractBetween(controlfiles_ep,'smooth10_','_'),'GM'));
ep_control.img=[];
ep_control.age=[];
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
    ep_control.img{count1,1}=temp_img;
    ep_control.age{count1,1}=tempage;
    ep_control.side{count1,1} = 0;
end

%%%%%%%%%%%%%% Load adni Alz %%%%%%%%%%%%%%%%%%
tempdata=Alzfiles(strcmp(extractBetween(Alzfiles,'smooth10_','_ADNI'),'GM'));
adni_alz.img=[];
adni_alz.age=[];
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
    adni_alz.img{count1,1}=temp_img;
    adni_alz.age{count1,1}=tempage;
    adni_alz.side{count1,1} = 0;
end

%%%%%%%%%%%%%% Load ep TLE %%%%%%%%%%%%%%%%%%
matter_idx = strcmp(extractBetween(tlefiles,'smooth10_','_'),'GM');
tempdata=tlefiles(matter_idx);
tempside=tleside(matter_idx);

ep_tle.img=[];
ep_tle.age=[];
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
    ep_tle.img{count1,1}=temp_img;
    ep_tle.age{count1,1}=tempage;
    ep_tle.side{count1,1} = tempside(con);
end

net=TCAmat.net.reg;
% analyzeNetwork(net{1})
for g=1:numel(groups)
    template=temp;
    template.img=zeros(112,136,113);
    if iscell(groups{g})
        tempdata=[eval([groups{g}{1},'.img']);eval([groups{g}{2},'.img'])];
        savename='controls';
        tempside = [eval([groups{g}{1},'.side']);eval([groups{g}{2},'.side'])];
    else
        tempdata=eval([groups{g},'.img']);
        savename=groups{g};
        tempside = eval([groups{g},'.side']);
    end
    
    clear image
    clear standardDev
    clear image_side
    for t=1:numel(tempdata)
        wkimg=tempdata{t};
        disp([num2str(t),'/',num2str(numel(tempdata))])
        
        parfor layer=1:numel(wkimg)
            inputimg=wkimg{layer};
            tempimage=[];
            tempimage_side = [];
            for n=1:numel(net)
                act = activations(net{n},inputimg,12);
                tempimage(:,:,n)=mat2gray(double(imresize(sum(act,3),4)));
                if FLIP
                    % If right else...
                    if tempside{t} == 0 || tempside{t} == 1
                        tempimage_side(:,:,n) = tempimage(:,:,n);
                        tempimage_side(56:end,:,n) = 1000;
                    else
                        tempimage(:,:,n) = flip(tempimage(:,:,n),1);
                        tempimage_side(:,:,n) = tempimage(:,:,n);
                        tempimage_side(56:end,:,n) = 1000;
                    end

                end
            end
            image(:,:,layer,t)=mean(tempimage,3);
            standardDev(:,:,layer,t)=std(tempimage,0,3);
            image_side(:,:,layer,t)=mean(tempimage_side,3);
        end
    end
    
    tot_image=mean(image,4);
    std_image=mean(standardDev,4);
    tot_side = mean(image_side,4);
    
    weight_img=template;
    weight_img.img(:,:,28:85)=tot_image;
    niftiwrite(weight_img.img,fullfile(savepath,[savename,'_Act.nii']));
    
    weight_img=template;
    weight_img.img(:,:,28:85)=std_image;
    niftiwrite(weight_img.img,fullfile(savepath,[savename,'_STD.nii']));

    weight_img=template;
    weight_img.img(:,:,28:85)=tot_side;
    niftiwrite(weight_img.img,fullfile(savepath,['side.nii']));
end
%% Feature visualization (occlusionSensitivity)

FLIP = true;

PatientData='F:\PatientData\smallSet';
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
tlefiles_R = {dir(fullfile(SmoothThres,'TLE','EP_RTLE_nifti','*','*.nii')).name}';
tlefiles_L = {dir(fullfile(SmoothThres,'TLE','EP_LTLE_nifti','*','*.nii')).name}';
tlefiles = [tlefiles_R;tlefiles_L];
tleside = [ones(numel(tlefiles_R),1);ones(numel(tlefiles_L),1)*2];

% look for control nifti files
controlfiles={dir(fullfile(SmoothThres,'Controls','*','*','*.nii')).name}';
controlfiles_adni=controlfiles(contains(controlfiles,'ADNI'));
controlfiles_ep=controlfiles(~contains(controlfiles,'ADNI'));

% Dedicate Side Var
side = [];

%%%%%%%%%%%%%% Load adni control %%%%%%%%%%%%%%%%%%
tempdata=controlfiles_adni(strcmp(extractBetween(controlfiles_adni,'smooth10_','_ADNI'),'GM'));
adni_control.img=[];
adni_control.age=[];
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
    adni_control.img{count1,1}=temp_img;
    adni_control.age{count1,1}=tempage;
    adni_control.side{count1,1} = 0;
end

%%%%%%%%%%%%%% Load ep control %%%%%%%%%%%%%%%%%%
tempdata=controlfiles_ep(strcmp(extractBetween(controlfiles_ep,'smooth10_','_'),'GM'));
ep_control.img=[];
ep_control.age=[];
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
    ep_control.img{count1,1}=temp_img;
    ep_control.age{count1,1}=tempage;
    ep_control.side{count1,1} = 0;
end

%%%%%%%%%%%%%% Load adni Alz %%%%%%%%%%%%%%%%%%
tempdata=Alzfiles(strcmp(extractBetween(Alzfiles,'smooth10_','_ADNI'),'GM'));
adni_alz.img=[];
adni_alz.age=[];
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
    adni_alz.img{count1,1}=temp_img;
    adni_alz.age{count1,1}=tempage;
    adni_alz.side{count1,1} = 0;
end

%%%%%%%%%%%%%% Load ep TLE %%%%%%%%%%%%%%%%%%
matter_idx = strcmp(extractBetween(tlefiles,'smooth10_','_'),'GM');
tempdata=tlefiles(matter_idx);
tempside=tleside(matter_idx);

ep_tle.img=[];
ep_tle.age=[];
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
    ep_tle.img{count1,1}=temp_img;
    ep_tle.age{count1,1}=tempage;
    ep_tle.side{count1,1} = tempside(con);
end

net=TCAmat.net.reg;
for g=1:numel(groups)
    template=temp;
    template.img=zeros(113,137,113);
    if iscell(groups{g})
        tempdata=[eval([groups{g}{1},'.img']);eval([groups{g}{2},'.img'])];
        savename='controls';
    else
        tempdata=eval([groups{g},'.img']);
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
                %%
                displayOcc(inputimg,act)
                %%
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

%% RELU3 AGE REGRESS
FLIP = true;

groups={'AD','TLE','Control'};
residual_imgs = load('F:\CNN output\2D_CNN\MATLAB\AgeRegress\residual_imgs.mat');
residual_imgs = residual_imgs.reshaped_residuals;
residual_disease_label =  load('F:\CNN output\2D_CNN\MATLAB\AgeRegress\disease_label.mat');
residual_disease_label = residual_disease_label.disease;
residual_side =  load('F:\CNN output\2D_CNN\MATLAB\AgeRegress\side_label.mat');
residual_side = residual_side.side;
net=TCAmat.net.reg;
% analyzeNetwork(net{1})
for g=1:numel(groups)
    template=load_nii('F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\Example.nii');
    template.img=zeros(113,137,113);
    savename=groups{g};
    switch savename
        case 'AD'
            d_label = 3;
        case 'TLE'
            d_label = 2;
        case 'Control'
            d_label = 1;
    end
    
    d_idx = residual_disease_label == d_label;
    tempdata = cellfun(@(x) reshape(x,[113,137,58]),residual_imgs(d_idx),'UniformOutput',false);
    tempside = residual_side(d_idx);
    clear image
    clear standardDev
    for t=1:numel(tempdata)
        wkimg=tempdata{t};
        disp([num2str(t),'/',num2str(numel(tempdata))])
        wkside = tempside(t);
        parfor layer=1:size(wkimg,3)
            inputimg=wkimg(:,:,layer);
            tempimage=[];
            for n=1:numel(net)
                act = activations(net{n},inputimg,12);
                tempimage(:,:,n)=mat2gray(double(imresize(sum(act,3),4)));
                if FLIP
                    % If right else...
                    if wkside == 0 ||wkside == 1
                    else
                        tempimage(:,:,n) = flip(tempimage(:,:,n),1);
                    end
                end
            end
            image(:,:,layer,t)=mean(tempimage,3);
            standardDev(:,:,layer,t)=std(tempimage,0,3);
        end
    end

    tot_image=mean(image,4);
    std_image=mean(standardDev,4);

    tot_image = imresize3(tot_image,[113 137,58]) ;
    std_image = imresize3(std_image,[113 137,58]) ;

    weight_img=template;
    weight_img.img(:,:,28:85)=tot_image;
    niftiwrite(weight_img.img,fullfile(savepath,[savename,'_Act.nii']));
    
    weight_img=template;
    weight_img.img(:,:,28:85)=std_image;
    niftiwrite(weight_img.img,fullfile(savepath,[savename,'_STD.nii']));
end

%% Occulsion sensitivity (Age Regress)

FLIP = true;

groups={'AD','TLE','Control'};
residual_imgs = load('F:\CNN output\2D_CNN\MATLAB\AgeRegress\residual_imgs.mat');
residual_imgs = residual_imgs.reshaped_residuals;
residual_disease_label =  load('F:\CNN output\2D_CNN\MATLAB\AgeRegress\disease_label.mat');
residual_disease_label = residual_disease_label.disease;
residual_side =  load('F:\CNN output\2D_CNN\MATLAB\AgeRegress\side_label.mat');
residual_side = residual_side.side;
net=TCAmat.net.reg;

for g=1:numel(groups)
    template=load_nii('F:\CNN output\2D_CNN\MATLAB\AgeRegress\Figures\Example.nii');
    template.img=zeros(113,137,113);
    savename=groups{g};
    switch savename
            case 'AD'
                label = 3;
            case 'TLE'
                label = 2;
            case 'Control'
                label = 1;
    end

    d_idx = residual_disease_label == label;
    tempdata = cellfun(@(x) reshape(x,[113,137,58]),residual_imgs(d_idx),'UniformOutput',false);    
    tempside = residual_side(d_idx);

    clear image standardDev
    for t=1:numel(tempdata)
        wkimg=tempdata{t};
        disp([num2str(t),'/',num2str(numel(tempdata))])
        wkside = tempside(t);

        parfor layer=1:size(wkimg,3)
            inputimg=wkimg(:,:,layer);
            tempimage=[];
            for n=1:numel(net)
                act = occlusionSensitivity(net{n},inputimg,categorical(label), "Stride", 10, "MaskSize", 15);
                    
                %displayOcc(inputimg,act)
                
                tempimage(:,:,n)=mat2gray(act);
                if FLIP
                    % If right else...
                    if wkside == 0 ||wkside == 1
                    else
                        tempimage(:,:,n) = flip(tempimage(:,:,n),1);
                    end
                end
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
    
    figure
    scoreMap(mat2gray(scoreMap)<0.95) = 0;
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