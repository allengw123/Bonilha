%% Network Visualization
clear
clc

githubpath='C:\Users\allen\Documents\GitHub\Bonilha';
cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
CNNoutput='C:\Users\allen\Box Sync\Desktop\Allen_Bonilha_EEG\Projects\ep_imaging_AI\CNN output';
cd(CNNoutput)

savepath=fullfile(CNNoutput,'Figures');mkdir(savepath);

TCmat=load(fullfile(CNNoutput,'ep_control(1) tle(2) -GM-CNN.mat'));
TCAmat=load(fullfile(CNNoutput,'ep_control(1) adni_control(1) tle(2) alz(3) CNN.mat'));

%% Disease accuracy

%%%% Historgram of accuracy (TLE v Control)
figure('WindowState','maximized');
figtitle='TLE vs Healthy (100 models) - Disease prediction';
sgtitle(figtitle)

subplot(2,2,1)
histogram(cell2mat(TCmat.acc.reg),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Reg Label')

subplot(2,2,3)
hold on
histogram(cellfun(@(x) x.C(1,1)/sum(x.C(1,:),'all'),TCmat.confmat.reg),'BinWidth',0.025);
histogram(cellfun(@(x) x.C(2,2)/sum(x.C(2,:),'all'),TCmat.confmat.reg),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
legend('Control','TLE')

subplot(2,2,2)
histogram(cell2mat(TCmat.acc.shuff),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Shuffled Label')

subplot(2,2,4)
hold on
histogram(cellfun(@(x) x.C(1,1)/sum(x.C(1,:),'all'),TCmat.confmat.shuff),'BinWidth',0.025);
histogram(cellfun(@(x) x.C(2,2)/sum(x.C(2,:),'all'),TCmat.confmat.shuff),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
legend('Control','TLE')
savefig(gcf,fullfile(savepath,figtitle));



%%%% Historgram of accuracy (TLE v Control v Alz)
figure('WindowState','maximized');
figtitle='TLE vs Healthy vs Alz (100 models) - Disease prediction';
sgtitle(figtitle)

subplot(2,2,1)
histogram(cell2mat(TCAmat.acc.reg),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Reg Label')

subplot(2,2,3)
hold on
histogram(cellfun(@(x) x.C(1,1)/sum(x.C(1,:),'all'),TCAmat.confmat.reg),'BinWidth',0.025);
histogram(cellfun(@(x) x.C(2,2)/sum(x.C(2,:),'all'),TCAmat.confmat.reg),'BinWidth',0.025);
histogram(cellfun(@(x) x.C(3,3)/sum(x.C(3,:),'all'),TCAmat.confmat.reg),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
legend('Control','TLE','Alz')

subplot(2,2,2)
histogram(cell2mat(TCAmat.acc.shuff),'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Shuffled Label')

subplot(2,2,4)
hold on
histogram(cellfun(@(x) x.C(1,1)/sum(x.C(1,:),'all'),TCAmat.confmat.shuff),'BinWidth',0.025);
histogram(cellfun(@(x) x.C(2,2)/sum(x.C(2,:),'all'),TCAmat.confmat.shuff),'BinWidth',0.025);
histogram(cellfun(@(x) x.C(3,3)/sum(x.C(3,:),'all'),TCAmat.confmat.shuff),'BinWidth',0.025);

xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
legend('Control','TLE','Alz')
savefig(gcf,fullfile(savepath,figtitle));


%% Confounding factor

%%%% Historgram of accuracy (TLE v Control)
figure('WindowState','maximized');
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
figtitle='TLE vs Healthy vs Alz (100 models) - Age prediction';
sgtitle(figtitle)

subplot(1,2,1)
[maxAcc,maxIdx]=cellfun(@(x) max(x),TCAmat.acc_CF.reg);
histogram(maxAcc,'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Reg Label')

subplot(1,2,2)
[maxAcc,maxIdx]=cellfun(@(x) max(x),TCAmat.acc_CF.shuff);
histogram(maxAcc,'BinWidth',0.025);
xlim([0 1.01])
ylim([0 100])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Shuffled Label')

savefig(gcf,fullfile(savepath,figtitle));






%%
%%%%% Feature visualization

analyzeNetwork(net.reg{1})

controlimg_smooth=load_nii(controlbrain_smooth);
controlimg=load_nii(controlbrain_gm);
imgSize = size(controlimg_smooth.img);
imgSize = imgSize(1:2);

l=12 % ReLU

vw = VideoWriter(fullfile('C:\Users\allen\Documents\GitHub\Bonilha','-Reconstruction.mp4'),'MPEG-4');
open(vw);
hFig = figure('Toolbar', 'none', 'Menu', 'none', 'WindowState', 'maximized'); 
for s=28:85
    sgtitle(['Slice # ',num2str(s)])
    subplot(6,10,5)
    imagesc(controlimg_smooth.img(:,:,s))
    title('Input image')
    [~,I]=sort(cell2mat(acc.reg),'descend');
    count=1;
    for p=1:numel(net.reg)
        tempnet=net.reg{p};
        subplot(6,10,p+10)
        act = activations(tempnet,controlimg_smooth.img(:,:,s),2);
        imagesc(sum(act,3))
        title(num2str(acc.reg{p}))
        count=count+1;
    end
    drawnow
    F = getframe(hFig);
    writeVideo(vw,F);
    writeVideo(vw,F);
    writeVideo(vw,F);
%     colormap jet
%     cbar=colorbar;
%     caxis([0 5000])
end
close(vw)
count=0;
for a=[1 4 8 12]
    figure
    title(['Layer ',num2str(a)])
    act = activations(tempnet,controlimg_smooth.img(:,:,s),a);
    for i=1:size(act,3)
        nexttile
        imagesc(act(:,:,i));
    end
    count=count+32;
end

%     if s==1
%         
%         con_h=nexttile;
%         imshow(controlimg.img(:,:,s),'InitialMagnification','fit','Parent',con_h)
%         title(con_h,'Original image')
%         for a=1:numel(act)
%             img = imtile(mat2gray(act{a}),'GridSize',[6 6]);
%             h(a)=nexttile;
%             imshow(img,'InitialMagnification','fit','Parent',h(a));
%             title(h(a),['Net ',num2str(a),' - Accuracy ',num2str(accuracy_val(a))])
%         end
%     else
%         imshow(controlimg.img(:,:,s),'InitialMagnification','fit','Parent',con_h)
%         title(con_h,'Original image')
%         for a=1:numel(act)
%             img = imtile(mat2gray(act{a}),'GridSize',[6 6]);
%             imshow(img,'InitialMagnification','fit','Parent',h(a));
%             title(h(a),['Net ',num2str(a),' - Accuracy ',num2str(accuracy_val(a))])
%         end
%     end
% end
