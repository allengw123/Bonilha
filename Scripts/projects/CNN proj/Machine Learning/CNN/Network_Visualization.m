%% Analyze network

%%%% Historgram of accuracy
figure('WindowState','maximized');
figtitle=['CNN - middle 50 percent slices - Axial',' ',matter{m}];
sgtitle(figtitle)

subplot(2,4,1)
histogram(cell2mat(acc.reg),'BinWidth',0.05);
xlim([0 1.2])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Reg Label')

subplot(2,4,5)
hold on
histogram(cellfun(@(x) x(1,1)/sum(x(1,:),'all'),confmat.reg),'BinWidth',0.05);
histogram(cellfun(@(x) x(2,2)/sum(x(2,:),'all'),confmat.reg),'BinWidth',0.05);
histogram(cellfun(@(x) x(3,3)/sum(x(3,:),'all'),confmat.reg),'BinWidth',0.05);
legend('control','alz','tle')
xlim([0 1.2])
xlabel('Accuracy')
ylabel('# of models')

subplot(2,4,2)
histogram(cell2mat(acc.shuff),'BinWidth',0.05);
xlim([0 1.2])
xlabel('Accuracy')
ylabel('# of models')
title('CNN Shuffle Label')

subplot(2,4,6)
hold on
histogram(cellfun(@(x) x(1,1)/sum(x(1,:),'all'),confmat.shuff),'BinWidth',0.05);
histogram(cellfun(@(x) x(2,2)/sum(x(2,:),'all'),confmat.shuff),'BinWidth',0.05);
histogram(cellfun(@(x) x(3,3)/sum(x(3,:),'all'),confmat.shuff),'BinWidth',0.05);
legend('control','alz','tle')
xlabel('Accuracy')
ylabel('# of models')
xlim([0 1.2])


subplot(2,4,3)
histogram(cellfun(@(x) mean(x),acc_CF.reg),'BinWidth',0.05);
xlim([0 1.2])
xlabel('Accuracy')
ylabel('# of models')
title('CF Reg Label')

subplot(2,4,7)
hold on
histogram(cellfun(@(x) mean(cellfun(@(y) y(1,1)/sum(y(1,:),'all'),x),'all'),confmat_CF.reg),'BinWidth',0.05);
histogram(cellfun(@(x) mean(cellfun(@(y) y(2,2)/sum(y(2,:),'all'),x),'all'),confmat_CF.reg),'BinWidth',0.05);
histogram(cellfun(@(x) mean(cellfun(@(y) y(3,3)/sum(y(3,:),'all'),x),'all'),confmat_CF.reg),'BinWidth',0.05);
legend('control','alz','tle')
xlabel('Accuracy')
ylabel('# of models')
xlim([0 1.2])

subplot(2,4,4)
histogram(cellfun(@(x) mean(x),acc_CF.shuff),'BinWidth',0.05);
xlim([0 1.2])
xlabel('Accuracy')
ylabel('# of models')
title('CF Shuffle Label')

subplot(2,4,8)
hold on
histogram(cellfun(@(x) mean(cellfun(@(y) y(1,1)/sum(y(1,:),'all'),x),'all'),confmat_CF.shuff),'BinWidth',0.05);
histogram(cellfun(@(x) mean(cellfun(@(y) y(2,2)/sum(y(2,:),'all'),x),'all'),confmat_CF.shuff),'BinWidth',0.05);
histogram(cellfun(@(x) mean(cellfun(@(y) y(3,3)/sum(y(3,:),'all'),x),'all'),confmat_CF.shuff),'BinWidth',0.05);
legend('control','alz','tle')
xlabel('Accuracy')
ylabel('# of models')
xlim([0 1.2])

saveas(gcf,fullfile(save_path,figtitle));
close all
clc

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
