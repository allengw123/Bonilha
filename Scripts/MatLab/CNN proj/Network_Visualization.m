%% Network Visualization
clear
clc

githubpath='/home/bonilha/Documents/GitHub/Bonilha';

cd(githubpath)
allengit_genpath(githubpath,'imaging')

% Inputs:
CNNoutput='/media/bonilha/AllenProj/CNN_project/CNN output/2D_CNN/MATLAB/disease_pred/AgeRegress';

cd(CNNoutput)

savepath=fullfile(CNNoutput,'Figures');mkdir(savepath);

TCAmat=load(fullfile(CNNoutput,'AgeRegress_GM_ADTLEHC_CNN.mat'));

%% Disease accuracy

%%%% Historgram of accuracy (TLE v Control v Alz)

conf_stat_reg=conf_analysis(TCAmat.confmat.reg);
conf_stat_shuff=conf_analysis(TCAmat.confmat.shuff);
%%

figure('WindowState','maximized');
set(gcf,'color','w');
hold on
h1 = histfit(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100,10);
m1 = mean(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)
s1 = std(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)
h2 = histfit(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100,5);
m2 = mean(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100)
s2 = std(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100)
xlim([40 101]) 
ylim([0 90])
yticks(0:20:80)
xlabel('Accuracy')
ylabel('# of models')
xticks([40:20:100])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
legend([h1(1) h2(1)],{'Reg','Shuff'})

fdic=sum((mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)>(mean(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100)))/100



figure('WindowState','maximized');
set(gcf,'color','w');
hold on
h1 = histfit(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'),10);
mean(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'))
std(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'))
h2 = histfit(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'),15);
mean(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'))
std(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'))
xlim([.10 1.01])
ylim([0 90])
xlabel('Precision')
ylabel('# of models')
xticks([0.10:0.20:1.00])
yticks(0:20:80)
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
legend([h1(1) h2(1)],{'Reg','Shuff'})


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
h1 = histfit(mean(cell2mat(conf_stat_reg.Recall'),2,'omitnan'),15);
mean(mean(cell2mat(conf_stat_reg.Recall'),2,'omitnan'))
std(mean(cell2mat(conf_stat_reg.Recall'),2,'omitnan'))
h2 = histfit(mean(cell2mat(conf_stat_shuff.Recall'),2,'omitnan'),5);
mean(mean(cell2mat(conf_stat_shuff.Recall'),2,'omitnan'))
std(mean(cell2mat(conf_stat_shuff.Recall'),2,'omitnan'))
xlim([.10 1.01])
ylim([0 90])
xlabel('Recall')
ylabel('# of models')
xticks([0.10:0.20:1.00])
yticks(0:20:80)
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
legend([h1(1) h2(1)],{'Reg','Shuff'})


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
h1 = histfit(mean(cell2mat(conf_stat_reg.F1'),2,'omitnan'),15);
mean(mean(cell2mat(conf_stat_reg.F1'),2,'omitnan'))
std(mean(cell2mat(conf_stat_reg.F1'),2,'omitnan'))
h2 = histfit(mean(cell2mat(conf_stat_shuff.F1'),2,'omitnan'),40);
mean(mean(cell2mat(conf_stat_shuff.F1'),2,'omitnan'))
std(mean(cell2mat(conf_stat_shuff.F1'),2,'omitnan'))
xlim([.10 1.01])
ylim([0 90])
xlabel('F1')
ylabel('# of models')
xticks([0.10:0.20:1.00])
yticks(0:20:80)
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
legend([h1(1) h2(1)],{'Reg','Shuff'})


%%
figure('WindowState','maximized');
set(gcf,'color','w');
hold on
h1 = histfit(conf_stat_reg.Acc{1}*100);
mean(conf_stat_reg.Acc{1}*100)
std(conf_stat_reg.Acc{1}*100)
h2 = histfit(conf_stat_reg.Acc{2}*100);
mean(conf_stat_reg.Acc{2}*100)
std(conf_stat_reg.Acc{2}*100)
h3 = histfit(conf_stat_reg.Acc{3}*100);
mean(conf_stat_reg.Acc{3}*100)
std(conf_stat_reg.Acc{3}*100)
xlim([40 101]) 
ylim([0 70])
yticks([0:20:60])
xlabel('Accuracy')
ylabel('# of models')
xticks([40:20:100])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
legend([h1(1) h2(1) h3(1)],{'Control','TLE','Alz'})


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
h1 = histfit(conf_stat_reg.Precision{1});
mean(conf_stat_reg.Precision{1})
std(conf_stat_reg.Precision{1})
h2 = histfit(conf_stat_reg.Precision{2});
mean(conf_stat_reg.Precision{2},'omitnan')
std(conf_stat_reg.Precision{2},'omitnan')
h3 = histfit(conf_stat_reg.Precision{3});
mean(conf_stat_reg.Precision{3},'omitnan')
std(conf_stat_reg.Precision{3},'omitnan')
xlim([.10 1.01])
ylim([0 70])
xlabel('Precision')
ylabel('# of models')
xticks([0.10:0.20:1.00])
yticks(0:20:80)
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
% legend([h1(1) h2(1) h3(1)],{'Control','TLE','Alz'})

figure('WindowState','maximized');
set(gcf,'color','w');
hold on
h1 = histfit(conf_stat_reg.Recall{1});
mean(conf_stat_reg.Recall{1},'omitnan')
std(conf_stat_reg.Recall{1},'omitnan')
h2 = histfit(conf_stat_reg.Recall{2});
mean(conf_stat_reg.Recall{2},'omitnan')
std(conf_stat_reg.Recall{2},'omitnan')
h3 = histfit(conf_stat_reg.Recall{3});
mean(conf_stat_reg.Recall{3},'omitnan')
std(conf_stat_reg.Recall{3},'omitnan')
xlim([.10 1.01])
ylim([0 70])
xlabel('Recall')
ylabel('# of models')
xticks([0.10:0.20:1.00])
yticks([0:20:60])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
legend([h1(1) h2(1) h3(1)],{'Control','TLE','Alz'})

figure('WindowState','maximized');
set(gcf,'color','w');
hold on
h1 = histfit(conf_stat_reg.F1{1});
mean(conf_stat_reg.F1{1},'omitnan')
std(conf_stat_reg.F1{1},'omitnan')
h2 = histfit(conf_stat_reg.F1{2});
mean(conf_stat_reg.F1{2},'omitnan')
std(conf_stat_reg.F1{2},'omitnan')
h3 = histfit(conf_stat_reg.F1{3});
mean(conf_stat_reg.F1{3},'omitnan')
std(conf_stat_reg.F1{3},'omitnan')
xlim([.10 1.01])
ylim([0 70])
xlabel('F1')
ylabel('# of models')
xticks([0.10:0.20:1.00])
yticks([0:20:60])
axis square
pbaspect([2 1 1])
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',30)
legend([h1(1) h2(1) h3(1)],{'Control','TLE','Alz'})

%% Confounding factor

[maxAcc,maxIdx_reg]=cellfun(@(x) max(x),TCAmat.acc_CF.reg);
tempconf=TCAmat.confmat_CF.reg;
for i=1:numel(TCAmat.confmat_CF.reg)
    tempconf{i}.C=tempconf{i}.C{maxIdx_reg(i)};
    tempconf{i}.order=tempconf{i}.order{maxIdx_reg(i)};
    tempconf{i}.perm=tempconf{i}.perm{maxIdx_reg(i)};
end
conf_stat_reg=conf_analysis(tempconf);

figure('WindowState','maximized');
set(gcf,'color','w');
histogram(maxIdx_reg)
xlim([0,7])
ylim([0,100])
axis square
ylabel('# of Models')


[maxAcc,maxIdx_shuff]=cellfun(@(x) max(x),TCAmat.acc_CF.shuff);
tempconf=TCAmat.confmat_CF.shuff;
for i=1:numel(TCAmat.confmat_CF.reg)
    tempconf{i}.C=tempconf{i}.C{maxIdx_shuff(i)};
    tempconf{i}.order=tempconf{i}.order{maxIdx_shuff(i)};
    tempconf{i}.perm=tempconf{i}.perm{maxIdx_shuff(i)};
end
conf_stat_shuff=conf_analysis(tempconf);


figure('WindowState','maximized');
set(gcf,'color','w');
histogram(maxIdx_shuff)
xlim([0,7])
ylim([0,100])
axis square
ylabel('# of Models')


%%
%%%% Historgram of accuracy (TLE v Control v Alz)

figure('WindowState','maximized');
set(gcf,'color','w');
hold on
proper = histfit(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100);
mean(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)
std(mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)
shuff = histfit(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100,7);
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
legend([proper(1),shuff(1)],{'Proper','Shuffled'})

fdic=sum((mean(cell2mat(conf_stat_reg.Acc'),2,'omitnan')*100)>(mean(mean(cell2mat(conf_stat_shuff.Acc'),2,'omitnan')*100)))/100


figure('WindowState','maximized');
set(gcf,'color','w');
hold on
histfit(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'),10);
mean(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'))
std(mean(cell2mat(conf_stat_reg.Precision'),2,'omitnan'))
histfit(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'),15);
mean(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'))
std(mean(cell2mat(conf_stat_shuff.Precision'),2,'omitnan'))
xlim([0 1.01])
ylim([0 90])
xlabel('Precision')
ylabel('# of models')
xticks(0:0.2:1)
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
xlim([0 1.01])
ylim([0 90])
xlabel('Recall')
ylabel('# of models')
xticks(0:.2:1)
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
mean(conf_stat_reg.Precision{1})
std(conf_stat_reg.Precision{1})
histfit(conf_stat_reg.Precision{2});
mean(conf_stat_reg.Precision{2})
std(conf_stat_reg.Precision{2})
histfit(conf_stat_reg.Precision{3});
mean(conf_stat_reg.Precision{3})
std(conf_stat_reg.Precision{3})
xlim([0 1.01])  
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
mean(conf_stat_reg.Recall{1})
std(conf_stat_reg.Recall{1})
histfit(conf_stat_reg.Recall{2});
mean(conf_stat_reg.Recall{2})
std(conf_stat_reg.Recall{2})
histfit(conf_stat_reg.Recall{3});
mean(conf_stat_reg.Recall{3})
std(conf_stat_reg.Recall{3})
xlim([0 1.01])  
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
        Acc{tempperm(g),1}(i,1)=(TP{tempperm(g),1}(i,1)+TN{tempperm(g),1}(i,1))/sum(sum(tempconf));
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
