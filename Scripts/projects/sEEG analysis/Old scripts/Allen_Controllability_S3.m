%% Set Paths
analysisdir='C:\Box Sync\Allen_Bonhila_EEG\sEEGdata\Analysis';

trials_label={'baseline','pre-baseline','pre-trans','post-trans','mid-seiz','late-seiz','early-post','late-post'};
for i=1:length(trials_label)-1
    diff_trials_label{i}=[trials_label{i},' VS ',trials_label{i+1}];
end

Patient_ID={dir(fullfile(analysisdir,'Patient*')).name};

datadir='C:\Box Sync\Allen_Bonhila_EEG\sEEGdata';

bandname = {'alpha','theta'};    
%% calculate the Gramian
for m=1:numel(Patient_ID)
    for b=1:numel(bandname)

        % Electrode detection
        electrode_reference=readtable(fullfile(datadir,'Electrode_Reference.xlsx'),'ReadVariableNames',0,'Sheet',Patient_ID{m});

        % Electrode labels
        elec_temp=string(electrode_reference{:,:});
        elec_temp=regexp(elec_temp,'\D','match');
        elec_temp=cellfun(@(x) [x{:}],elec_temp,'UniformOutput',false);
        elec_temp=elec_temp(~cellfun(@isempty,elec_temp));
        electrode_labels=unique(elec_temp);

        ROI_lbl=[];
        for i=1:numel(electrode_labels)
            ROI_temp={[electrode_labels{i},'(D)'],[electrode_labels{i},'(M)'],[electrode_labels{i},'(S)']};
            ROI_lbl=[ROI_lbl ROI_temp];
        end

        A=load(fullfile(analysisdir,Patient_ID{m},[Patient_ID{m},'-',bandname{b},'_coh_comb_variables.mat']));
        A=A.coh_comb_all;


        ave_con=[];
        modal_con=[];
        closeness=[];
        betweenness=[];
        for i=1:size(A,3)
            W=A(:,:,i);
            for ll=1:length(W)
                nodes=zeros(length(W),1);
                nodes(ll,1)=1;
                B=nodes;
                [H2, smeig ] = Gramian(W, nodes );
                al_H2(ll,:)=H2;
                all_H2=transpose(al_H2);
                al_smeig(ll,:)=smeig;
                all_smeig=transpose(al_smeig);
                min_smeig(i)=min(all_smeig);
            end
            temp_ave_control=ave_control(W);
            ave_con=[ave_con temp_ave_control];

            temp_modal_con=modal_control(W);
            modal_con=[modal_con temp_modal_con];

            g=graph(W);

            closeness_temp=centrality(g,'closeness','cost',(1-g.Edges.Weight));
            closeness=[closeness closeness_temp];

            betweenness_temp=centrality(g,'betweenness','cost',(1-g.Edges.Weight));
            betweenness=[betweenness betweenness_temp];

        end

        % Controllability Measure
        figure('Name',[Patient_ID{m},'-','Controllability Measure(',bandname{b},')'])
        sgtitle(get(gcf,'Name'))
        bar(log10(min_smeig))
        set(gca,'XTickLabel',trials_label,'XTickLabelRotation',45)
        ylabel('Log Transformed (min smeig)')
        set(gcf, 'Position', get(0, 'Screensize'))
        title('Network Controllability')
        saveas(gcf,fullfile(analysisdir,Patient_ID{m},get(gcf,'Name')));


        % Average Controllability
        figure('Name',[Patient_ID{m},'-','Average Controllability(',bandname{b},')'])
        sgtitle(get(gcf,'Name'))
        subplot(2,1,1)
        imagesc(ave_con)
        set(gca,'XTickLabel',trials_label,'XTickLabelRotation',45,'YTickLabel',ROI_lbl,'YTick',1:length(ROI_lbl))
        set(gcf, 'Position', get(0, 'Screensize'))
        c=colorbar;
        c.FontSize = 12;
        colormap jet;
        title(['Average Controllability',])

        subplot(2,1,2)
        imagesc(diff(ave_con,1,2))
        set(gca,'XTickLabel',diff_trials_label,'XTickLabelRotation',45,'YTickLabel',ROI_lbl,'YTick',1:length(ROI_lbl))
        c=colorbar;
        c.FontSize = 12;
        colormap jet;
        title(['Average Controllability Difference'])
        saveas(gcf,fullfile(analysisdir,Patient_ID{m},get(gcf,'Name')));


        % Modal Controllability
        figure('Name',[Patient_ID{m},'-','Modal Controllability(',bandname{b},')'])
        sgtitle(get(gcf,'Name'))
        subplot(2,1,1)
        imagesc(modal_con)
        set(gca,'XTickLabel',trials_label,'XTickLabelRotation',45,'YTickLabel',ROI_lbl,'YTick',1:length(ROI_lbl))
        set(gcf, 'Position', get(0, 'Screensize'))
        c=colorbar;
        c.FontSize = 12;
        colormap jet;
        title(['Modal Controllability'])

        subplot(2,1,2)
        imagesc(diff(modal_con,1,2))
        set(gca,'XTickLabel',diff_trials_label,'XTickLabelRotation',45,'YTickLabel',ROI_lbl,'YTick',1:length(ROI_lbl))
        c=colorbar;
        c.FontSize = 12;
        colormap jet;
        title(['Modal Controllability Difference'])
        saveas(gcf,fullfile(analysisdir,Patient_ID{m},get(gcf,'Name')));

        % Centrallity (Closeness)
        figure('Name',[Patient_ID{m},'-','Centrallity (Closeness- ',bandname{b},')'])
        sgtitle(get(gcf,'Name'))
        subplot(2,1,1)
        imagesc(closeness)
        set(gca,'XTickLabel',trials_label,'XTickLabelRotation',45,'YTickLabel',ROI_lbl,'YTick',1:length(ROI_lbl))
        set(gcf, 'Position', get(0, 'Screensize'))
        c=colorbar;
        c.FontSize = 12;
        colormap jet;
        title(['Centrallity (Closeness)'])

        subplot(2,1,2)
        imagesc(diff(closeness,1,2))
        set(gca,'XTickLabel',diff_trials_label,'XTickLabelRotation',45,'YTickLabel',ROI_lbl,'YTick',1:length(ROI_lbl))
        c=colorbar;
        c.FontSize = 12;
        colormap jet;
        title(['Centrallity (Closeness) Difference'])
        saveas(gcf,fullfile(analysisdir,Patient_ID{m},get(gcf,'Name')));

        % Centrallity (Betweenness)
        figure('Name',[Patient_ID{m},'-','Centrallity (Betweenness- ',bandname{b},')'])
        sgtitle(get(gcf,'Name'))
        subplot(2,1,1)
        imagesc(betweenness)
        set(gca,'XTickLabel',trials_label,'XTickLabelRotation',45,'YTickLabel',ROI_lbl,'YTick',1:length(ROI_lbl))
        set(gcf, 'Position', get(0, 'Screensize'))
        c=colorbar;
        c.FontSize = 12;
        colormap jet;
        title(['Centrallity (Betweenness)'])

        subplot(2,1,2)
        imagesc(diff(betweenness,1,2))
        set(gca,'XTickLabel',diff_trials_label,'XTickLabelRotation',45,'YTickLabel',ROI_lbl,'YTick',1:length(ROI_lbl))
        c=colorbar;
        c.FontSize = 12;
        colormap jet;
        title(['Centrallity (Betweenness) Difference'])
        saveas(gcf,fullfile(analysisdir,Patient_ID{m},get(gcf,'Name')));
    end
    close all
end
