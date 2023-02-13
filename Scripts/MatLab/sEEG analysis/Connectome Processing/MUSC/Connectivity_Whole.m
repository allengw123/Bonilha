%% Add correct paths
clear all
close all
clc

gitpath='/home/bonilha/Documents/GitHub/Bonilha';
cd(gitpath)
allengit_genpath(gitpath,'EEG')

%% Subject/Trial info
% Define info
trialnames={'Baseline','Pre-transition','Post-transition','Mid Sezuire','Late Sezuire','Early Post','Late Post'};

datadir='/media/bonilha/AllenProj/sEEG_project/PatientData/Original/';

subjID = {dir(fullfile(datadir,'Patient*')).name}';

master_electrode={'LA','LAH','LAI','LLF','LMF','LPH','LPI','RA','RAH','RAI','RLF','RMF','RPH','RPI'};

master_electrode_labels=[];
for i=1:numel(master_electrode)
    for z=1:9
        master_electrode_labels=[master_electrode_labels;{[master_electrode{i},num2str(z)]}];
    end
end

master_electrode_labels_grouped=[];
group_label={'(D)','(M)','(S)'};
for i=1:numel(master_electrode)
    for z=1:3
        if z==1
            master_electrode_labels_grouped=[master_electrode_labels_grouped;{[master_electrode{i},'-',group_label{z},'-']}];
        else
            master_electrode_labels_grouped=[master_electrode_labels_grouped;{[group_label{z},'-']}];
        end
    end
end

foi={'beta';[15 25]};

%% Connectivity
for subj=1:numel(subjID)

    % Dedicate working subject paths
    wk_subject_dir = fullfile(datadir,subjID{subj});
    wk_ieeg_dir = fullfile(wk_subject_dir,'sEEG');
    save_folder = fullfile(wk_ieeg_dir,'matdata_whole');
    mkdir(save_folder)

    % Find recording files
    sez_files = dir(fullfile(wk_ieeg_dir,'*.edf'));

    % Find baseline file
    baselineEDF = sez_files(contains({sez_files.name},'Baseline'),:);
    sez_files(contains({sez_files.name},'Baseline'),:) = [];

    for freq=1:size(foi,2)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%% Baseline %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        % Filters
        cfg = [];
        cfg.continuous  = 'yes';
        cfg.dataset     = fullfile(baselineEDF.folder,baselineEDF.name);
        cfg.hpfiltord   = 5;
        cfg.hpfilter    = 'yes';
        cfg.hpfreq      = 1;
        cfg.bsfilter    = 'yes';
        cfg.bsfreq      = [58 62;118 122;178 182];

        % Re-Reference
        cfg.refmethod = 'avg';

        % Apply Pre-proc protocol
        data_eeg        = ft_preprocessing(cfg);

        % Downsample to 256
        cfg = [];
        cfg.resamplefs = 256;
        data_eeg = ft_resampledata(cfg, data_eeg);

        % iEEG sensory level coherence
        % Calculate coherence
        cfg                 = [];
        cfg.output          = 'powandcsd';
        cfg.method          = 'mtmfft';
        cfg.taper           = 'dpss';
        cfg.pad             = 'nextpow2';
        cfg.keeptrials      = 'yes';
        cfg.tapsmofrq       = 2;
        cfg.foilim          = foi{2,freq};
        freq_csd            = ft_freqanalysis(cfg, data_eeg);


        clearvars con_mean


        cfg                 = [];
        cfg.method          = 'coh';
        conn                = ft_connectivityanalysis(cfg, freq_csd);

        % Organize connectivity matrix into template
        connectivitymat=nan(numel(master_electrode_labels),numel(master_electrode_labels));
        for i=1:length(conn.labelcmb)
            tempcony=conn.labelcmb{i,1};
            row_idx=find(strcmp(tempcony,master_electrode_labels));

            tempconx=conn.labelcmb{i,2};
            column_idx=find(strcmp(tempconx,master_electrode_labels));
            
            if ~isempty(row_idx) && ~isempty(column_idx)
                connectivitymat(row_idx,column_idx)=mean(conn.cohspctrm(i,:));
            end
        end

        % Mirror amoung diag
        A=connectivitymat;
        B=connectivitymat';
        C=sum(cat(3,A,B),3,'omitnan');
        C(isnan(A)&isnan(B)) = NaN;
        connectivitymat=C;

        % Group Deep, Middle, Shallow electrodes
        rowcount=0;
        for row=1:3:size(connectivitymat,1)
            rowcount=rowcount+1;
            colcount=0;
            for col=1:3:size(connectivitymat,2)
                colcount=colcount+1;

                temp=connectivitymat(row:row+2,col:col+2);
                temp=mean(mean(temp,1),2);
                connectivitymat_grouped(rowcount,colcount)=temp;
            end
        end

        connectivitymat_grouped(logical(eye(size(connectivitymat_grouped,1))))=0;

        save(fullfile(save_folder,[subjID{subj},'_',foi{1,freq},'_baseline']),'connectivitymat_grouped')

        %%%%%%%%%%%%%%%%%%%%%%%%%%%% Sezuire CUSTOM EPOCH  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Define info
        trialnames={'Baseline','Pre-transition','Post-transition','Mid Sezuire','Late Sezuire','Early Post','Late Post'};



%         for sez=1:numel(sez_files)
% 
%             % Seizure name
%             wk_sez_name = extractBetween(sez_files(sez).name,' ','.edf');
%             wk_sez_name = wk_sez_name{:};
%             
%             cfg = [];
%             cfg.dataset     = fullfile(sez_files(sez).folder,sez_files(sez).name);
%             cfg.continuous  = 'yes';
%             data_eeg        = ft_preprocessing(cfg);
% 
%             fs = data_eeg.fsample;
% 
%             cfg = [];
%             cfg.trl(1,1) = 30*fs;
%             cfg.trl(1,2) = data_eeg.sampleinfo(2) - (30*fs);
%             cfg.trl(1,3) = 0;
% 
%             % Add Filter
%             cfg.hpfiltord   = 5;
%             cfg.hpfilter    = 'yes';
%             cfg.hpfreq      = 1;
%             cfg.bsfilter    = 'yes';
%             cfg.bsfreq      = [58 62;118 122;178 182];
%             cfg.dataset     = fullfile(sez_files(sez).folder,sez_files(sez).name);
% 
%             % Re-Reference
%             cfg.refmethod = 'avg';
% 
%             % Apply Pre-proc protocol
%             data_eeg        = ft_preprocessing(cfg);
% 
%             % Downsample to 256
%             cfg = [];
%             cfg.resamplefs = 256;
%             data_eeg = ft_resampledata(cfg, data_eeg);
% 
% 
%             % Calculated connectivity
%             cfg                 = [];
%             cfg.output          = 'powandcsd';
%             cfg.method          = 'mtmfft';
%             cfg.taper           = 'dpss';
%             cfg.pad             = 'nextpow2';
%             cfg.keeptrials      = 'yes';
%             cfg.tapsmofrq       = 2;
%             cfg.foilim          = foi{2,freq};
%             freq_csd            = ft_freqanalysis(cfg, data_eeg);
% 
%             connectivitymat=nan(numel(master_electrode_labels),numel(master_electrode_labels),numel(data_eeg.trial));
%             for q=1:numel(data_eeg.trial)
%                 cfg                 = [];
%                 cfg.method          = 'coh';
%                 cfg.trials          = q;
%                 conn                = ft_connectivityanalysis(cfg, freq_csd);
% 
%                 % Organize connectivity matrix into template
%                 for i=1:length(conn.labelcmb)
%                     tempcony=conn.labelcmb{i,1};
%                     row_idx=find(strcmp(tempcony,master_electrode_labels));
% 
%                     tempconx=conn.labelcmb{i,2};
%                     column_idx=find(strcmp(tempconx,master_electrode_labels));
%                     connectivitymat(row_idx,column_idx,q)=mean(conn.cohspctrm(i,:));
%                 end
% 
%                 % Mirror amoung diag
%                 A=connectivitymat(:,:,q);
%                 B=connectivitymat(:,:,q)';
%                 C=sum(cat(3,A,B),3,'omitnan');
%                 C(isnan(A)&isnan(B)) = NaN;
% 
%                 connectivitymat(:,:,q)=C;
%                 % Group Deep, Middle, Shallow electrodes
%                 rowcount=0;
%                 for row=1:3:size(connectivitymat,1)
%                     rowcount=rowcount+1;
%                     colcount=0;
%                     for col=1:3:size(connectivitymat,2)
%                         colcount=colcount+1;
% 
%                         temp=connectivitymat(row:row+2,col:col+2,q);
%                         temp=mean(mean(temp,1),2);
%                         connectivitymat_grouped(rowcount,colcount,q)=temp;
%                     end
%                 end
%             end
%             save(fullfile(save_folder,[foi{1,freq},'_',extractBefore(sez_files(sez).name,'.edf')]),'connectivitymat_grouped')
%         end
    end
end
