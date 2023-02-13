%% Add correct paths
clear all
close all
clc

gitpath='C:\Users\allen\Documents\GitHub\Bonilha';
cd(gitpath)
allengit_genpath(gitpath,'imaging')

%% Subject/Trial info

datadir='C:\Users\allen\Box Sync\Desktop\Allen_Bonilha_EEG\Projects\sEEG project\PatientData';
analysisdir=fullfile(datadir,'Analysis','Functional');
mkdir(analysisdir);
figdir=fullfile(analysisdir,'figures');
mkdir(figdir);

subjID = {dir(fullfile(datadir,'Patient *')).name};
subjnum = regexp(subjID,'\d*','Match');

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

trial_length=5;
foi={'alpha_theta', 'beta', 'low_gamma', 'high_gamma';[5 15],[15 25],[30 40],[95 105]};

%% Connectivity
for subj=1:numel(subjID)
        
    % Electrode Reference sheet
    electrode_reference=readtable(fullfile(datadir,'Electrode_Reference.xlsx'),'ReadVariableNames',0,'Sheet',subjID{subj});
    
    % Compare to master electrode list
    temp=electrode_reference{:,1};
    Electrodes=temp(endsWith(cellfun(@(x) [x{:}],regexp(temp,'\D','match'),'UniformOutput',false),master_electrode));
   
    % Remove 10th Electrode
    Electrodes(endsWith(Electrodes,'10'))=[];
    
    % Subject Directory    
    subjdir=fullfile(datadir,subjID{subj},'sEEG');
    subjdir_abv=['P',subjnum{subj}{1}];
    
    % Save Electrodes
    save(fullfile(subjdir,'Electrodes'),'Electrodes');
    
    % Baseline EDF
    baselineEDF=fullfile(subjdir,[subjdir_abv,' Baseline.edf']);
    
    % Find seizure files
    seizureEDF={dir(fullfile(subjdir,[subjdir_abv,'*'])).name};
    seizureEDF=seizureEDF(~strcmp([subjdir_abv,' Baseline.edf'],seizureEDF));
    
    % Make data storage folder
    matdatafolder=fullfile(subjdir,'matdata');
    mkdir(matdatafolder);
      
        
     for freq=1:size(foi,2)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%% Baseline %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Load EDF and divide EEG to trials
        cfg = [];
        cfg.trialdef.triallength = trial_length;
        cfg.dataset     = baselineEDF;
        cfg.continuous  = 'yes';
        cfg.channel     = Electrodes;
        cfg = ft_definetrial(cfg);


        % Filters
        cfg.hpfiltord   = 5;
        cfg.hpfilter    = 'yes';
        cfg.hpfreq      = 1;
        cfg.bsfilter    = 'yes';
        cfg.bsfreq      = [58 62;118 122;178 182];
        
        % Re-Reference
        cfg.refmethod = 'avg';
        
        % Apply Pre-proc protocol
        data_eeg        = ft_preprocessing(cfg);
        
        
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
            connectivitymat(row_idx,column_idx)=mean(conn.cohspctrm(i,:));
        end
        
        
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
        
        save(fullfile(matdatafolder,[subjID{subj},'_',foi{1,freq},'_baseline']),'connectivitymat_grouped')

        %%%%%%%%%%%%%%%%%%%%%%%%%%%% Sezuire CUSTOM EPOCH  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        

        % Define info
        trialnames={'Baseline','Pre-transition','Post-transition','Mid Sezuire','Late Sezuire','Early Post','Late Post'};
        
              
        
        for sez=1:numel(seizureEDF)
            
            % Located datafile and add info
            cfg = [];
            cfg.dataset     = fullfile(subjdir,seizureEDF{sez});
            cfg.continuous  = 'yes';
            cfg.channel     = Electrodes;
            data_eeg        = ft_preprocessing(cfg);

            % Define trials
            trialtimes.baseline(1,1)=1;
            trialtimes.baseline(1,2)=data_eeg.hdr.Fs*30;

            trialtimes.post(1,1)=data_eeg.hdr.nSamples-data_eeg.hdr.Fs*30;
            trialtimes.post(1,2)=data_eeg.hdr.nSamples;

            trialtimes.seizure(1,1)=trialtimes.baseline(1,2);
            trialtimes.seizure(1,2)= trialtimes.post(1,1);


            cfg = [];
            cfg.trl(1,1)=1;                                                     %Baseline
            cfg.trl(2,1)=trialtimes.baseline(1,2)-trial_length*data_eeg.hdr.Fs; %Pre-transition
            cfg.trl(3,1)=trialtimes.baseline(1,2);                              %Post-transition
            cfg.trl(4,1)=mean(trialtimes.seizure);                              %Mid Sezuire
            cfg.trl(5,1)=trialtimes.post(1,1)-trial_length*data_eeg.hdr.Fs;     %Late Sezuire
            cfg.trl(6,1)=trialtimes.post(1,1);                                  %Early Post
            cfg.trl(7,1)=trialtimes.post(1,2)-trial_length*data_eeg.hdr.Fs;     %Late Post

            cfg.trl(:,2)=cfg.trl(:,1)+trial_length*data_eeg.hdr.Fs;
            cfg.trl(:,3)=0;
            
            
            % Add Filter
            cfg.hpfiltord   = 5;
            cfg.hpfilter    = 'yes';
            cfg.hpfreq      = 1;
            cfg.bsfilter    = 'yes';
            cfg.bsfreq      = [58 62;118 122;178 182];
            cfg.channel     = Electrodes;
            cfg.dataset     = fullfile(subjdir,seizureEDF{sez});
            
            % Re-Reference
            cfg.refmethod = 'avg';

            % Apply Pre-proc protocol
            data_eeg        = ft_preprocessing(cfg);


            % Calculated connectivity
            cfg                 = [];
            cfg.output          = 'powandcsd';
            cfg.method          = 'mtmfft';
            cfg.taper           = 'dpss';
            cfg.pad             = 'nextpow2';
            cfg.keeptrials      = 'yes';
            cfg.tapsmofrq       = 2;
            cfg.foilim          = foi{2,freq};
            cfg.channel         = Electrodes;
            freq_csd            = ft_freqanalysis(cfg, data_eeg);
            
            connectivitymat=nan(numel(master_electrode_labels),numel(master_electrode_labels),numel(data_eeg.trial));
            for q=1:numel(data_eeg.trial)
                cfg                 = [];
                cfg.method          = 'coh';
                cfg.trials          = q;
                conn                = ft_connectivityanalysis(cfg, freq_csd);
                        
                % Organize connectivity matrix into template
                for i=1:length(conn.labelcmb)
                    tempcony=conn.labelcmb{i,1};
                    row_idx=find(strcmp(tempcony,master_electrode_labels));            

                    tempconx=conn.labelcmb{i,2};
                    column_idx=find(strcmp(tempconx,master_electrode_labels));
                    connectivitymat(row_idx,column_idx,q)=mean(conn.cohspctrm(i,:));
                end
                
                % Mirror amoung diag
                A=connectivitymat(:,:,q);
                B=connectivitymat(:,:,q)';
                C=sum(cat(3,A,B),3,'omitnan');
                C(isnan(A)&isnan(B)) = NaN;
                
                connectivitymat(:,:,q)=C;
                % Group Deep, Middle, Shallow electrodes
                rowcount=0;
                for row=1:3:size(connectivitymat,1)
                    rowcount=rowcount+1;
                    colcount=0;
                    for col=1:3:size(connectivitymat,2)
                        colcount=colcount+1;

                        temp=connectivitymat(row:row+2,col:col+2,q);
                        temp=mean(mean(temp,1),2);
                        connectivitymat_grouped(rowcount,colcount,q)=temp;
                    end
                end
            end
            
%             % Add zeroes to the diag
%             for i=1:size(connectivitymat_grouped,3)
%                 tempmat=connectivitymat_grouped(:,:,i);
%                 tempmat(logical(eye(size(tempmat,1))))=0;
%                 connectivitymat_grouped(:,:,i)=tempmat;
%             end
            save(fullfile(matdatafolder,[subjID{subj},'_',foi{1,freq},'_',extractBefore(seizureEDF{sez},'.edf')]),'connectivitymat_grouped')
        end
    end
end
