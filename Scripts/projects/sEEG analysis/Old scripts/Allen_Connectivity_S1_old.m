%% Add correct paths
clear all
close all
clc

addpath 'C:\Box Sync\Functions\EEG_toolboxes\Matlab\fieldtrip-20200607'
ft_defaults
addpath 'C:\Box Sync\Functions\EEG_toolboxes\Matlab\fieldtrip-20200607\external\spm12'
addpath 'C:\Box Sync\Functions\EEG_toolboxes\Matlab\fieldtrip-20200607\external\bsmart'

%% Subject/Trial info

datadir='C:\Box Sync\Allen_Bonhila_EEG\sEEGdata';

subjID = {dir(fullfile(datadir,'Patient*')).name};

subjnum = regexp(subjID,'\d*','Match');

analysisdir='C:\Box Sync\Allen_Bonhila_EEG\sEEGdata\Analysis';


trial_length=5;
foi={'alpha','theta';[8 12],[4 7]};

%% Connectivity

for subj=1:numel(subjID)
    
    % Electrode Reference sheet
    electrode_reference=readtable(fullfile(datadir,'Electrode_Reference.xlsx'),'ReadVariableNames',0,'Sheet',subjID{subj});
    
    
    % Subject Directory    
    subjdir=fullfile(datadir,subjID{subj});
    subjdir_abv=['P',subjnum{subj}{1}];
    
    % Baseline EDF
    baselineEDF=fullfile(subjdir,[subjID{subj},' Baseline'],[subjdir_abv,' Baseline.edf']);
    
    % Find seizure files
    seizureEDF={dir(fullfile(subjdir,[subjdir_abv,'*'])).name};
    
    % Make data storage folder
    
    mkdir(fullfile(subjdir,'matdata'))
  
        
    for freq=1:size(foi,2)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%% Baseline %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        Electrodes =electrode_reference{:,1};
        % Remove 10th electrode
        Electrodes(endsWith(Electrodes,'10'))=[];
        
        % Load EDF and divide EEG to trials
        cfg = [];
        cfg.trialdef.triallength = trial_length;
        cfg.dataset     = baselineEDF;
        cfg.continuous  = 'yes';
        cfg.channel     = Electrodes;
        cfg = ft_definetrial(cfg);


        % Filters
        % cfg = [];
        cfg.hpfiltord   = 5;
        cfg.hpfilter    = 'yes';
        cfg.hpfreq      = 1;
%         cfg.dftfilter   = 'yes';
%         cfg.dftfreq     = [60 120 180];
        cfg.bsfilter    = 'yes';
        cfg.bsfreq      = [58 62;118 122;178 182];
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


        con=mean(conn.cohspctrm,2);
        conr=zeros(numel(Electrodes)-1,numel(Electrodes)-1);

        temp1_groupcon=[];
        temp2_groupcon=[];
        ylabelcon=[];
        xlabelcon=[];

        for i=1:length(Electrodes)-1
            if i==1
               countb=1;
               counte=numel(Electrodes)-1;
               conr(i:end,i)=con(countb:counte);
            else
                countb=countb+counte;
                counte=counte-1;
                conr(i:end,i)=con(countb:countb+counte-1);
            end
        end

        count=0;
        for z=1:3:numel(Electrodes)-3
            count=count+1;
            temp1_groupcon=conr(:,z:z+2);
            temp1_groupcon(~all(temp1_groupcon,2),:)=0;
            temp2_groupcon(:,count)=mean(temp1_groupcon,2);
        end 

        temp2_groupcon(~any(temp2_groupcon,2),:)=[];

        count=0;
        for z=1:3:size(temp2_groupcon,1)-1
            count=count+1;
            con_mean(count,:)=mean(temp2_groupcon(z:z+2,:),1);
        end
        
        
        count=0;
        for z=1:3:numel(Electrodes)
            count=count+1;
            xlabelcon{count}=[Electrodes{z},'-',num2str(str2double(regexp(Electrodes{z}, '\d+', 'match'))+2)];
        end

        count=0;
        for z=1:3:numel(Electrodes)-3
            count=count+1;
            ylabelcon{count}=[Electrodes{z+3},'-',num2str(str2double(regexp(Electrodes{z+3}, '\d+', 'match'))+2)];
        end


        figure('Name',[subjID{subj},' ',foi{1,freq},' Baseline'])
        imagesc(con_mean)
        set(gca, 'XTick', 1:size(con_mean,2)); % center x-axis ticks on bins
        set(gca, 'YTick', 1:size(con_mean,1)); % center y-axis ticks on bins
        set(gca, 'YTickLabel', ylabelcon); % set y-axis labels
        set(gca, 'XTickLabel', xlabelcon,'XTickLabelRotation',90); % set x-axis labels
        title('Baseline Alpha Coherence Grouped', 'FontSize', 14); % set title
        colormap('jet'); % set the colorscheme
        colorbar; % enable colorbar 

        save([fullfile(subjdir,'matdata',[foi{1,freq},'_coh_baseline_',subjID{subj}]),'.mat'],'con_mean_total')


        %%%%%%%%%%%%%%%%%%%%%%%%%%%% Sezuire CUSTOM EPOCH  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        

        % Define info
        trialnames={'Baseline','Pre-transition','Post-transition','Mid Sezuire','Late Sezuire','Early Post','Late Post'};
        
              
        
        for sez=1:numel(seizureEDF)
            
            Electrodes=electrode_reference{:,sez+1};
            
            % Remove 10th electrode
            Electrodes(endsWith(Electrodes,'10'))=[];
            
            % Remove empty cells
            Electrodes=Electrodes(~cellfun('isempty',Electrodes));
            
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

            cfg.hpfiltord   = 5;
            cfg.hpfilter    = 'yes';
            cfg.hpfreq      = 1;
            cfg.bsfilter    = 'yes';
            cfg.bsfreq      = [58 62;118 122;178 182];
            cfg.channel     = Electrodes;
            cfg.dataset     = fullfile(subjdir,seizureEDF{sez});
            

            % Add trial
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
            
            
            figure('Name',[subjID{subj},' ',foi{1,freq},' ',seizureEDF{sez}])
            clearvars con_mean
            for q=1:numel(data_eeg.trial)
                cfg                 = [];
                cfg.method          = 'coh';
                cfg.trials          = q;
                conn                = ft_connectivityanalysis(cfg, freq_csd);

                con=mean(conn.cohspctrm,2);
                conr=zeros(numel(Electrodes)-1,numel(Electrodes)-1);

                temp1_groupcon=[];
                temp2_groupcon=[];
                ylabelcon=[];
                xlabelcon=[];

                for i=1:length(Electrodes)-1
                    if i==1
                       countb=1;
                       counte=numel(Electrodes)-1;
                       conr(i:end,i)=con(countb:counte);
                    else
                        countb=countb+counte;
                        counte=counte-1;
                        conr(i:end,i)=con(countb:countb+counte-1);
                    end
                end

                count=0;
                for z=1:3:numel(Electrodes)-3
                    count=count+1;
                    temp1_groupcon=conr(:,z:z+2);
                    temp1_groupcon(~all(temp1_groupcon,2),:)=0;
                    temp2_groupcon(:,count)=mean(temp1_groupcon,2);
                end 

                temp2_groupcon(~any(temp2_groupcon,2),:)=[];

                count=0;
                for z=1:3:size(temp2_groupcon,1)-1
                    count=count+1;
                    con_mean(count,:,q)=mean(temp2_groupcon(z:z+2,:),1);
                end
                
                count=0;
                for z=1:3:numel(Electrodes)
                    count=count+1;
                    xlabelcon{count}=[Electrodes{z},'-',num2str(str2double(regexp(Electrodes{z}, '\d+', 'match'))+2)];
                end

                count=0;
                for z=1:3:numel(Electrodes)-3
                    count=count+1;
                    ylabelcon{count}=[Electrodes{z+3},'-',num2str(str2double(regexp(Electrodes{z+3}, '\d+', 'match'))+2)];
                end

                subplot(3,3,q)
                imagesc(con_mean(:,:,q))
                set(gca, 'XTick', 1:size(con_mean,2)); % center x-axis ticks on bins
                set(gca, 'YTick', 1:size(con_mean,1)); % center y-axis ticks on bins
                set(gca, 'YTickLabel', ylabelcon); % set y-axis labels
                set(gca, 'XTickLabel', xlabelcon,'XTickLabelRotation',90); % set x-axis labels
                title(trialnames{q}, 'FontSize', 14); % set title
                colormap('jet'); % set the colorscheme
                colorbar; % enable colorbar
            end
            save([fullfile(subjdir,'matdata',[foi{1,freq},'_coh_seizure_',seizureEDF{sez}]),'.mat'],'con_mean')
         end
    end
end
%% iEEG Reconstruction



%-------------------------- Preprocessing MRI------------------------
mri = ft_read_mri(['C:\Box Sync\Allen_Bonhila_EEG\Patient 001\Images\Nifti\Pre\97499868_T1_MPRAGE_3D_VOL_SAG_20140509161556_2.nii']);

% determine left-right axis (x,y,or z)
% if right is (+) and left is (-) then correct orientation
ft_determine_coordsys(mri);



% press r --> define right hemisphere
% press a --> define anterior commissure
% press p --> define posterior commissure
% press z --> interhemispheric point along the positive midline
cfg           = [];
cfg.method    = 'interactive';
cfg.coordsys  = 'acpc';
mri_acpc = ft_volumerealign(cfg, mri);

% Save file
cfg           = [];
cfg.filename  = [subjID{subj} '_MR_acpc'];
cfg.filetype  = 'nifti';
cfg.parameter = 'anatomy';
ft_volumewrite(cfg, mri_acpc);


% Cortical Extraction with FreeSurfer 
% fshome     = 'C:\Box Sync\Allen_Bonhila_EEG\EEG_toolboxes\Matlab\fieldtrip-20200607\external\freesurfer';
% subdir     = 'C:\Box Sync\Allen_Bonhila_EEG\Patient 001';
% mrfile     = 'C:\Box Sync\Allen_Bonhila_EEG\Patient 001\Patient001_MR_acpc.nii';
% system(['export FREESURFER_HOME=' fshome '; ' ...
% 'source $FREESURFER_HOME/SetUpFreeSurfer.sh; ' ...
% 'mri_convert -c -oc 0 0 0 ' mrfile ' ' [subdir '\tmp.nii'] '; ' ...
% 'recon-all -i ' [subdir '/tmp.nii'] ' -s ' 'freesurfer' ' -sd ' subdir ' -all'])


% Import the extracted cortical surfaces into the MATLAB workspace and examine their quality.
pial_lh = ft_read_headshape([subjdir,'freesurferdata\surf\lh.pial.T1']);
pial_lh.coordsys = 'acpc';
ft_plot_mesh(pial_lh);
lighting gouraud;
camlight;

pial_rh = ft_read_headshape([subjdir,'freesurferdata\surf\rh.pial.T1']);
pial_rh.coordsys = 'acpc';
ft_plot_mesh(pial_rh);
lighting gouraud;
camlight;

%  Import the FreeSurfer-processed MRI into the MATLAB workspace for the purpose of fusing with the CT scan at a later step,
fsmri_acpc = ft_read_mri('freesurferdata/mri/T1.mgz');
fsmri_acpc.coordsys = 'acpc';

%-------------------------- Preprocessing of the anatomical CT ------------------------
ct = ft_read_mri(['C:\Box Sync\Allen_Bonhila_EEG\Patient 001\Images\Nifti\Post\19761842_SAG_MPRAGE_20141119001814_3.nii']);


% press l --> define left peri-auricular point (behind ear canal)
% press r --> define right peri-auricular point (behind ear canal)
% press n --> define nasion
% press z --> interhemispheric point along the positive midline
cfg           = [];
cfg.method    = 'interactive';
cfg.coordsys  = 'ctf';
ct_ctf = ft_volumerealign(cfg, ct);

% Confirm correct orientation (same as MRI)
ct_acpc = ft_convert_coordsys(ct_ctf, 'acpc');


%--------------------------  Fusion of the CT with the MRI ------------------------
cfg             = [];
cfg.method      = 'spm';
cfg.spmversion  = 'spm12';
cfg.coordsys    = 'acpc';
cfg.viewresult  = 'yes';
ct_acpc_f = ft_volumerealign(cfg, ct_acpc, fsmri_acpc);


% Write the MRI-fused anatomical CT out to file.
cfg           = [];
cfg.filename  = [subjID '_CT_acpc_f'];
cfg.filetype  = 'nifti';
cfg.parameter = 'anatomy';
ft_volumewrite(cfg, ct_acpc_f);

%------------------------ Electrode Placement-------------------------------

%Import header
hdr = ft_read_header('C:\Box Sync\Allen_Bonhila_EEG\Patient 001\P001 Baseline\P001 Baseline.edf');

% Localize the electrodes in the post-implant CT
cfg         = [];
cfg.channel = hdr.label;
elec_acpc_f = ft_electrodeplacement(cfg, ct_acpc_f, fsmri_acpc);

% Visualize the MRI along with the electrodes and their labels and examine whether they show expected behavior.
ft_plot_ortho(fsmri_acpc.anatomy, 'transform', fsmri_acpc.transform, 'style', 'intersect');
ft_plot_sens(elec_acpc_f, 'label', 'on', 'fontcolor', 'w');

% Save the resulting electrode information to file
save([subjID '_elec_acpc_f.mat'], 'elec_acpc_f');

%------------------------- Brain shift compensation ----------------------\

% Create Hull
cfg           = [];
cfg.method    = 'cortexhull';
cfg.headshape = 'freesurfer/surf/lh.pial';
%cfg.fshome    = <path to freesurfer home directory>; % for instance, '/Applications/freesurfer'
hull_lh = ft_prepare_mesh(cfg);

save([subjID '_hull_lh.mat'], hull_lh);

% Project the electrode grids to the surface hull of the implanted hemisphere
elec_acpc_fr = elec_acpc_f;
grids = {'LPG*', 'LTG*'};
for g = 1:numel(grids)
cfg             = [];
cfg.channel     = grids{g};
cfg.keepchannel = 'yes';
cfg.elec        = elec_acpc_fr;
cfg.method      = 'headshape';
cfg.headshape   = hull_lh;
cfg.warp        = 'dykstra2012';
cfg.feedback    = 'yes';
elec_acpc_fr = ft_electroderealign(cfg);
end

% Visualize the cortex and electrodes together and examine whether they show expected behavior
ft_plot_mesh(pial_lh);
ft_plot_sens(elec_acpc_fr);
view([-55 10]);
material dull;
lighting gouraud;
camlight;

ft_plot_mesh(pial_rh);
ft_plot_sens(elec_acpc_fr);
view([-55 10]);
material dull;
lighting gouraud;
camlight;

save([subjID '_elec_acpc_fr.mat'], 'elec_acpc_fr');

% ----------------------- Volume-based registration -------------------- 
cfg            = [];
cfg.nonlinear  = 'yes';
cfg.spmversion = 'spm12';
cfg.spmmethod  = 'new';
fsmri_mni = ft_volumenormalise(cfg, fsmri_acpc);

elec_mni_frv = elec_acpc_fr;
elec_mni_frv.elecpos = ft_warp_apply(fsmri_mni.params, elec_acpc_fr.elecpos, 'individual2sn');
elec_mni_frv.chanpos = ft_warp_apply(fsmri_mni.params, elec_acpc_fr.chanpos, 'individual2sn');
elec_mni_frv.coordsys = 'mni';

% Visualize the coritcal mesh extracted from the standard MNI
[ftver, ftpath] = ft_version;
load([ftpath filesep 'template/anatomy/surface_pial_left.mat']);
ft_plot_mesh(mesh);
ft_plot_sens(elec_mni_frv);
view([-90 20]);
material dull;
lighting gouraud;
camlight;

save([subjID '_elec_mni_frv.mat'], 'elec_mni_frv');

% Anatomical labeling
atlas = ft_read_atlas([ftpath filesep 'template/atlas/aal/ROI_MNI_V4.nii']);

cfg            = [];
cfg.roi        = elec_mni_frv.chanpos(match_str(elec_mni_frv.label,'LHH1'),:);
cfg.atlas      = atlas;
cfg.inputcoord = 'mni';
cfg.output     = 'label';
labels = ft_volumelookup(cfg, atlas);

[~, indx] = max(labels.count);
labels.name(indx)


% ---------------------------sEEG Functional Workflow---------------------------

% Define the trials
cfg                     = [];
cfg.dataset             = 'C:\Box Sync\Allen_Bonhila_EEG\Patient 001\P001 Baseline\P001 Baseline.edf';
cfg.trialdef.eventtype  = 'TRIGGER';
cfg.trialdef.eventvalue = 4;
cfg.trialdef.prestim    = 0.4;
cfg.trialdef.poststim   = 0.9;
cfg = ft_definetrial(cfg);

% Import the data segments of interest into the MATLAB workspace and filter the data for high-frequency and power line noise 
cfg.demean         = 'yes';
cfg.baselinewindow = 'all';
cfg.lpfilter       = 'yes';
cfg.lpfreq         = 200;
cfg.padding        = 2;
cfg.padtype        = 'data';
cfg.bsfilter       = 'yes';
cfg.bsfiltord      = 3;
cfg.bsfreq         = [59 61; 119 121; 179 181];
data = ft_preprocessing(cfg);

% Add the elec structure originating from the anatomical workflow and save the preprocessed electrophysiological data to file. 
data.elec = elec_acpc_fr;
save([subjID '_data.mat'], 'data');

%  Inspect the neural recordings 
cfg          = [];
cfg.viewmode = 'vertical';
cfg = ft_databrowser(cfg, data);

% Mark the bad segments by drawing a box around the corrupted signal. Write down the labels of bad channels.
data = ft_rejectartifact(cfg, data);

% Re-montage the cortical grids to a common average reference in order to remove noise that is shared across all channels. 
% Bad channels identified in Step 39 can be excluded from this step by adding those channels to cfg.channel with a minus prefix. 
% That is, cfg.channel = {‘LPG’, ‘LTG’, ‘-LPG1’} if one were to exclude the LPG1 channel from the list of LPG and LTG channels.
cfg             = [];
cfg.channel     = {'LPG*', 'LTG*'};
cfg.reref       = 'yes';
cfg.refchannel  = 'all';
cfg.refmethod   = 'avg';
reref_grids = ft_preprocessing(cfg, data);

% Apply bipolar montage to depth electrode

depths = {'RAM*', 'RHH*', 'RTH*', 'ROC*', 'LAM*', 'LHH*', 'LTH*'};
for d = 1:numel(depths)
cfg            = [];
cfg.channel    = ft_channelselection(depths{d}, data.label);
cfg.reref      = 'yes';
cfg.refchannel = 'all';
cfg.refmethod  = 'bipolar';
cfg.updatesens = 'yes';
reref_depths{d} = ft_preprocessing(cfg, data);
end

% Combine the data from both electrode types into one data structure for the ease of further processing.
cfg            = [];
cfg.appendsens = 'yes';
reref = ft_appenddata(cfg, reref_grids, reref_depths{:});

save([subjID '_reref.mat'], 'reref');

%---------------- Time Frequency Analysis--------------------
cfg            = [];
cfg.method     = 'mtmconvol';
cfg.toi        = -.3:0.01:.8;
cfg.foi        = 5:5:200;
cfg.t_ftimwin  = ones(length(cfg.foi),1).*0.2;
cfg.taper      = 'hanning';
cfg.output     = 'pow';
cfg.keeptrials = 'no';
freq = ft_freqanalysis(cfg, reref);

save([subjID '_freq.mat'], 'freq');


% interactive plot
cfg            = [];
cfg.headshape  = pial_lh;
cfg.projection = 'orthographic';
cfg.channel    = {'LPG*', 'LTG*'};
cfg.viewpoint  = 'left';
cfg.mask       = 'convex';
cfg.boxchannel = {'LTG30', 'LTG31'};
lay = ft_prepare_layout(cfg, freq);

cfg              = [];
cfg.baseline     = [-.3 -.1];
cfg.baselinetype = 'relchange';
freq_blc = ft_freqbaseline(cfg, freq);

cfg             = [];
cfg.layout      = lay;
cfg.showoutline = 'yes';
ft_multiplotTFR(cfg, freq_blc);

% SEEG data representation

atlas = ft_read_atlas('freesurfer/mri/aparc+aseg.mgz');
atlas.coordsys = 'acpc';
cfg            = [];
cfg.inputcoord = 'acpc';
cfg.atlas      = atlas;
cfg.roi        = {'Right-Hippocampus', 'Right-Amygdala'};
mask_rha = ft_volumelookup(cfg, atlas);

seg = keepfields(atlas, {'dim', 'unit','coordsys','transform'});
seg.brain = mask_rha;
cfg             = [];
cfg.method      = 'iso2mesh';
cfg.radbound    = 2;
cfg.maxsurf     = 0;
cfg.tissue      = 'brain';
cfg.numvertices = 1000;
cfg.smooth      = 3;
cfg.spmversion  = 'spm12';
mesh_rha = ft_prepare_mesh(cfg, seg);

cfg         = [];
cfg.channel = {'RAM*', 'RTH*', 'RHH*'};
freq_sel2 = ft_selectdata(cfg, freq_sel);
cfg              = [];
cfg.funparameter = 'powspctrm';
cfg.funcolorlim  = [-.5 .5];
cfg.method       = 'cloud';
cfg.slice        = '3d';
cfg.nslices      = 2;
cfg.facealpha    = .25;
ft_sourceplot(cfg, freq_sel2, mesh_rha);
view([120 40]);
lighting gouraud;
camlight;

cfg.slice        = '2d';
ft_sourceplot(cfg, freq_sel2, mesh_rha);


cfg         = [];
cfg.method      = 'ar';
cfg.order   = 5;
cfg.ntrials     = 500;
cfg.triallength = 1;
cfg.fsample     = 200;
cfg.nsignal     = 3;

cfg.toolbox = 'bsmart';
mdata       = ft_mvaranalysis(cfg, data);
mdata;


%% sEEG connectome

cfg            = [];
cfg.dataset= 'C:\Box Sync\Allen_Bonhila_EEG\Patient 001\P001 Baseline\P001 Baseline.edf';
cfg.channel= {'LMF1' 'LMF2' 'LMF3' 'LMF4' 'LMF5' 'LMF6' 'LMF7' 'LMF8' 'LMF9' 'LMF10' 'RLF1' 'RLF2' 'RLF3' 'RLF4' 'RLF5' 'RLF6' 'RLF7' 'RLF8' 'RLF9' 'RLF10' 'EA1' 'EA2' 'EA3' 'EA4' 'EA5' 'EA6' 'EA7' 'EA8' 'EA9' 'EA10' }';
cfg.continuous = 'yes';
data           = ft_preprocessing(cfg);

% ---------------Computation of the multivariate autoregressive model---------------

cfg         = [];
cfg.order   = 30;
cfg.toolbox = 'bsmart';
mdata       = ft_mvaranalysis(cfg, data);

%---------------Computation of the spectral transfer function---------------

% Parametric route

cfg        = [];
cfg.method = 'mvar';
mfreq      = ft_freqanalysis(cfg, mdata);

% Non-parametric route
cfg           = [];
cfg.method    = 'mtmfft';
cfg.taper     = 'dpss';
cfg.output    = 'fourier';
cfg.tapsmofrq = 2;
freq          = ft_freqanalysis(cfg, data);

% ---------------Computation and inspection of the connectivity measures ---------------

% coherence coefficient
cfg           = [];
cfg.method    = 'coh';
coh           = ft_connectivityanalysis(cfg, freq);
cohm          = ft_connectivityanalysis(cfg, mfreq);

cfg           = [];
cfg.parameter = 'cohspctrm';
cfg.zlim      = [0 1];
ft_connectivityplot(cfg, coh, cohm);
ft_connectivityplot(cfg, cohm);

% granger causality

cfg           = [];
cfg.method    = 'granger';
granger       = ft_connectivityanalysis(cfg, mfreq);

cfg           = [];
cfg.parameter = 'grangerspctrm';
cfg.zlim      = [0 1];
ft_connectivityplot(cfg, granger);