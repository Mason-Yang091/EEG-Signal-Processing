%% EEG BIOPAC - Complete Digital Filtering, Analysis and Validation
% Input: g5L03_4C_raw.txt (BIOPAC text export)
% Output: figures (PNG), filtered channels (CSV), statistics (CSV), report (TXT)
%
% Requires MATLAB Signal Processing Toolbox.
% MATLAB R2019b or later is recommended (local functions at end of script).
%
% DATA CONFIRMED FROM UPLOADED FILE:
%   12179 samples, dt=0.005 sec, Fs=200 Hz, duration 60.89 sec
%   CH1=raw EEG; CH40=alpha; CH41=beta; CH42=delta; CH43=theta
%
% EXPERIMENT TIMING CONFIRMED BY USER:
% Closed 0-20.20 s; Open 20.20-40.39 s; Closed 40.39 s-end.
% State transitions are user-reported, not encoded in the BIOPAC TXT file.
%
% Reproducibility notes:
%   - Each band uses butter(4,[low high]/(Fs/2),'bandpass') (8th-order
%     *overall* bandpass transfer function); implemented with stable SOS.
%   - filtfilt is a zero-phase, offline, forward-backward operation.
%   - PSD and band powers are estimated using Welch's method.
%   - Coherence near 1 is NOT sufficient to prove equal amplitude/phase.

clear; close all; clc;

%% 0. USER CONFIGURATION - EDIT THIS SECTION FIRST
cfg.inputFile = fullfile(fileparts(mfilename('fullpath')), 'g5L03_4C_raw.txt');
cfg.outputFolder = fullfile(fileparts(mfilename('fullpath')), 'EEG_Results');

% Epochs: left inclusive, right exclusive. End time set using data after load.
% Confirmed by user: closed -> open -> closed, transitions 20.20, 40.39 s.
cfg.transitionTimes_s = [20.20 40.39];
cfg.epochs = [0 20.20; 20.20 40.39; 40.39 NaN];
cfg.epochLabels = {'Eyes_Closed_1','Eyes_Open','Eyes_Closed_2'};
cfg.conditionsVerified = true;
% Do not join noncontiguous closed EEG samples before estimating PSD.

cfg.bandNames = {'Delta','Theta','Alpha','Beta'};
cfg.bandLimits = [0.5 4; 4 8; 8 13; 13 30];
cfg.referenceChannelNames = {'CH42','CH43','CH40','CH41'};
cfg.filterPrototypeOrder = 4;  % bandpass overall order = 8
cfg.welchWindowSec = 4;         % Hamming window, 50% overlap
cfg.nfft = 4096;
cfg.maxLagSec = 0.25;          % correlation search range for diagnostics
cfg.waveformPreviewSec = 3;
cfg.figureVisible = 'off';     % use 'on' to inspect figures interactively

if ~isfolder(cfg.outputFolder), mkdir(cfg.outputFolder); end
assert(size(cfg.epochs,1) == numel(cfg.epochLabels), 'Epoch/label size mismatch.');
assert(size(cfg.epochs,2)==2, 'cfg.epochs must have 2 columns.');
if ~cfg.conditionsVerified
    warning(['Eyes-open / eyes-closed assignment is NOT VERIFIED by the TXT file. ' ...
             'Edit cfg.epochs and cfg.epochLabels based on experimental log.']);
end

%% 1. READ BIOPAC TEXT, CHECK SAMPLING AND CHANNELS
D = readBIOPACtxt(cfg.inputFile);
t = D(:,1); raw = D(:,2);       % time (s), EEG (uV)
ref = [D(:,5),D(:,6),D(:,3),D(:,4)]; % Delta CH42, Theta CH43, Alpha CH40, Beta CH41

dt = median(diff(t)); Fs = 1/dt;
assert(all(isfinite(D(:))), 'Missing/invalid values in BIOPAC TXT.');
assert(all(diff(t)>0), 'Time values must increase.');
assert(max(abs(diff(t)-dt))<1e-5, 'Time steps are irregular. Inspect input file.');
assert(cfg.bandLimits(end,2)<Fs/2, 'Nyquist frequency too low for Beta band.');
cfg.epochs(end,2) = t(end)+dt;  % Include the final 60.89 s sample
assert(all(cfg.epochs(:,1)>=t(1)) && all(cfg.epochs(:,2)<=t(end)+dt+1e-9), ...
    'Epoch bounds exceed the recording duration.');

fprintf('Loaded %d samples (%.3f s), sampling rate %.5f Hz.\n', numel(t),t(end)-t(1),Fs);
fprintf('CH1=EEG; CH42=delta; CH43=theta; CH40=alpha; CH41=beta (uV).\n');

%% 2. DESIGN FOUR BANDPASS FILTERS AND PROCESS FULL RECORDING
nBands = size(cfg.bandLimits,1);
filtered = zeros(numel(raw),nBands);
filters = cell(nBands,1);
fResponse = linspace(0,Fs/2,3000).';
Hresponse = zeros(numel(fResponse),nBands);

for k=1:nBands
    Wn = cfg.bandLimits(k,:)/(Fs/2);
    [z,p,g0] = butter(cfg.filterPrototypeOrder,Wn,'bandpass');
    [sos,g] = zp2sos(z,p,g0);
    % Stable SOS version of filtfilt; do not use an unstable high-order
    % direct-form [b,a] implementation for low-frequency EEG bands.
    filtered(:,k) = filtfilt(sos,g,raw);
    filters{k} = struct('sos',sos,'gain',g);
    H = g*ones(size(fResponse));
    for j=1:size(sos,1)
        H = H .* freqz(sos(j,1:3),sos(j,4:6),fResponse,Fs);
    end
    Hresponse(:,k)=abs(H);
end

filteredTable=table(t,raw,filtered(:,1),filtered(:,2),filtered(:,3),filtered(:,4), ...
    ref(:,1),ref(:,2),ref(:,3),ref(:,4), ...
    'VariableNames',{'Time_s','EEG_CH1_uV','Designed_Delta_uV', ...
    'Designed_Theta_uV','Designed_Alpha_uV','Designed_Beta_uV', ...
    'BIOPAC_Delta_CH42_uV','BIOPAC_Theta_CH43_uV', ...
    'BIOPAC_Alpha_CH40_uV','BIOPAC_Beta_CH41_uV'});
writetable(filteredTable,fullfile(cfg.outputFolder,'Filtered_EEG_AllChannels.csv'));

%% 3. PLOT RAW SIGNAL AND FOUR FILTER FREQUENCY RESPONSES
fig=figure('Visible',cfg.figureVisible,'Color','w','Position',[100 100 1200 420]);
plot(t,raw,'k','LineWidth',0.7); grid on; hold on;
for i=1:size(cfg.epochs,1)
    xline(cfg.epochs(i,1),'--'); xline(cfg.epochs(i,2),'--');
end
xlabel('Time (s)');ylabel('Raw EEG (\muV)');
title('Raw BIOPAC EEG (CH1): Closed | Open | Closed');
saveFigure(fig,cfg.outputFolder,'01_Raw_EEG.png');

fig=figure('Visible',cfg.figureVisible,'Color','w','Position',[100 100 1150 740]);
for k=1:nBands
    subplot(2,2,k);
    plot(fResponse,20*log10(max(Hresponse(:,k),1e-10)),'LineWidth',1.1);
    xlim([0 40]);ylim([-100 5]);grid on;hold on;
    xline(cfg.bandLimits(k,1),'--');xline(cfg.bandLimits(k,2),'--');
    xlabel('Frequency (Hz)');ylabel('Magnitude (dB)');
    title(sprintf('%s: %.1f to %.1f Hz (Butterworth)',cfg.bandNames{k},cfg.bandLimits(k,:)));
end
saveFigure(fig,cfg.outputFolder,'02_Butterworth_FilterResponses.png');

%% 4. ANALYZE EACH EPOCH / BAND: PSD, BANDPOWER, PEAKS, COMPARISON
wlen = max(32,round(cfg.welchWindowSec*Fs));
overlap = floor(wlen/2);
Nfft = max(cfg.nfft,2^nextpow2(wlen));
win = hamming(wlen);
maxLagSamples=round(cfg.maxLagSec*Fs);

% Initialize output columns (one record per epoch x band).
Epoch = {}; Band = {}; BIOPAC_Channel = {};
EpochStart_s=[]; EpochEnd_s=[];
BandLow_Hz=[]; BandHigh_Hz=[];
DesignedPower_uV2=[]; BIOPACPower_uV2=[]; PowerRatio_DesignOverRef=[];
DesignedPeak_Hz=[]; BIOPACPeak_Hz=[];
Correlation_ZeroLag=[]; MaxAbsCorrelation=[]; BestLag_s=[];
MeanMagnitudeSquaredCoherence=[]; RMSE_uV=[]; NormalizedRMSE=[];

for e=1:size(cfg.epochs,1)
    idx = t>=cfg.epochs(e,1) & t<cfg.epochs(e,2);
    x = raw(idx); time=t(idx);
    if numel(x) < max(2*wlen,1000)
        error('Epoch %d has only %d samples. Reduce window or choose longer epoch.',e,numel(x));
    end

    figTime=figure('Visible',cfg.figureVisible,'Color','w','Position',[100 100 1300 900]);
    figPSD=figure('Visible',cfg.figureVisible,'Color','w','Position',[120 120 1300 900]);
    figCompare=figure('Visible',cfg.figureVisible,'Color','w','Position',[150 150 1300 900]);

    for k=1:nBands
        y = filtered(idx,k); r=ref(idx,k);
        [Py,f]=pwelch(y,win,overlap,Nfft,Fs);
        [Pr,~]=pwelch(r,win,overlap,Nfft,Fs);
        [Cx,fc]=mscohere(y,r,win,overlap,Nfft,Fs);
        inband=(f>=cfg.bandLimits(k,1) & f<=cfg.bandLimits(k,2));
        inCoherence=(fc>=cfg.bandLimits(k,1) & fc<=cfg.bandLimits(k,2));

        powerY=trapz(f(inband),Py(inband));
        powerR=trapz(f(inband),Pr(inband));
        fband=f(inband);
        [~,imax]=max(Py(inband));peakY=fband(imax);
        [~,imax]=max(Pr(inband));peakR=fband(imax);
        c=pearsonPair(y,r);
        [xc,lags]=xcorr(y-mean(y),r-mean(r),maxLagSamples,'coeff');
        [~,bestI]=max(abs(xc));
        bestC=xc(bestI);bestLag=lags(bestI)/Fs;
        coherence=mean(Cx(inCoherence),'omitnan');
        rmse=sqrt(mean((y-r).^2));
        nrmse=rmse/max(std(r),eps);

        Epoch{end+1,1}=cfg.epochLabels{e};
        Band{end+1,1}=cfg.bandNames{k};
        BIOPAC_Channel{end+1,1}=cfg.referenceChannelNames{k};
        EpochStart_s(end+1,1)=cfg.epochs(e,1);
        EpochEnd_s(end+1,1)=cfg.epochs(e,2);
        BandLow_Hz(end+1,1)=cfg.bandLimits(k,1);
        BandHigh_Hz(end+1,1)=cfg.bandLimits(k,2);
        DesignedPower_uV2(end+1,1)=powerY;
        BIOPACPower_uV2(end+1,1)=powerR;
        PowerRatio_DesignOverRef(end+1,1)=powerY/max(powerR,eps);
        DesignedPeak_Hz(end+1,1)=peakY;
        BIOPACPeak_Hz(end+1,1)=peakR;
        Correlation_ZeroLag(end+1,1)=c;
        MaxAbsCorrelation(end+1,1)=bestC;
        BestLag_s(end+1,1)=bestLag;
        MeanMagnitudeSquaredCoherence(end+1,1)=coherence;
        RMSE_uV(end+1,1)=rmse;
        NormalizedRMSE(end+1,1)=nrmse;

        % Figure: time-domain waveforms, 4 rows x 2 columns
        figure(figTime);
        subplot(4,2,2*k-1);plot(time,y,'b');grid on;
        ylabel('\muV');title([cfg.bandNames{k} ' / Designed']);
        subplot(4,2,2*k);plot(time,r,'r');grid on;
        ylabel('\muV');title([cfg.bandNames{k} ' / BIOPAC']);
        if k==4
            subplot(4,2,7);xlabel('Time (s)');
            subplot(4,2,8);xlabel('Time (s)');
        end

        % Figure: PSD comparison (full band + inset-like zoom)
        figure(figPSD);
        subplot(2,2,k);
        plot(f,10*log10(max(Py,realmin)),'b','LineWidth',1.2);hold on;
        plot(f,10*log10(max(Pr,realmin)),'r--','LineWidth',1.1);
        xlim([0 min(40,Fs/2)]);grid on;
        xline(cfg.bandLimits(k,1),':');xline(cfg.bandLimits(k,2),':');
        xlabel('Frequency (Hz)');ylabel('PSD (dB re \muV^2/Hz)');
        title(sprintf('%s / PSD / designed=%.3g, BIOPAC=%.3g \muV^2', ...
            cfg.bandNames{k},powerY,powerR));
        legend('Designed','BIOPAC Reference','Location','best');

        % Figure: short waveform overlay + MSC as 2-column row
        figure(figCompare);
        preview=(time<time(1)+cfg.waveformPreviewSec);
        subplot(4,2,2*k-1);
        plot(time(preview),y(preview),'b');hold on;
        plot(time(preview),r(preview),'r--');grid on;
        ylabel('\muV');title(sprintf('%s: time overlay (r=%.3f)',cfg.bandNames{k},c));
        if k==1,legend('Designed','BIOPAC','Location','best');end
        subplot(4,2,2*k);
        plot(fc,Cx,'k','LineWidth',1.1);grid on;
        xlim([0 min(40,Fs/2)]);ylim([0 1.05]);
        xline(cfg.bandLimits(k,1),':');xline(cfg.bandLimits(k,2),':');
        ylabel('MSC');title(sprintf('%s: mean in-band MSC = %.3f',cfg.bandNames{k},coherence));
        if k==4
            subplot(4,2,7);xlabel('Time (s)');
            subplot(4,2,8);xlabel('Frequency (Hz)');
        end
    end
    safeName=regexprep(cfg.epochLabels{e},'[^a-zA-Z0-9_-]','_');
    saveFigure(figTime,cfg.outputFolder,['03_TimeDomain_' safeName '.png']);
    saveFigure(figPSD,cfg.outputFolder,['04_PSD_Designed_vs_BIOPAC_' safeName '.png']);
    saveFigure(figCompare,cfg.outputFolder,['05_Correlation_Coherence_' safeName '.png']);
end

%% 5. SAVE DETAILED STATISTICS AND RELATIVE BAND POWERS
T=table(Epoch,Band,BIOPAC_Channel,EpochStart_s,EpochEnd_s, ...
    BandLow_Hz,BandHigh_Hz,DesignedPower_uV2,BIOPACPower_uV2, ...
    PowerRatio_DesignOverRef,DesignedPeak_Hz,BIOPACPeak_Hz, ...
    Correlation_ZeroLag,MaxAbsCorrelation,BestLag_s, ...
    MeanMagnitudeSquaredCoherence,RMSE_uV,NormalizedRMSE);
writetable(T,fullfile(cfg.outputFolder,'EEG_Band_Comparison.csv'));
disp(T);

% Compare relative band contributions across the four analyzed bands.
T.DesignedRelativePower_pct=zeros(height(T),1);
T.BIOPACRelativePower_pct=zeros(height(T),1);
for e=1:size(cfg.epochs,1)
    rows=(1:nBands)+(e-1)*nBands;
    T.DesignedRelativePower_pct(rows)=100*T.DesignedPower_uV2(rows)/sum(T.DesignedPower_uV2(rows));
    T.BIOPACRelativePower_pct(rows)=100*T.BIOPACPower_uV2(rows)/sum(T.BIOPACPower_uV2(rows));
end
writetable(T,fullfile(cfg.outputFolder,'EEG_Band_Comparison.csv'));

% Bandpower figure: compare epochs for each band (four bars per band)
fig=figure('Visible',cfg.figureVisible,'Color','w','Position',[100 100 1200 680]);
for k=1:nBands
    subplot(2,2,k);
    rows=k:nBands:height(T);
    vals=[T.DesignedPower_uV2(rows),T.BIOPACPower_uV2(rows)];
    bar(vals);grid on;
    set(gca,'XTick',1:numel(rows),'XTickLabel',cfg.epochLabels);
    ylabel('Integrated PSD (\muV^2)');
    title([cfg.bandNames{k} ' Band Power']);
    legend('Designed','BIOPAC','Location','best');
end
saveFigure(fig,cfg.outputFolder,'06_BandPower_Comparison.png');

% Figure: zero-lag correlation and mean coherence for every band and epoch
fig=figure('Visible',cfg.figureVisible,'Color','w','Position',[100 100 1200 650]);
for e=1:size(cfg.epochs,1)
    rows=(1:nBands)+(e-1)*nBands;
    subplot(size(cfg.epochs,1),1,e);
    bar([T.Correlation_ZeroLag(rows),T.MeanMagnitudeSquaredCoherence(rows)]);
    set(gca,'XTick',1:nBands,'XTickLabel',cfg.bandNames);
    ylabel('Coefficient');ylim([-1 1.05]);grid on;
    legend('Pearson zero-lag r','Mean magnitude-squared coherence','Location','best');
    title([cfg.epochLabels{e} ' / MATLAB vs BIOPAC']);
end
saveFigure(fig,cfg.outputFolder,'07_Comparison_Metrics.png');

%% 5B. STATE COMPARISON: TWO CLOSED EPOCHS VS ONE OPEN EPOCH
% Compare CLOSED 1 and CLOSED 2 separately. A duration-weighted average
% of their independently estimated powers is used for the CLOSED summary;
% EEG samples are never concatenated for Welch PSD computation.
closedEpochs = find(startsWith(cfg.epochLabels,'Eyes_Closed'));
openEpochs   = find(strcmp(cfg.epochLabels,'Eyes_Open'));
assert(numel(closedEpochs)==2 && numel(openEpochs)==1, ...
    'Expected exactly 2 closed epochs and 1 open epoch.');
closedDurations = cfg.epochs(closedEpochs,2)-cfg.epochs(closedEpochs,1);
closedWeights = closedDurations/sum(closedDurations);

BandName = cfg.bandNames(:);
Closed1Power_uV2 = zeros(nBands,1);
OpenPower_uV2 = zeros(nBands,1);
Closed2Power_uV2 = zeros(nBands,1);
ClosedWeightedPower_uV2 = zeros(nBands,1);
ClosedOverOpenRatio = zeros(nBands,1);
BIOPACClosedWeightedPower_uV2 = zeros(nBands,1);
BIOPACOpenPower_uV2 = zeros(nBands,1);
BIOPACClosedOverOpenRatio = zeros(nBands,1);
for k=1:nBands
    c1=(closedEpochs(1)-1)*nBands+k;
    oo=(openEpochs(1)-1)*nBands+k;
    c2=(closedEpochs(2)-1)*nBands+k;
    Closed1Power_uV2(k)=T.DesignedPower_uV2(c1);
    OpenPower_uV2(k)=T.DesignedPower_uV2(oo);
    Closed2Power_uV2(k)=T.DesignedPower_uV2(c2);
    ClosedWeightedPower_uV2(k)=closedWeights(1)*Closed1Power_uV2(k) ...
        +closedWeights(2)*Closed2Power_uV2(k);
    ClosedOverOpenRatio(k)=ClosedWeightedPower_uV2(k)/max(OpenPower_uV2(k),eps);
    BIOPACClosedWeightedPower_uV2(k)= ...
        closedWeights(1)*T.BIOPACPower_uV2(c1) ...
        +closedWeights(2)*T.BIOPACPower_uV2(c2);
    BIOPACOpenPower_uV2(k)=T.BIOPACPower_uV2(oo);
    BIOPACClosedOverOpenRatio(k)=BIOPACClosedWeightedPower_uV2(k) ...
        /max(BIOPACOpenPower_uV2(k),eps);
end
StateSummary=table(BandName,Closed1Power_uV2,OpenPower_uV2, ...
    Closed2Power_uV2,ClosedWeightedPower_uV2,ClosedOverOpenRatio, ...
    BIOPACClosedWeightedPower_uV2,BIOPACOpenPower_uV2,BIOPACClosedOverOpenRatio);
writetable(StateSummary,fullfile(cfg.outputFolder,'EEG_EyesClosed_vs_Open.csv'));
disp(StateSummary);

fig=figure('Visible',cfg.figureVisible,'Color','w','Position',[100 100 1200 750]);
for k=1:nBands
    subplot(2,2,k);
    rows=k:nBands:height(T);
    bar([T.DesignedPower_uV2(rows),T.BIOPACPower_uV2(rows)]);
    set(gca,'XTick',1:3,'XTickLabel',{'Closed 1','Open','Closed 2'});
    grid on;ylabel('Band Power (uV^2)');
    title(sprintf('%s: MATLAB closed/open = %.3f', ...
        cfg.bandNames{k},ClosedOverOpenRatio(k)));
    legend('Designed','BIOPAC','Location','best');
end
saveFigure(fig,cfg.outputFolder,'09_ThreeStage_Closed_Open_Closed.png');

% Overlay power spectral densities by actual experimental state.
fig=figure('Visible',cfg.figureVisible,'Color','w','Position',[100 100 1250 820]);
for k=1:nBands
    c1=t>=cfg.epochs(1,1) & t<cfg.epochs(1,2);
    op=t>=cfg.epochs(2,1) & t<cfg.epochs(2,2);
    c2=t>=cfg.epochs(3,1) & t<cfg.epochs(3,2);
    [p1,ff]=pwelch(filtered(c1,k),win,overlap,Nfft,Fs);
    [po,~]=pwelch(filtered(op,k),win,overlap,Nfft,Fs);
    [p2,~]=pwelch(filtered(c2,k),win,overlap,Nfft,Fs);
    pc=closedWeights(1)*p1+closedWeights(2)*p2;
    subplot(2,2,k);
    plot(ff,10*log10(max(pc,realmin)),'b','LineWidth',1.4);hold on;
    plot(ff,10*log10(max(po,realmin)),'r--','LineWidth',1.3);
    xlim([0 35]);grid on;
    xline(cfg.bandLimits(k,1),':');xline(cfg.bandLimits(k,2),':');
    xlabel('Frequency (Hz)');ylabel('PSD (dB re uV^2/Hz)');
    title([cfg.bandNames{k} ' | Designed Filter']);
    legend('Closed (weighted PSD)','Open','Location','best');
end
saveFigure(fig,cfg.outputFolder,'10_EyesClosed_vs_Open_PSD.png');

%% 6. POWER TIME COURSE ACROSS THE WHOLE RECORDING
% Useful for inspection; a rise/fall cannot independently identify eye state.
windowSec=4;stepSec=2;
centers=[];powerTime=[];
for startSec=t(1):stepSec:(t(end)-windowSec)
    mask=t>=startSec & t<startSec+windowSec;
    if nnz(mask)<windowSec*Fs*0.9,continue;end
    tmp=zeros(1,nBands);
    for k=1:nBands
        [pseg,fseg]=pwelch(filtered(mask,k), ...
            hamming(min(wlen,nnz(mask))), floor(min(wlen,nnz(mask))/2), Nfft,Fs);
        sel=fseg>=cfg.bandLimits(k,1) & fseg<=cfg.bandLimits(k,2);
        tmp(k)=trapz(fseg(sel),pseg(sel));
    end
    centers(end+1,1)=startSec+windowSec/2;
    powerTime(end+1,:)=tmp;
end
if ~isempty(centers)
    powerTrajectory=array2table([centers powerTime], ...
        'VariableNames',{'WindowCenter_s','DeltaPower_uV2','ThetaPower_uV2', ...
                         'AlphaPower_uV2','BetaPower_uV2'});
    writetable(powerTrajectory,fullfile(cfg.outputFolder,'EEG_BandPower_Timeline.csv'));
    fig=figure('Visible',cfg.figureVisible,'Color','w','Position',[100 100 1200 580]);
    plot(centers,powerTime,'LineWidth',1.25);grid on;
    xlabel('Window center (s)');ylabel('Band power (\muV^2)');
    legend(cfg.bandNames,'Location','best');
    title('EEG Band Power over Time (Closed -> Open -> Closed)');
    hold on;
    xline(cfg.transitionTimes_s(1),'--','Eyes Open');
    xline(cfg.transitionTimes_s(2),'--','Eyes Closed');
    saveFigure(fig,cfg.outputFolder,'08_BandPower_Timeline.png');
end

%% 7. GENERATE MACHINE-READABLE MAT FILE + EXPLANATORY TEXT REPORT
save(fullfile(cfg.outputFolder,'EEG_Analysis_Workspace.mat'), ...
    'cfg','t','raw','ref','filtered','Fs','T','StateSummary','filters','Hresponse','fResponse');

reportFile=fullfile(cfg.outputFolder,'EEG_Analysis_Report.txt');
fid=fopen(reportFile,'w','n','UTF-8');
assert(fid>0,'Unable to write report.');
fprintf(fid,'EEG SIGNAL PROCESSING - BIOPAC / MATLAB\n');
fprintf(fid,'=====================================\n\n');
fprintf(fid,'Input file: %s\n',cfg.inputFile);
fprintf(fid,'Samples: %d; Fs = %.6f Hz; Duration = %.3f seconds\n',numel(t),Fs,t(end)-t(1));
fprintf(fid,'Unit: microvolts (uV). CH1 = raw EEG.\n');
fprintf(fid,'BIOPAC reference mapping: CH42=Delta, CH43=Theta, CH40=Alpha, CH41=Beta.\n\n');
fprintf(fid,'Experimental state changes (user log): 20.20 s, 40.39 s.\n');
fprintf(fid,'Experimental states confirmed by user: %d\n',cfg.conditionsVerified);
if ~cfg.conditionsVerified
    fprintf(fid,['WARNING: The condition labels below were copied from an EXAMPLE, ' ...
        'NOT established by the input file. Verify the actual order with your log.\n\n']);
end
fprintf(fid,'Filter: MATLAB butter(%d,[lo hi]/(Fs/2),''bandpass''), realized as SOS\n',cfg.filterPrototypeOrder);
fprintf(fid,'Overall bandpass transfer-function order: %d\n',2*cfg.filterPrototypeOrder);
fprintf(fid,'Filtering: filtfilt (zero-phase OFFLINE); PSD: pwelch (window %.1f s).\n',cfg.welchWindowSec);
fprintf(fid,'Correlation: Pearson r, zero-lag; lag search is diagnostic only.\n');
fprintf(fid,'Coherence: average magnitude-squared coherence over each band.\n\n');
for e=1:size(cfg.epochs,1)
    fprintf(fid,'Epoch %d: %s, %.2f to %.2f seconds\n',e,cfg.epochLabels{e},cfg.epochs(e,:));
    fprintf(fid,'Band      DesignedPower  BIOPACPower    Ratio      r(0)      MSC\n');
    for k=1:nBands
        i=(e-1)*nBands+k;
        fprintf(fid,'%-8s  %12.5g  %12.5g  %8.4f  %+8.4f  %8.4f\n', ...
            T.Band{i},T.DesignedPower_uV2(i),T.BIOPACPower_uV2(i), ...
            T.PowerRatio_DesignOverRef(i),T.Correlation_ZeroLag(i), ...
            T.MeanMagnitudeSquaredCoherence(i));
    end
    fprintf(fid,'\n');
end
fprintf(fid,'CLOSED vs OPEN BAND POWER (duration-weighted closed periods)\n');
fprintf(fid,'Note: two closed epochs are analyzed separately, NOT concatenated.\n');
for k=1:nBands
    fprintf(fid,'%s: MATLAB closed/open = %.5f; BIOPAC closed/open = %.5f\n', ...
        cfg.bandNames{k},ClosedOverOpenRatio(k),BIOPACClosedOverOpenRatio(k));
end
fprintf(fid,'\n');
fprintf(fid,['\nLIMITATIONS:\n' ...
'1. BIOPAC and MATLAB filters can differ in roll-off, phase, gain, and edge effects.\n' ...
'2. High coherence does not imply identical waveforms, gain, or phase.\n' ...
'3. A frequency band is not a diagnosis; eye labels must be verified.\n' ...
'4. BIOPAC reference channels are processed signals, not independent ground truth.\n' ...
'5. The segment boundaries 20.20/40.39 s come from the experiment log, not the TXT file.\n' ...
'6. Transition-related artifacts near condition switches may affect estimated power.\n']);
fclose(fid);
fprintf('\nDONE. All results saved to:\n%s\n',cfg.outputFolder);

%% Local functions
function X=readBIOPACtxt(filename)
% Parses BIOPAC ASCII with a metadata preamble (possibly non-UTF-8),
% followed by sec,CH1,CH40,CH41,CH42,CH43 and a samples-count line.
    fid=fopen(filename,'r','n','ISO-8859-1');
    assert(fid>0,'Cannot open file: %s',filename);
    cleanup=onCleanup(@()fclose(fid));
    found=false;
    while ~feof(fid)
        s=fgetl(fid);
        if ischar(s) && startsWith(strtrim(s),'sec,CH1')
            found=true;break;
        end
    end
    assert(found,'Could not find BIOPAC header sec,CH1,...');
    fgetl(fid); % skip line with sample count
    X=zeros(13000,6);i=0;
    while ~feof(fid)
        line=fgetl(fid);
        if ~ischar(line) || isempty(strtrim(line)),continue;end
        nums=sscanf(strrep(line,',',' '),'%f');
        if numel(nums)~=6
            error('Expected exactly 6 values on data row %d; found %d.',i+1,numel(nums));
        end
        i=i+1;
        if i>size(X,1),X(end+10000,6)=0;end
        X(i,:)=nums(:).';
    end
    X=X(1:i,:);
    assert(~isempty(X),'No numeric samples found.');
end

function r=pearsonPair(a,b)
    R=corrcoef(a(:),b(:));
    r=R(1,2);
end

function saveFigure(fig,outFolder,fileName)
    filePath=fullfile(outFolder,fileName);
    try
        exportgraphics(fig,filePath,'Resolution',175);
    catch
        saveas(fig,filePath);
    end
    close(fig);
end
