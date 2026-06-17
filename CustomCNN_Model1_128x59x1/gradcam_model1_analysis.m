clear; clc; close all;

%% ============================================================
% GRAD-CAM ANALYSIS FOR MODEL 1 v3 (1s windows, HPF 300 Hz)
%
% Verifies what features the v3 model uses to classify mosquito
% vs background. Specifically checks for:
%   - Wingbeat harmonic attention (400-1500 Hz) → real features
%   - Sub-200 Hz attention → original shortcut returned
%   - Vertical seam attention → looping artifact shortcut
%   - Padding-edge attention → padding artifact (shouldn't exist in v3)
%
% Outputs (in GradCAM_Model1_v3_Final/):
%   - GradCAM_TruePositive.png       individual heatmaps on correct mosquitoes
%   - GradCAM_TrueNegative.png       individual heatmaps on correct backgrounds
%   - GradCAM_FalseNegative.png      individual heatmaps on missed mosquitoes
%   - GradCAM_FalsePositive.png      individual heatmaps on false alarms
%   - GradCAM_AverageAttention.png   THE KEY FIGURE — averaged attention per category
%% ============================================================

%% ----- PATHS -----
modelPath = 'model1_hpf300_v3_acc97.96.mat';

mosquitoTestDir   = 'G:\My Drive\Thesis_Dataset\6_Mosquitos_hpf300_v3\test';
backgroundTestDir = 'G:\My Drive\Thesis_Dataset\ESC-50_hpf300_background_v3\test\background';

outputDir = 'GradCAM_Model1_v3_Final';

%% ----- SETTINGS -----
inputSize = [128 59 1];     % v3: 1-second windows
samplesPerCategory = 8;     % per-category figure
samplesForAvg     = 50;     % for the average-attention figure

% Audio params (only for axis labels)
sampleRate = 16000;
nMels      = 128;
fMin       = 0;
fMax       = sampleRate / 2;

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% ----- LOAD TRAINED MODEL -----
fprintf('========================================\n');
fprintf('GRAD-CAM ANALYSIS — MODEL 1 v3\n');
fprintf('========================================\n\n');

fprintf('Loading model: %s\n', modelPath);

if ~exist(modelPath, 'file')
    error(['Model file not found: %s\n' ...
           'Run this script from the folder where the .mat file lives, ' ...
           'or update modelPath to the full path.'], modelPath);
end

loaded = load(modelPath);

if isfield(loaded, 'model1_detector')
    net = loaded.model1_detector;
    fprintf('  Network loaded from variable: model1_detector\n');
else
    fnames = fieldnames(loaded);
    net = [];
    for i = 1:length(fnames)
        if isa(loaded.(fnames{i}), 'SeriesNetwork') || ...
           isa(loaded.(fnames{i}), 'DAGNetwork')    || ...
           isa(loaded.(fnames{i}), 'dlnetwork')
            net = loaded.(fnames{i});
            fprintf('  Network loaded from variable: %s\n', fnames{i});
            break;
        end
    end
    if isempty(net)
        error('No network found in %s. Variables: %s', ...
              modelPath, strjoin(fnames, ', '));
    end
end

if isfield(loaded, 'accuracy')
    fprintf('  Saved accuracy: %.2f%%\n\n', loaded.accuracy*100);
end

%% ----- LOAD TEST DATA -----
fprintf('Loading test data...\n');
[testImages, testLabels] = loadBinaryFromTwoFolders( ...
    mosquitoTestDir, backgroundTestDir, inputSize);

%% ----- RUN PREDICTIONS -----
fprintf('Running predictions on test set...\n');
[predictions, scores] = classify(net, testImages);
accuracy = mean(predictions == testLabels);
fprintf('  Confirmed test accuracy: %.2f%%\n\n', accuracy*100);

%% ----- IDENTIFY THE FOUR CATEGORIES -----
fprintf('Identifying samples per category...\n');

isMosquitoTrue = (testLabels == 'Mosquito');
isMosquitoPred = (predictions == 'Mosquito');

tpIdx = find( isMosquitoTrue &  isMosquitoPred);
tnIdx = find(~isMosquitoTrue & ~isMosquitoPred);
fnIdx = find( isMosquitoTrue & ~isMosquitoPred);
fpIdx = find(~isMosquitoTrue &  isMosquitoPred);

fprintf('  TP (correct mosquito):   %d\n', length(tpIdx));
fprintf('  TN (correct background): %d\n', length(tnIdx));
fprintf('  FN (missed mosquito):    %d\n', length(fnIdx));
fprintf('  FP (false alarm):        %d\n\n', length(fpIdx));

%% ----- PICK SAMPLES TO VISUALIZE -----
mosquitoScores = scores(:, 2);  % column 2 = Mosquito class

% TPs: highest-confidence correct mosquito predictions
[~, tpOrder] = sort(mosquitoScores(tpIdx), 'descend');
tpPick = tpIdx(tpOrder(1:min(samplesPerCategory, length(tpIdx))));

% TNs: most-confidently background (lowest mosquito score)
[~, tnOrder] = sort(mosquitoScores(tnIdx), 'ascend');
tnPick = tnIdx(tnOrder(1:min(samplesPerCategory, length(tnIdx))));

% FNs: most confidently wrong (lowest mosquito score on actual mosquitoes)
if ~isempty(fnIdx)
    [~, fnOrder] = sort(mosquitoScores(fnIdx), 'ascend');
    fnPick = fnIdx(fnOrder(1:min(samplesPerCategory, length(fnIdx))));
else
    fnPick = [];
end

% FPs: most confidently wrong (highest mosquito score on actual backgrounds)
if ~isempty(fpIdx)
    [~, fpOrder] = sort(mosquitoScores(fpIdx), 'descend');
    fpPick = fpIdx(fpOrder(1:min(samplesPerCategory, length(fpIdx))));
else
    fpPick = [];
end

%% ----- PER-CATEGORY GRAD-CAM FIGURES -----
categories = {
    'TruePositive',  'Correctly Classified Mosquito',    tpPick;
    'TrueNegative',  'Correctly Classified Background',  tnPick;
    'FalseNegative', 'Missed Mosquito (False Negative)', fnPick;
    'FalsePositive', 'False Alarm (False Positive)',     fpPick
};

for c = 1:size(categories, 1)
    catName  = categories{c, 1};
    catTitle = categories{c, 2};
    catIdx   = categories{c, 3};
    
    fprintf('Running Grad-CAM for: %s (%d samples)\n', catTitle, length(catIdx));
    
    if isempty(catIdx)
        fprintf('  (No samples in this category, skipping)\n\n');
        continue;
    end
    
    nSamples = length(catIdx);
    nCols = min(4, nSamples);
    nRows = ceil(nSamples / nCols) * 2;
    
    fig = figure('Position', [50, 50, 350*nCols, 250*ceil(nSamples/nCols)*2], ...
                 'Name', catTitle, 'Color', 'white');
    
    for s = 1:nSamples
        idx = catIdx(s);
        sampleMel = testImages(:, :, 1, idx);
        trueLabel = char(testLabels(idx));
        predLabel = char(predictions(idx));
        confidence = mosquitoScores(idx);
        
        try
            scoreMap = gradCAM(net, sampleMel, predictions(idx));
        catch ME
            fprintf('  Warning: gradCAM failed on sample %d: %s\n', idx, ME.message);
            continue;
        end
        
        col = mod(s-1, nCols) + 1;
        rowGroup = floor((s-1) / nCols);
        
        % Top: mel spectrogram
        subplot(nRows, nCols, rowGroup*2*nCols + col);
        imagesc(sampleMel);
        axis xy;
        colormap(gca, 'gray');
        title(sprintf('Sample %d\nTrue: %s | Pred: %s (%.2f)', ...
              idx, trueLabel, predLabel, confidence), 'FontSize', 9);
        ylabel('Frequency (Hz)');
        addFrequencyTicks(nMels, fMin, fMax);
        
        % Bottom: Grad-CAM overlay
        subplot(nRows, nCols, rowGroup*2*nCols + nCols + col);
        imagesc(sampleMel);
        axis xy;
        colormap(gca, 'gray');
        hold on;
        h = imagesc(scoreMap);
        set(h, 'AlphaData', 0.5);
        colormap(gca, 'jet');
        title('Grad-CAM heatmap', 'FontSize', 9);
        xlabel('Time frame');
        ylabel('Frequency (Hz)');
        addFrequencyTicks(nMels, fMin, fMax);
        hold off;
    end
    
    sgtitle(sprintf('%s — Model 1 v3 (1s)', catTitle), 'FontSize', 13, 'FontWeight', 'bold');
    
    savePath = fullfile(outputDir, sprintf('GradCAM_%s.png', catName));
    exportgraphics(fig, savePath, 'Resolution', 200);
    fprintf('  Saved: %s\n\n', savePath);
end

%% ----- AVERAGE ATTENTION HEATMAP (THE KEY FIGURE) -----
fprintf('Computing average attention maps per category...\n');
fprintf('(This figure is what tells us if the model uses real features.)\n\n');

avgFig = figure('Position', [50, 50, 1400, 400], 'Color', 'white');

for c = 1:size(categories, 1)
    catName  = categories{c, 1};
    catTitle = categories{c, 2};
    catIdx   = categories{c, 3};
    
    if isempty(catIdx)
        continue;
    end
    
    nForAvg = min(samplesForAvg, length(catIdx));
    
    if strcmp(catName, 'TruePositive')
        [~, ord] = sort(mosquitoScores(catIdx), 'descend');
        avgIdx = catIdx(ord(1:nForAvg));
    elseif strcmp(catName, 'TrueNegative')
        [~, ord] = sort(mosquitoScores(catIdx), 'ascend');
        avgIdx = catIdx(ord(1:nForAvg));
    else
        avgIdx = catIdx(1:nForAvg);
    end
    
    avgMap = zeros(inputSize(1), inputSize(2));
    successCount = 0;
    
    for s = 1:length(avgIdx)
        try
            scoreMap = gradCAM(net, testImages(:,:,1,avgIdx(s)), predictions(avgIdx(s)));
            scoreMap = (scoreMap - min(scoreMap(:))) / (max(scoreMap(:)) - min(scoreMap(:)) + eps);
            avgMap = avgMap + scoreMap;
            successCount = successCount + 1;
        catch
            continue;
        end
    end
    avgMap = avgMap / max(successCount, 1);
    
    subplot(1, 4, c);
    imagesc(avgMap);
    axis xy;
    colormap(gca, 'jet');
    colorbar;
    title(sprintf('%s\n(avg of %d)', catTitle, successCount), 'FontSize', 10);
    xlabel('Time frame');
    ylabel('Frequency (Hz)');
    addFrequencyTicks(nMels, fMin, fMax);
end

sgtitle('Average Grad-CAM Attention by Category — Model 1 v3 (1s, HPF 300 Hz)', ...
        'FontSize', 13, 'FontWeight', 'bold');

avgPath = fullfile(outputDir, 'GradCAM_AverageAttention.png');
exportgraphics(avgFig, avgPath, 'Resolution', 200);
fprintf('  Saved: %s\n\n', avgPath);

%% ----- DONE -----
fprintf('========================================\n');
fprintf('GRAD-CAM ANALYSIS COMPLETE\n');
fprintf('========================================\n');
fprintf('Test accuracy: %.2f%%\n\n', accuracy*100);
fprintf('Results saved to: %s\n\n', outputDir);
fprintf('What to look for in GradCAM_AverageAttention.png:\n');
fprintf('  TRUE POSITIVES panel (this is the most important):\n');
fprintf('    Hot in 400-1500 Hz band  -> wingbeat harmonics. REAL features.\n');
fprintf('    Hot below 200 Hz         -> original shortcut returned.\n');
fprintf('    Hot vertical stripe(s)   -> looping artifact shortcut.\n');
fprintf('    Hot at time-edges        -> padding artifact (unlikely in v3).\n\n');
fprintf('  FALSE POSITIVES panel:\n');
fprintf('    Mid-frequency attention  -> acoustically plausible mistakes.\n');
fprintf('    Vertical stripe          -> loop boundary shortcut.\n');
fprintf('========================================\n');


%% ============================================================
% HELPER: Add frequency tick labels (mel scale -> Hz)
%% ============================================================
function addFrequencyTicks(nMels, fMin, fMax)
    tickHz = [200, 500, 1000, 2000, 4000, 8000];
    tickHz = tickHz(tickHz <= fMax);
    mel = @(f) 2595 * log10(1 + f/700);
    melMin = mel(fMin);
    melMax = mel(fMax);
    tickBins = (mel(tickHz) - melMin) / (melMax - melMin) * (nMels - 1) + 1;
    yticks(tickBins);
    yticklabels(arrayfun(@(x) sprintf('%d', x), tickHz, 'UniformOutput', false));
end


%% ============================================================
% HELPER: Load binary dataset from two separate folders
%   - mosquitoDir contains species subfolders (everything is "Mosquito")
%   - backgroundDir contains .mat files directly (everything is "Background")
%% ============================================================
function [images, labels] = loadBinaryFromTwoFolders(mosquitoDir, backgroundDir, inputSize)
    
    mosqFiles = dir(fullfile(mosquitoDir, '**', '*.mat'));
    nMosq = length(mosqFiles);
    
    bgFiles = dir(fullfile(backgroundDir, '*.mat'));
    nBg = length(bgFiles);
    
    N = nMosq + nBg;
    fprintf('  Mosquito files: %d\n', nMosq);
    fprintf('  Background files: %d\n', nBg);
    fprintf('  Total: %d\n', N);
    
    images = zeros([inputSize(1:2), 1, N], 'single');
    labels = categorical(zeros(N, 1), [0, 1], {'Background', 'Mosquito'});
    validIdx = true(N, 1);
    
    progressInterval = max(1, floor(N/20));
    
    % Load mosquitoes
    for i = 1:nMosq
        fullPath = fullfile(mosqFiles(i).folder, mosqFiles(i).name);
        labels(i) = 'Mosquito';
        
        try
            data = load(fullPath);
            mel = data.melSpec;
            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(i) = false;
                continue;
            end
            mel = max(0, min(1, mel));
            if size(mel, 1) ~= inputSize(1) || size(mel, 2) ~= inputSize(2)
                mel = imresize(mel, inputSize(1:2));
            end
            if ndims(mel) > 2
                mel = mel(:, :, 1);
            end
            images(:, :, 1, i) = single(mel);
        catch
            validIdx(i) = false;
        end
        
        if mod(i, progressInterval) == 0
            fprintf('    Progress: %d/%d (%.1f%%)\n', i, N, 100*i/N);
        end
    end
    
    % Load backgrounds
    for j = 1:nBg
        idx = nMosq + j;
        fullPath = fullfile(bgFiles(j).folder, bgFiles(j).name);
        labels(idx) = 'Background';
        
        try
            data = load(fullPath);
            mel = data.melSpec;
            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(idx) = false;
                continue;
            end
            mel = max(0, min(1, mel));
            if size(mel, 1) ~= inputSize(1) || size(mel, 2) ~= inputSize(2)
                mel = imresize(mel, inputSize(1:2));
            end
            if ndims(mel) > 2
                mel = mel(:, :, 1);
            end
            images(:, :, 1, idx) = single(mel);
        catch
            validIdx(idx) = false;
        end
        
        if mod(idx, progressInterval) == 0
            fprintf('    Progress: %d/%d (%.1f%%)\n', idx, N, 100*idx/N);
        end
    end
    
    images = images(:, :, :, validIdx);
    labels = labels(validIdx);
    
    nBgFinal  = sum(labels == 'Background');
    nMosFinal = sum(labels == 'Mosquito');
    
    fprintf('  Loaded successfully: %d\n', length(labels));
    fprintf('    Background: %d (%.1f%%)\n', nBgFinal, 100*nBgFinal/length(labels));
    fprintf('    Mosquito:   %d (%.1f%%)\n\n', nMosFinal, 100*nMosFinal/length(labels));
end