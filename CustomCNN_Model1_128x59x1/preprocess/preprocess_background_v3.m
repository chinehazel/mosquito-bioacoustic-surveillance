clear; clc; close all;

%% ============================================================
% ESC-50 BACKGROUND PREPROCESSING v3 — 1-SECOND SEGMENTS, HPF 300 Hz
%
% Each 5-second ESC-50 clip is split into 5 non-overlapping 1-second
% segments. Segments that are mostly silent (>40%) are discarded —
% this removes empty portions of transient sounds (door knocks etc).
%% ============================================================

%% PATHS
sourceDir = 'G:\My Drive\Thesis_Dataset\ESC-50-master\audio';
metaPath  = 'G:\My Drive\Thesis_Dataset\ESC-50-master\meta\esc50.csv';
outputDir = 'G:\My Drive\Thesis_Dataset\ESC-50_hpf300_background_v3';

%% AUDIO PARAMETERS
targetFs    = 16000;
clipLength  = 1;
hpfCutoff   = 300;
hpfOrder    = 4;

silenceThresh  = 0.005;
maxSilenceFrac = 0.40;   % discard segments with more than this fraction silent

excludedCategories = {'sea_waves'};

%% MEL SPECTROGRAM PARAMETERS
nMels   = 128;
nFFT    = 1024;
hopLen  = 256;
winLen  = nFFT;
fMinMel = 0;
fMaxMel = targetFs / 2;

targetSamples = targetFs * clipLength;
nFramesTarget = floor((targetSamples - winLen) / hopLen) + 1;
targetSize    = [nMels, nFramesTarget];

%% FILTER
[hpfB, hpfA] = butter(hpfOrder, hpfCutoff / (targetFs/2), 'high');

%% OUTPUT DIRS
splitNames = {'train', 'val', 'test'};
for s = 1:length(splitNames)
    bgDir = fullfile(outputDir, splitNames{s}, 'background');
    if ~exist(bgDir, 'dir')
        mkdir(bgDir);
    end
end

fprintf('========================================\n');
fprintf('ESC-50 v3 PREPROCESSING (1s, HPF %d Hz)\n', hpfCutoff);
fprintf('========================================\n');
fprintf('Source: %s\n', sourceDir);
fprintf('Output: %s\n', outputDir);
fprintf('Target spectrogram: [%d x %d]\n', targetSize(1), targetSize(2));
fprintf('Excluded categories: %s\n\n', strjoin(excludedCategories, ', '));

%% METADATA
meta = readtable(metaPath);
keep = ~ismember(meta.category, excludedCategories);
meta = meta(keep, :);
fprintf('Loaded metadata: %d entries across %d categories\n\n', ...
        height(meta), length(unique(meta.category)));

%% Split assignment via fold
splitAssignment = repmat({''}, height(meta), 1);
for i = 1:height(meta)
    if meta.fold(i) <= 3
        splitAssignment{i} = 'train';
    elseif meta.fold(i) == 4
        splitAssignment{i} = 'val';
    else
        splitAssignment{i} = 'test';
    end
end

%% PROCESS
splitCounts = struct('train', 0, 'val', 0, 'test', 0);
totalSegments = 0;
totalDiscarded = 0;
totalFailed = 0;

fprintf('Processing %d files...\n\n', height(meta));
progressInterval = max(1, floor(height(meta)/20));

for i = 1:height(meta)
    wavFile  = meta.filename{i};
    category = meta.category{i};
    split    = splitAssignment{i};
    wavPath  = fullfile(sourceDir, wavFile);
    
    if ~exist(wavPath, 'file')
        totalFailed = totalFailed + 1;
        continue;
    end
    
    try
        [audio, fs] = audioread(wavPath);
        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end
        audio = audio(:);
        
        if fs ~= targetFs
            audio = resample(audio, targetFs, fs);
        end
        
        % HPF
        audio = filter(hpfB, hpfA, audio);
        
        % Segment into non-overlapping 1s chunks
        nSegments = floor(length(audio) / targetSamples);
        [~, baseName, ~] = fileparts(wavFile);
        outBgDir = fullfile(outputDir, split, 'background');
        
        for seg = 1:nSegments
            startIdx = (seg - 1) * targetSamples + 1;
            endIdx   = startIdx + targetSamples - 1;
            segment  = audio(startIdx:endIdx);
            
            % Skip if mostly silent
            silentFrac = sum(abs(segment) < silenceThresh) / length(segment);
            if silentFrac > maxSilenceFrac
                totalDiscarded = totalDiscarded + 1;
                continue;
            end
            
            % Mel spectrogram
            mel = melSpectrogram(segment, targetFs, ...
                'NumBands', nMels, ...
                'Window', hann(winLen, 'periodic'), ...
                'OverlapLength', winLen - hopLen, ...
                'FFTLength', nFFT, ...
                'FrequencyRange', [fMinMel fMaxMel]);
            
            melDB = pow2db(mel + eps);
            melSpec = (melDB - min(melDB(:))) / (max(melDB(:)) - min(melDB(:)) + eps);
            
            if size(melSpec, 1) ~= targetSize(1) || size(melSpec, 2) ~= targetSize(2)
                melSpec = imresize(melSpec, targetSize);
            end
            melSpec = single(melSpec);
            
            outName = sprintf('%s_%s_seg%d.mat', category, baseName, seg);
            outPath = fullfile(outBgDir, outName);
            save(outPath, 'melSpec');
            
            totalSegments = totalSegments + 1;
            splitCounts.(split) = splitCounts.(split) + 1;
        end
        
    catch ME
        totalFailed = totalFailed + 1;
        if totalFailed <= 5
            fprintf('  ERROR on %s: %s\n', wavFile, ME.message);
        end
    end
    
    if mod(i, progressInterval) == 0
        fprintf('  %d/%d (%.0f%%) | saved %d, discarded %d\n', ...
                i, height(meta), 100*i/height(meta), totalSegments, totalDiscarded);
    end
end

fprintf('\n========================================\n');
fprintf('ESC-50 v3 PREPROCESSING COMPLETE\n');
fprintf('========================================\n');
fprintf('Total saved: %d\n', totalSegments);
fprintf('  train: %d\n', splitCounts.train);
fprintf('  val:   %d\n', splitCounts.val);
fprintf('  test:  %d\n', splitCounts.test);
fprintf('Discarded (mostly silent): %d\n', totalDiscarded);
fprintf('Failed files: %d\n', totalFailed);
fprintf('Output: %s\n', outputDir);
fprintf('========================================\n');
