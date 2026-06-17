clear; clc; close all;

%% ============================================================
% HUMBUG BACKGROUND PREPROCESSING v3 — 1-SECOND SEGMENTS
%
% Each 2-second Humbug background clip is split into 2
% non-overlapping 1-second segments. Mostly-silent segments
% are discarded.
%
% Splits are made by recording ID (all segments from a recording
% land in the same split) so segments from the same session
% can never end up in both train and test.
%
% Subsamples to roughly match mosquito training counts so Model 1
% sees a balanced binary task. Output is added to the v3 ESC-50
% background folder.
%% ============================================================

%% PATHS
sourceDir = 'G:\My Drive\Thesis_Dataset\background_segments';
outputDir = 'G:\My Drive\Thesis_Dataset\ESC-50_hpf300_background_v3';

%% TARGET COUNTS
% These will need adjustment based on the actual mosquito v3 counts.
% You'll see those after the mosquito script runs. For now, generous
% upper bounds — script will subsample down to whatever fits.
targetCounts = struct( ...
    'train', 25000, ...
    'val',    4000, ...
    'test',   5000);

%% AUDIO PARAMETERS
targetFs    = 16000;
clipLength  = 1;
hpfCutoff   = 300;
hpfOrder    = 4;

silenceThresh  = 0.005;
maxSilenceFrac = 0.40;

splitRatios = struct('train', 0.70, 'val', 0.15, 'test', 0.15);

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

%% RNG
rng(42);

%% FILTER
[hpfB, hpfA] = butter(hpfOrder, hpfCutoff / (targetFs/2), 'high');

%% INDEX FILES BY RECORDING ID
fprintf('========================================\n');
fprintf('HUMBUG BG v3 (1s, HPF %d Hz)\n', hpfCutoff);
fprintf('========================================\n');
fprintf('Source: %s\n', sourceDir);
fprintf('Output: %s (added to existing ESC-50 v3)\n\n', outputDir);

allFiles = dir(fullfile(sourceDir, '*.wav'));
nFiles = length(allFiles);
fprintf('Found %d files. Indexing by recording ID...\n', nFiles);

recordingMap = containers.Map();
for i = 1:nFiles
    name = allFiles(i).name;
    tok = regexp(name, '^(.+)_seg\d+\.wav$', 'tokens', 'once');
    if isempty(tok)
        [~, recID, ~] = fileparts(name);
    else
        recID = tok{1};
    end
    
    if isKey(recordingMap, recID)
        recordingMap(recID) = [recordingMap(recID); i];
    else
        recordingMap(recID) = i;
    end
end

recordingIDs = keys(recordingMap);
nRecordings  = length(recordingIDs);
fprintf('Grouped into %d unique recording IDs\n\n', nRecordings);

%% SPLIT BY ID
shuffledIdx = randperm(nRecordings);
nTrain = floor(splitRatios.train * nRecordings);
nVal   = floor(splitRatios.val   * nRecordings);

trainIDs = recordingIDs(shuffledIdx(1:nTrain));
valIDs   = recordingIDs(shuffledIdx(nTrain+1 : nTrain+nVal));
testIDs  = recordingIDs(shuffledIdx(nTrain+nVal+1 : end));

fprintf('Recordings: train=%d, val=%d, test=%d\n\n', ...
        length(trainIDs), length(valIDs), length(testIDs));

%% COLLECT INDICES PER SPLIT
function fileIdx = collectFiles(idList, recMap)
    fileIdx = [];
    for k = 1:length(idList)
        fileIdx = [fileIdx; recMap(idList{k})];
    end
end

trainFileIdx = collectFiles(trainIDs, recordingMap);
valFileIdx   = collectFiles(valIDs,   recordingMap);
testFileIdx  = collectFiles(testIDs,  recordingMap);

% At 2 segments per file, available segments per split
fprintf('Available 1s segments per split (max, before silence filter):\n');
fprintf('  train: ~%d\n', length(trainFileIdx) * 2);
fprintf('  val:   ~%d\n', length(valFileIdx) * 2);
fprintf('  test:  ~%d\n\n', length(testFileIdx) * 2);

%% SUBSAMPLE FILES — each file produces up to 2 segments, so
%% subsample input file count to roughly match desired output count
function picked = subsample(fileIdx, targetCount)
    targetFiles = ceil(targetCount / 2);  % each file = 2 segments
    if length(fileIdx) <= targetFiles
        picked = fileIdx;
    else
        picked = fileIdx(randperm(length(fileIdx), targetFiles));
    end
end

trainPicked = subsample(trainFileIdx, targetCounts.train);
valPicked   = subsample(valFileIdx,   targetCounts.val);
testPicked  = subsample(testFileIdx,  targetCounts.test);

%% PROCESS
splits = {'train', trainPicked; 'val', valPicked; 'test', testPicked};
totalProcessed = 0;
totalDiscarded = 0;
totalFailed = 0;

for s = 1:size(splits, 1)
    splitName = splits{s, 1};
    fileIdx   = splits{s, 2};
    nThis     = length(fileIdx);
    
    fprintf('--- Split: %s (%d files) ---\n', splitName, nThis);
    
    bgOutDir = fullfile(outputDir, splitName, 'background');
    if ~exist(bgOutDir, 'dir')
        mkdir(bgOutDir);
    end
    
    splitProc = 0;
    splitDisc = 0;
    splitFail = 0;
    progressInterval = max(1, floor(nThis/10));
    
    for k = 1:nThis
        i = fileIdx(k);
        wavPath = fullfile(allFiles(i).folder, allFiles(i).name);
        
        try
            [audio, fs] = audioread(wavPath);
            if size(audio, 2) > 1
                audio = mean(audio, 2);
            end
            audio = audio(:);
            
            if fs ~= targetFs
                audio = resample(audio, targetFs, fs);
            end
            
            audio = filter(hpfB, hpfA, audio);
            
            nSegments = floor(length(audio) / targetSamples);
            [~, baseName, ~] = fileparts(allFiles(i).name);
            
            for seg = 1:nSegments
                outName = sprintf('humbug_%s_seg%d.mat', baseName, seg);
                outPath = fullfile(bgOutDir, outName);
                
                % RESUME: skip if already processed
                if exist(outPath, 'file')
                    splitProc = splitProc + 1;
                    continue;
                end
                
                startIdx = (seg-1)*targetSamples + 1;
                endIdx   = startIdx + targetSamples - 1;
                segment  = audio(startIdx:endIdx);
                
                silentFrac = sum(abs(segment) < silenceThresh) / length(segment);
                if silentFrac > maxSilenceFrac
                    splitDisc = splitDisc + 1;
                    continue;
                end
                
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
                
                save(outPath, 'melSpec');
                splitProc = splitProc + 1;
            end
            
        catch ME
            splitFail = splitFail + 1;
            if splitFail <= 3
                fprintf('  ERROR on %s: %s\n', allFiles(i).name, ME.message);
            end
        end
        
        if mod(k, progressInterval) == 0
            fprintf('  %d/%d (%.0f%%)\n', k, nThis, 100*k/nThis);
        end
    end
    
    fprintf('  -> processed=%d, discarded=%d, failed=%d\n\n', ...
            splitProc, splitDisc, splitFail);
    
    totalProcessed = totalProcessed + splitProc;
    totalDiscarded = totalDiscarded + splitDisc;
    totalFailed    = totalFailed    + splitFail;
end

fprintf('========================================\n');
fprintf('HUMBUG BG v3 COMPLETE\n');
fprintf('========================================\n');
fprintf('Total processed: %d\n', totalProcessed);
fprintf('Discarded:       %d\n', totalDiscarded);
fprintf('Failed:          %d\n', totalFailed);
fprintf('Output: %s\n', outputDir);

fprintf('\nFinal background totals (ESC-50 v3 + Humbug v3):\n');
for s = 1:size(splits, 1)
    splitName = splits{s, 1};
    bgDir = fullfile(outputDir, splitName, 'background');
    nMat = length(dir(fullfile(bgDir, '*.mat')));
    fprintf('  %s: %d .mat files\n', splitName, nMat);
end
fprintf('========================================\n');