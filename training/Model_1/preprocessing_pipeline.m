clc; clear; close all;

% ============================================================
% COMBINED PREPROCESSING PIPELINE
% MSU-IIT Thesis Project
% ============================================================

%% CONFIGURATION

esc50Dir         = 'G:\My Drive\Thesis_Dataset\ESC-50-master';
rawSegmentsDir   = 'G:\My Drive\Thesis_Dataset\background_segments';

segmentedWavDir  = 'G:\My Drive\Thesis_Dataset\background_ESC50';
melOutputDir     = 'G:\My Drive\Thesis_Dataset\mel_output1\background';
splitBaseDir     = 'G:\My Drive\Thesis_Dataset\split_mosquitos1';
outputBaseDir    = 'G:\My Drive\Thesis_Dataset\1vector_borne';

targetSampleRate = 16000;
segmentLength    = 2.0;
segmentSamples   = targetSampleRate * segmentLength;

nMels     = 128;
fftSize   = 1024;
hopLength = 256;

trainRatio = 0.70;
valRatio   = 0.15;
testRatio  = 0.15;

targetTrain = 15757;
targetVal   = 3417;
targetTest  = 4130;

removeClasses = {'sea_waves'};

fprintf('============================================================\n');
fprintf('  COMBINED PREPROCESSING PIPELINE\n');
fprintf('  MSU-IIT Thesis\n');
fprintf('============================================================\n\n');

%% STAGE 1: RESAMPLE AND SEGMENT RAW WAV FILES

fprintf('STAGE 1: Resampling and segmenting raw WAV files\n\n');

if ~exist(segmentedWavDir, 'dir')
    mkdir(segmentedWavDir);
end

metaFile = fullfile(esc50Dir, 'esc50.csv');
if ~exist(metaFile, 'file')
    error('ESC-50 metadata not found: %s', metaFile);
end

metadata = readtable(metaFile);
fprintf('Loaded %d files from ESC-50 metadata\n', height(metadata));

categories = lower(strrep(metadata.category, ' ', '_'));
keepIdx = true(height(metadata), 1);
for i = 1:height(metadata)
    for j = 1:length(removeClasses)
        if contains(lower(categories{i}), removeClasses{j})
            keepIdx(i) = false;
            break;
        end
    end
end
filteredMetadata = metadata(keepIdx, :);
fprintf('Keeping %d files after filtering\n\n', height(filteredMetadata));

audioDir     = fullfile(esc50Dir, 'audio');
segmentCount = 0;
errorCount   = 0;

for i = 1:height(filteredMetadata)
    filename  = filteredMetadata.filename{i};
    inputFile = fullfile(audioDir, filename);

    categoryName = lower(strrep(strrep(strrep( ...
        filteredMetadata.category{i}, ' ', '_'), '(', ''), ')', ''));

    try
        [audio, fs] = audioread(inputFile);

        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end

        if fs ~= targetSampleRate
            audio = resample(audio, targetSampleRate, fs);
        end

        numSegments = floor(length(audio) / segmentSamples);
        for seg = 1:numSegments
            startIdx = (seg - 1) * segmentSamples + 1;
            endIdx   = startIdx + segmentSamples - 1;
            if endIdx > length(audio); break; end

            segment = audio(startIdx:endIdx);
            [~, baseName, ~] = fileparts(filename);
            outName = sprintf('%s_%s_seg%d.wav', categoryName, baseName, seg);
            audiowrite(fullfile(segmentedWavDir, outName), segment, targetSampleRate);
            segmentCount = segmentCount + 1;
        end

    catch ME
        fprintf('  Warning: skipping %s\n', filename);
        errorCount = errorCount + 1;
    end

    if mod(i, 100) == 0
        fprintf('  Progress: %d/%d files — %d segments\n', ...
            i, height(filteredMetadata), segmentCount);
    end
end

fprintf('\nStage 1 complete: %d segments, %d errors\n\n', segmentCount, errorCount);

%% STAGE 2: CONVERT WAV SEGMENTS TO MEL SPECTROGRAMS

fprintf('STAGE 2: Converting WAV segments to mel spectrograms\n\n');

if ~exist(melOutputDir, 'dir')
    mkdir(melOutputDir);
end

inputDirs = {segmentedWavDir, rawSegmentsDir};
melCount = 0; melSkipped = 0; melFailed = 0;

for d = 1:length(inputDirs)
    currentDir = inputDirs{d};
    if ~exist(currentDir, 'dir')
        fprintf('  Skipping missing directory: %s\n', currentDir);
        continue;
    end

    files = dir(fullfile(currentDir, '*.wav'));
    fprintf('  Processing %d files from: %s\n', length(files), currentDir);

    for i = 1:length(files)
        inputFile  = fullfile(currentDir, files(i).name);
        [~, name]  = fileparts(files(i).name);
        outputFile = fullfile(melOutputDir, [name '_mel.mat']);

        if isfile(outputFile)
            melSkipped = melSkipped + 1;
            continue;
        end

        try
            melSpec = computeMelSpectrogram(inputFile, targetSampleRate, ...
                segmentSamples, nMels, fftSize, hopLength);
            save(outputFile, 'melSpec', '-v7.3');
            melCount = melCount + 1;
        catch ME
            fprintf('  Warning: skipping %s\n', files(i).name);
            melFailed = melFailed + 1;
        end

        if mod(i, 100) == 0
            fprintf('  Progress: %d/%d\n', i, length(files));
        end
    end
end

fprintf('\nStage 2 complete: %d converted, %d skipped, %d failed\n\n', ...
    melCount, melSkipped, melFailed);

%% STAGE 3: SESSION-AWARE SPLIT (70/15/15)

fprintf('STAGE 3: Session-aware train/val/test split\n\n');

trainDir = fullfile(splitBaseDir, 'train', 'background');
valDir   = fullfile(splitBaseDir, 'val',   'background');
testDir  = fullfile(splitBaseDir, 'test',  'background');

for d = {trainDir, valDir, testDir}
    if ~exist(d{1}, 'dir'); mkdir(d{1}); end
end

allMelFiles     = dir(fullfile(melOutputDir, '*_mel.mat'));
sourceFiles     = struct();
sourceFileNames = {};

for i = 1:length(allMelFiles)
    filename = allMelFiles(i).name;
    baseName = strrep(filename, '_mel.mat', '');
    segIdx   = regexp(baseName, '_seg\d+$');

    if ~isempty(segIdx)
        sourceName = baseName(1:segIdx-1);
    else
        sourceName = baseName;
    end

    validName = matlab.lang.makeValidName(sourceName);
    if ~isfield(sourceFiles, validName)
        sourceFiles.(validName) = {};
        sourceFileNames{end+1}  = sourceName;
    end
    sourceFiles.(validName){end+1} = filename;
end

numSources = length(sourceFileNames);
fprintf('Grouped into %d unique source files\n', numSources);

rng(42);
shuffleIdx      = randperm(numSources);
sourceFileNames = sourceFileNames(shuffleIdx);

nTrain = floor(numSources * trainRatio);
nVal   = floor(numSources * valRatio);

trainSources = sourceFileNames(1:nTrain);
valSources   = sourceFileNames(nTrain+1:nTrain+nVal);
testSources  = sourceFileNames(nTrain+nVal+1:end);

fprintf('Source split — Train: %d  Val: %d  Test: %d\n\n', ...
    length(trainSources), length(valSources), length(testSources));

splitTrainCount = copyBySource(trainSources, sourceFiles, melOutputDir, trainDir);
splitValCount   = copyBySource(valSources,   sourceFiles, melOutputDir, valDir);
splitTestCount  = copyBySource(testSources,  sourceFiles, melOutputDir, testDir);

fprintf('\nStage 3 complete — Train: %d  Val: %d  Test: %d\n\n', ...
    splitTrainCount, splitValCount, splitTestCount);

%% STAGE 4: DISTRIBUTE INTO FINAL 1vector_borne DIRECTORY

fprintf('STAGE 4: Final distribution to 1vector_borne\n\n');

finalTrainDir = fullfile(outputBaseDir, 'train', 'background');
finalValDir   = fullfile(outputBaseDir, 'val',   'background');
finalTestDir  = fullfile(outputBaseDir, 'test',  'background');

for d = {finalTrainDir, finalValDir, finalTestDir}
    if ~exist(d{1}, 'dir'); mkdir(d{1}); end
end

distributeFiles(trainDir, finalTrainDir, targetTrain);
distributeFiles(valDir,   finalValDir,   targetVal);
distributeFiles(testDir,  finalTestDir,  targetTest);

fprintf('\n============================================================\n');
fprintf('PIPELINE COMPLETE\n');
fprintf('============================================================\n\n');

finalTrain = length(dir(fullfile(finalTrainDir, '*.mat')));
finalVal   = length(dir(fullfile(finalValDir,   '*.mat')));
finalTest  = length(dir(fullfile(finalTestDir,  '*.mat')));

fprintf('Output: %s\n\n', outputBaseDir);
fprintf('  train/background: %d files\n', finalTrain);
fprintf('  val/background:   %d files\n', finalVal);
fprintf('  test/background:  %d files\n', finalTest);
fprintf('\n============================================================\n');

%% HELPER FUNCTIONS

function melNorm = computeMelSpectrogram(filePath, sampleRate, samples, nMels, fftSize, hopLength)
    [audio, sr] = audioread(filePath);
    audio = resample(audio, sampleRate, sr);
    audio = double(audio(:));

    if length(audio) < samples
        audio = [audio; zeros(samples - length(audio), 1)];
    else
        audio = audio(1:samples);
    end

    win = hamming(fftSize, 'periodic');
    mel = melSpectrogram(audio, sampleRate, ...
        'FFTLength',     fftSize, ...
        'Window',        win, ...
        'OverlapLength', fftSize - hopLength, ...
        'NumBands',      nMels);

    melDB   = pow2db(mel + eps);
    melNorm = (melDB + 80) / 80;
end

function count = copyBySource(sourceList, sourceFiles, inputDir, outputDir)
    count = 0;
    for i = 1:length(sourceList)
        validName = matlab.lang.makeValidName(sourceList{i});
        segments  = sourceFiles.(validName);
        for j = 1:length(segments)
            src  = fullfile(inputDir,  segments{j});
            dest = fullfile(outputDir, segments{j});
            if ~isfile(dest)
                copyfile(src, dest);
            end
            count = count + 1;
        end
    end
end

function distributeFiles(sourceDir, destDir, targetCount)
    allFiles = dir(fullfile(sourceDir, '*.mat'));
    total    = length(allFiles);

    if total > targetCount
        rng(42);
        idx      = randperm(total, targetCount);
        allFiles = allFiles(idx);
    end

    copied = 0; skipped = 0;

    for i = 1:length(allFiles)
        src  = fullfile(sourceDir, allFiles(i).name);
        dest = fullfile(destDir,   allFiles(i).name);

        if isfile(dest)
            skipped = skipped + 1;
            continue;
        end

        try
            data = load(src);
            save(dest, '-struct', 'data', '-v7.3');
            copied = copied + 1;
        catch
            fprintf('  Warning: failed to copy %s\n', allFiles(i).name);
        end
    end

    fprintf('  %s: copied %d, skipped %d\n', destDir, copied, skipped);
end
