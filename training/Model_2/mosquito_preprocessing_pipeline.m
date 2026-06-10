clc; clear; close all;

% ============================================================
% MOSQUITO PREPROCESSING PIPELINE
% MSU-IIT Thesis Project
%
% Stage 1: Resample and segment raw mosquito WAV files (non-overlapping)
% Stage 2: Convert WAV segments to mel spectrograms (.mat)
% Output:  G:\My Drive\Thesis_Dataset\1vector_borne\
% ============================================================

%% CONFIGURATION

rawAudioDir   = 'G:\My Drive\Thesis_Dataset\6Mosq';
segmentedDir  = 'G:\My Drive\Thesis_Dataset\6segmented_2sec';
outputBaseDir = 'G:\My Drive\Thesis_Dataset\1vector_borne';

runStage1 = false; % set to true if raw WAV files need segmenting
runStage2 = true;  % set to true to convert WAV segments to mel spectrograms

targetSampleRate = 16000;
segmentLength    = 2.0;
segmentSamples   = targetSampleRate * segmentLength;

nMels         = 128;
fftSize       = 1024;
hopLength     = 256;
overlapLength = fftSize - hopLength;

trainRatio = 0.70;
valRatio   = 0.15;

fprintf('============================================================\n');
fprintf('  MOSQUITO PREPROCESSING PIPELINE\n');
fprintf('  MSU-IIT Thesis\n');
fprintf('============================================================\n\n');

%% STAGE 1: RESAMPLE, SEGMENT, AND SPLIT

if runStage1

    fprintf('STAGE 1: Resampling, segmenting, and splitting\n\n');

    trainDir = fullfile(segmentedDir, 'train');
    valDir   = fullfile(segmentedDir, 'val');
    testDir  = fullfile(segmentedDir, 'test');

    for d = {trainDir, valDir, testDir}
        if ~exist(d{1}, 'dir'); mkdir(d{1}); end
    end

    speciesFolders = dir(rawAudioDir);
    speciesFolders = speciesFolders([speciesFolders.isdir]);
    speciesFolders = speciesFolders(~ismember({speciesFolders.name}, {'.','..'}));

    for speciesIdx = 1:length(speciesFolders)
        speciesName = speciesFolders(speciesIdx).name;
        speciesPath = fullfile(rawAudioDir, speciesName);

        fprintf('  Species %d/%d: %s\n', speciesIdx, length(speciesFolders), speciesName);

        trainSpeciesDir = fullfile(trainDir, speciesName);
        valSpeciesDir   = fullfile(valDir,   speciesName);
        testSpeciesDir  = fullfile(testDir,  speciesName);

        for d = {trainSpeciesDir, valSpeciesDir, testSpeciesDir}
            if ~exist(d{1}, 'dir'); mkdir(d{1}); end
        end

        sessionFolders = dir(speciesPath);
        sessionFolders = sessionFolders([sessionFolders.isdir]);
        sessionFolders = sessionFolders(~ismember({sessionFolders.name}, {'.','..'}));

        if isempty(sessionFolders)
            fprintf('    No session folders found, skipping\n');
            continue;
        end

        rng(42);
        shuffleIdx     = randperm(length(sessionFolders));
        sessionFolders = sessionFolders(shuffleIdx);

        nSessions = length(sessionFolders);
        nTrain    = floor(nSessions * trainRatio);
        nVal      = floor(nSessions * valRatio);

        trainSessions = sessionFolders(1:nTrain);
        valSessions   = sessionFolders(nTrain+1:nTrain+nVal);
        testSessions  = sessionFolders(nTrain+nVal+1:end);

        fprintf('    Sessions — Train: %d  Val: %d  Test: %d\n', ...
            length(trainSessions), length(valSessions), length(testSessions));

        segmentSessions(trainSessions, speciesPath, trainSpeciesDir, targetSampleRate, segmentSamples);
        segmentSessions(valSessions,   speciesPath, valSpeciesDir,   targetSampleRate, segmentSamples);
        segmentSessions(testSessions,  speciesPath, testSpeciesDir,  targetSampleRate, segmentSamples);
    end

    fprintf('\nStage 1 complete\n\n');

else
    fprintf('Stage 1 skipped\n\n');
end

%% STAGE 2: CONVERT WAV SEGMENTS TO MEL SPECTROGRAMS

if runStage2

    fprintf('STAGE 2: Converting WAV segments to mel spectrograms\n\n');

    splits = {'train', 'val', 'test'};

    for sp = 1:length(splits)
        splitName      = splits{sp};
        inputSplitDir  = fullfile(segmentedDir,  splitName);
        outputSplitDir = fullfile(outputBaseDir, splitName);

        if ~exist(outputSplitDir, 'dir'); mkdir(outputSplitDir); end

        fprintf('  Processing %s split...\n', splitName);

        speciesFolders = dir(inputSplitDir);
        speciesFolders = speciesFolders([speciesFolders.isdir]);
        speciesFolders = speciesFolders(~ismember({speciesFolders.name}, {'.','..'}));

        for s = 1:length(speciesFolders)
            speciesName      = speciesFolders(s).name;
            speciesInputDir  = fullfile(inputSplitDir,  speciesName);
            speciesOutputDir = fullfile(outputSplitDir, speciesName);

            if ~exist(speciesOutputDir, 'dir'); mkdir(speciesOutputDir); end

            wavFiles = dir(fullfile(speciesInputDir, '*.wav'));
            fprintf('    %s: %d files\n', speciesName, length(wavFiles));

            converted = 0; skipped = 0; failed = 0;

            for f = 1:length(wavFiles)
                [~, baseName]  = fileparts(wavFiles(f).name);
                inputFile      = fullfile(speciesInputDir,  wavFiles(f).name);
                outputFile     = fullfile(speciesOutputDir, [baseName '_mel.mat']);

                if isfile(outputFile)
                    skipped = skipped + 1;
                    continue;
                end

                try
                    melSpec = computeMelSpectrogram(inputFile, targetSampleRate, ...
                        segmentSamples, nMels, fftSize, overlapLength);

                    tempFile = [outputFile '.tmp'];
                    save(tempFile, 'melSpec', '-v7.3');
                    movefile(tempFile, outputFile);
                    converted = converted + 1;
                catch
                    fprintf('      Warning: skipping %s\n', wavFiles(f).name);
                    failed = failed + 1;
                end

                if mod(f, 500) == 0
                    fprintf('      Progress: %d/%d\n', f, length(wavFiles));
                end
            end

            fprintf('      Converted: %d  Skipped: %d  Failed: %d\n', converted, skipped, failed);
        end
    end

    fprintf('\nStage 2 complete\n\n');

else
    fprintf('Stage 2 skipped\n\n');
end

fprintf('============================================================\n');
fprintf('PIPELINE COMPLETE\n');
fprintf('Output: %s\n', outputBaseDir);
fprintf('============================================================\n');

%% HELPER FUNCTIONS

function segmentSessions(sessions, speciesPath, outputDir, targetSampleRate, segmentSamples)
    for s = 1:length(sessions)
        sessionName = sessions(s).name;
        sessionPath = fullfile(speciesPath, sessionName);
        files       = dir(fullfile(sessionPath, '*.wav'));

        for f = 1:length(files)
            [~, baseName] = fileparts(files(f).name);
            inputFile     = fullfile(sessionPath, files(f).name);

            try
                [audio, fs] = audioread(inputFile);

                if size(audio, 2) > 1
                    audio = mean(audio, 2);
                end

                if fs ~= targetSampleRate
                    audio = resample(audio, targetSampleRate, fs);
                end

                audio = double(audio(:));

                numSegments = floor(length(audio) / segmentSamples);

                for segIdx = 1:numSegments
                    startIdx = (segIdx - 1) * segmentSamples + 1;
                    endIdx   = startIdx + segmentSamples - 1;
                    if endIdx > length(audio); break; end

                    segment = audio(startIdx:endIdx);
                    outName = sprintf('%s_%s_seg%03d.wav', sessionName, baseName, segIdx);
                    outFile = fullfile(outputDir, outName);

                    if ~isfile(outFile)
                        audiowrite(outFile, segment, targetSampleRate);
                    end
                end
            catch
                continue;
            end
        end
    end
end

function melNorm = computeMelSpectrogram(filePath, sampleRate, samples, nMels, fftSize, overlapLength)
    [audio, fs] = audioread(filePath);
    audio = resample(audio, sampleRate, fs);
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
        'OverlapLength', overlapLength, ...
        'NumBands',      nMels);

    melDB   = pow2db(mel + eps);
    melNorm = (melDB + 80) / 80;
end
