% =============================================================
% STEP 2: CONVERT 2-SECOND WAV SEGMENTS TO MEL SPECTROGRAMS
% Input: Already segmented 2-second WAV files from Step 1
% Output: Normalized mel spectrograms in .mat format
% =============================================================
clc; clear; close all;

% ===================== CONFIGURATION =====================
% --- Paths ---
inputBaseDir = 'G:\My Drive\Thesis_Dataset\6segmented_2sec';
outputBaseDir = 'G:\My Drive\Thesis_Dataset\1vector_borne';

% Input directories (already split by Step 1)
trainInputDir = fullfile(inputBaseDir, 'train');
valInputDir = fullfile(inputBaseDir, 'val');
testInputDir = fullfile(inputBaseDir, 'test');

% Output directories
trainOutputDir = fullfile(outputBaseDir, 'train');
valOutputDir = fullfile(outputBaseDir, 'val');
testOutputDir = fullfile(outputBaseDir, 'test');

% Create output directories
if ~exist(trainOutputDir, 'dir')
    mkdir(trainOutputDir);
end
if ~exist(valOutputDir, 'dir')
    mkdir(valOutputDir);
end
if ~exist(testOutputDir, 'dir')
    mkdir(testOutputDir);
end

% --- Mel Spectrogram Parameters ---
sampleRate = 16000;
duration = 2.0;
samples = sampleRate * duration;

nMels = 128;
fftSize = 1024;
hopLength = 256;
overlapLength = fftSize - hopLength;

% ===================== MAIN PROCESSING =====================
fprintf('========================================\n');
fprintf('MEL SPECTROGRAM GENERATION (STEP 2)\n');
fprintf('========================================\n');
fprintf('Input:  %s\n', inputBaseDir);
fprintf('Output: %s\n', outputBaseDir);
fprintf('Sample rate: %d Hz\n', sampleRate);
fprintf('Mel bands: %d\n', nMels);
fprintf('FFT size: %d, Hop: %d\n', fftSize, hopLength);
fprintf('Expected shape: [128 x 122]\n');
fprintf('Variable name: melSpec\n');
fprintf('========================================\n\n');

totalStats = struct('trainFiles', 0, 'valFiles', 0, 'testFiles', 0);
tic;

% === PROCESS TRAIN SET ===
fprintf('[TRAIN] Converting to mel spectrograms...\n');
trainCount = processSplitToMel(trainInputDir, trainOutputDir, ...
    sampleRate, samples, nMels, fftSize, overlapLength);
fprintf('SUCCESS: Train - %d spectrograms created\n\n', trainCount);
totalStats.trainFiles = trainCount;

% === PROCESS VAL SET ===
fprintf('[VAL] Converting to mel spectrograms...\n');
valCount = processSplitToMel(valInputDir, valOutputDir, ...
    sampleRate, samples, nMels, fftSize, overlapLength);
fprintf('SUCCESS: Val - %d spectrograms created\n\n', valCount);
totalStats.valFiles = valCount;

% === PROCESS TEST SET ===
fprintf('[TEST] Converting to mel spectrograms...\n');
testCount = processSplitToMel(testInputDir, testOutputDir, ...
    sampleRate, samples, nMels, fftSize, overlapLength);
fprintf('SUCCESS: Test - %d spectrograms created\n\n', testCount);
totalStats.testFiles = testCount;

elapsedTime = toc;

fprintf('========================================\n');
fprintf('SUCCESS: MEL SPECTROGRAM GENERATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total spectrograms:\n');
fprintf('  Train: %d\n', totalStats.trainFiles);
fprintf('  Val:   %d\n', totalStats.valFiles);
fprintf('  Test:  %d\n', totalStats.testFiles);
fprintf('  TOTAL: %d\n', totalStats.trainFiles + totalStats.valFiles + totalStats.testFiles);
fprintf('\nTime: %.2f minutes (%.2f hours)\n', elapsedTime/60, elapsedTime/3600);

if (totalStats.trainFiles + totalStats.valFiles + totalStats.testFiles) > 0
    avgTime = elapsedTime / (totalStats.trainFiles + totalStats.valFiles + totalStats.testFiles);
    fprintf('Average: %.3f sec/file\n', avgTime);
end
fprintf('========================================\n');

% Save metadata
metadata = struct();
metadata.params = struct('sampleRate', sampleRate, 'duration', duration, ...
    'nMels', nMels, 'fftSize', fftSize, 'hopLength', hopLength);
metadata.stats = totalStats;
metadata.timestamp = datetime('now');

metadataFile = fullfile(outputBaseDir, 'mel_metadata.mat');
save(metadataFile, 'metadata', '-v7.3');
fprintf('\nSUCCESS: Metadata saved to: %s\n', metadataFile);

% =============================================================
% HELPER FUNCTION: Process one split (train/val/test)
% =============================================================
function totalProcessed = processSplitToMel(inputSplitDir, outputSplitDir, ...
    sampleRate, samples, nMels, fftSize, overlapLength)
    
    totalProcessed = 0;
    
    % Find all species folders in this split
    speciesFolders = dir(inputSplitDir);
    speciesFolders = speciesFolders([speciesFolders.isdir]);
    speciesFolders = speciesFolders(~ismember({speciesFolders.name}, {'.','..'}));
    
    if isempty(speciesFolders)
        warning('No species folders found in: %s', inputSplitDir);
        return;
    end
    
    fprintf('  Found %d species folders\n', length(speciesFolders));
    
    for s = 1:length(speciesFolders)
        speciesName = speciesFolders(s).name;
        speciesInputDir = fullfile(inputSplitDir, speciesName);
        speciesOutputDir = fullfile(outputSplitDir, speciesName);
        
        % Create output directory for this species
        if ~exist(speciesOutputDir, 'dir')
            mkdir(speciesOutputDir);
        end
        
        % Find all WAV files
        wavFiles = dir(fullfile(speciesInputDir, '*.wav'));
        
        if isempty(wavFiles)
            fprintf('  WARNING: %s - No WAV files found\n', speciesName);
            continue;
        end
        
        fprintf('  %s: Processing %d files...\n', speciesName, length(wavFiles));
        
        speciesProcessed = 0;
        speciesSkipped = 0;
        speciesErrors = 0;
        
        for f = 1:length(wavFiles)
            [~, baseName, ~] = fileparts(wavFiles(f).name);
            inputFile = fullfile(speciesInputDir, wavFiles(f).name);
            outputFile = fullfile(speciesOutputDir, [baseName, '_mel.mat']);
            
            % Skip if already exists
            if exist(outputFile, 'file')
                speciesProcessed = speciesProcessed + 1;
                totalProcessed = totalProcessed + 1;
                speciesSkipped = speciesSkipped + 1;
                continue;
            end
            
            try
                % === READ AUDIO ===
                [audio, fs] = audioread(inputFile);
                
                % Resample if needed
                if fs ~= sampleRate
                    audio = resample(audio, sampleRate, fs);
                end
                
                % Convert to mono
                if size(audio, 2) > 1
                    audio = mean(audio, 2);
                end
                
                audio = audio(:);
                
                % Ensure exact length
                if length(audio) < samples
                    audio = [audio; zeros(samples - length(audio), 1)];
                elseif length(audio) > samples
                    audio = audio(1:samples);
                end
                
                % === GENERATE MEL SPECTROGRAM ===
                win = hamming(fftSize, 'periodic');
                
                mel = melSpectrogram(audio, sampleRate, ...
                    'FFTLength', fftSize, ...
                    'Window', win, ...
                    'OverlapLength', overlapLength, ...
                    'NumBands', nMels);
                
                % Convert to dB and normalize
                melDB = pow2db(mel + eps);
                melSpec = (melDB + 80) / 80;
                
                % Validate
                if isempty(melSpec) || any(isnan(melSpec(:))) || any(isinf(melSpec(:)))
                    warning('Invalid spectrogram for: %s', wavFiles(f).name);
                    speciesErrors = speciesErrors + 1;
                    continue;
                end
                
                % === SAVE ===
                tempFile = [outputFile, '.tmp'];
                save(tempFile, 'melSpec', '-v7.3');
                movefile(tempFile, outputFile);
                
                speciesProcessed = speciesProcessed + 1;
                totalProcessed = totalProcessed + 1;
                
            catch ME
                fprintf('    ERROR processing %s: %s\n', wavFiles(f).name, ME.message);
                speciesErrors = speciesErrors + 1;
                continue;
            end
            
            % Progress update every 500 files
            if mod(f, 500) == 0
                fprintf('    Progress: %d/%d (%.1f%%)\n', f, length(wavFiles), 100*f/length(wavFiles));
            end
        end
        
        % Summary for this species
        if speciesSkipped > 0
            fprintf('    SUCCESS: %s - %d processed (%d already existed, %d errors)\n', ...
                speciesName, speciesProcessed, speciesSkipped, speciesErrors);
        else
            fprintf('    SUCCESS: %s - %d/%d files processed (%d errors)\n', ...
                speciesName, speciesProcessed, length(wavFiles), speciesErrors);
        end
    end
    
    fprintf('\n');
end