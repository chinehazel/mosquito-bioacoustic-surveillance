clear; clc; close all;

%% ============================================================
% MODEL 2: MOSQUITO SPECIES CLASSIFIER (6-CLASS CNN-LSTM)
% Purpose: Identify which mosquito species from wingbeat audio
% Classes: 6 mosquito species (background excluded)
%% ============================================================

%% SETTINGS
trainDir = 'G:\My Drive\Thesis_Dataset\1vector_borne\train';
valDir   = 'G:\My Drive\Thesis_Dataset\1vector_borne\val';
testDir  = 'G:\My Drive\Thesis_Dataset\1vector_borne\test';

inputSize = [128 122 1];

% Stage 1: CNN training
cnnEpochs = 30;
cnnBatchSize = 32;
cnnLearningRate = 1e-3;

% Stage 2: LSTM training - FIXED WINDOWING
numTimeWindows = 4;      % Changed from 10 to 4
windowOverlap = 0.0;     % Changed from 0.5 to 0.0 (no overlap)
lstmEpochs = 50;
lstmBatchSize = 16;
lstmLearningRate = 1e-3;

%% ============================================================
% LOAD DATA (EXCLUDE BACKGROUND)
%% ============================================================
fprintf('========================================\n');
fprintf('MODEL 2: SPECIES CLASSIFIER (6 CLASSES)\n');
fprintf('========================================\n\n');

fprintf('Loading dataset files (excluding background)...\n');

trainFiles = dir(fullfile(trainDir,'**','*.mat'));
valFiles   = dir(fullfile(valDir,'**','*.mat'));
testFiles  = dir(fullfile(testDir,'**','*.mat'));

[trainPaths, trainLabels] = getValidFilesAndLabels(trainFiles);
[valPaths,   valLabels]   = getValidFilesAndLabels(valFiles);
[testPaths,  testLabels]  = getValidFilesAndLabels(testFiles);

trainLabelsCat = categorical(trainLabels);
valLabelsCat   = categorical(valLabels);
testLabelsCat  = categorical(testLabels);

classes = categories(trainLabelsCat);
numClasses = numel(classes);

fprintf('\n========================================\n');
fprintf('DATASET SUMMARY\n');
fprintf('========================================\n');
fprintf('  Classes: %d\n', numClasses);
fprintf('  Train samples: %d\n', numel(trainPaths));
fprintf('  Val samples: %d\n', numel(valPaths));
fprintf('  Test samples: %d\n', numel(testPaths));
fprintf('========================================\n\n');

% Display per-class distribution
fprintf('Per-Class Distribution (Train):\n');
for i = 1:numClasses
    count = sum(trainLabelsCat == classes{i});
    fprintf('  %-25s: %d samples\n', classes{i}, count);
end
fprintf('\n');

%% ============================================================
% STAGE 1: TRAIN CNN FEATURE EXTRACTOR
%% ============================================================
fprintf('========================================\n');
fprintf('STAGE 1: TRAINING CNN FEATURE EXTRACTOR\n');
fprintf('========================================\n\n');

% Build CNN for classification
cnnLayers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    globalAveragePooling2dLayer('Name', 'gap')
    
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

% Load and prepare data for CNN training
fprintf('Loading mel spectrograms for CNN training...\n');
[trainImages, trainLabelsForCNN] = loadAllMelSpectrograms(trainPaths, trainLabelsCat, inputSize);
[valImages, valLabelsForCNN] = loadAllMelSpectrograms(valPaths, valLabelsCat, inputSize);

fprintf('  Loaded %d training images\n', size(trainImages, 4));
fprintf('  Loaded %d validation images\n\n', size(valImages, 4));

% CNN training options
cnnOptions = trainingOptions('adam', ...
    'MaxEpochs', cnnEpochs, ...
    'MiniBatchSize', cnnBatchSize, ...
    'InitialLearnRate', cnnLearningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valImages, valLabelsForCNN}, ...
    'ValidationFrequency', floor(size(trainImages, 4) / cnnBatchSize), ...
    'ValidationPatience', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% Train CNN
fprintf('Training CNN classifier...\n\n');
tic;
trainedCNN = trainNetwork(trainImages, trainLabelsForCNN, cnnLayers, cnnOptions);
cnnTrainingTime = toc;

fprintf('\nCNN training completed in %.2f minutes\n', cnnTrainingTime/60);

% Evaluate CNN alone (baseline)
fprintf('\n========================================\n');
fprintf('CNN-ONLY BASELINE EVALUATION\n');
fprintf('========================================\n\n');

[testImages, testLabelsForCNN] = loadAllMelSpectrograms(testPaths, testLabelsCat, inputSize);
cnnPredictions = classify(trainedCNN, testImages);
cnnAccuracy = mean(cnnPredictions == testLabelsForCNN);

fprintf('CNN-only Test Accuracy: %.2f%%\n', cnnAccuracy*100);

% CNN confusion matrix
figure('Position', [100, 100, 800, 700]);
cm = confusionchart(testLabelsForCNN, cnnPredictions);
cm.Title = sprintf('CNN-Only Baseline (Acc: %.2f%%)', cnnAccuracy*100);
cm.FontSize = 10;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ============================================================
% STAGE 2: EXTRACT CNN FEATURES FOR LSTM
%% ============================================================
fprintf('\n========================================\n');
fprintf('STAGE 2: EXTRACTING CNN FEATURES FOR LSTM\n');
fprintf('========================================\n\n');

% Convert trained CNN to feature extractor
featureNet = layerGraph(trainedCNN);
featureNet = removeLayers(featureNet, {'dropout', 'fc', 'softmax', 'classification'});
featureNet = dlnetwork(featureNet);

fprintf('Feature extraction settings:\n');
fprintf('  Layer: gap (256 features)\n');
fprintf('  Time windows: %d\n', numTimeWindows);
fprintf('  Window overlap: %.1f%%\n\n', windowOverlap*100);

% Extract features from time windows
fprintf('Extracting features: TRAIN\n');
[trainFeatures, trainLabelsFiltered] = extractCNNLSTMFeatures(trainPaths, trainLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

fprintf('Extracting features: VAL\n');
[valFeatures, valLabelsFiltered] = extractCNNLSTMFeatures(valPaths, valLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

fprintf('Extracting features: TEST\n');
[testFeatures, testLabelsFiltered] = extractCNNLSTMFeatures(testPaths, testLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

% Validate features
fprintf('\n========================================\n');
fprintf('FEATURE EXTRACTION SUMMARY\n');
fprintf('========================================\n');
fprintf('  Train sequences: %d\n', numel(trainFeatures));
fprintf('  Val sequences: %d\n', numel(valFeatures));
fprintf('  Test sequences: %d\n', numel(testFeatures));

if ~isempty(trainFeatures)
    fprintf('  Feature dimensions: [%d × %d] (features × time)\n', ...
        size(trainFeatures{1}, 1), size(trainFeatures{1}, 2));
    
    % Check feature quality
    allFeats = cat(2, trainFeatures{:});
    featMean = mean(allFeats(:));
    featStd = std(allFeats(:));
    fprintf('  Feature mean: %.4f\n', featMean);
    fprintf('  Feature std: %.4f\n', featStd);
    
    if featStd < 0.01
        warning('Low feature variance! Features may not be discriminative.');
    end
end
fprintf('========================================\n\n');

%% ============================================================
% STAGE 3: TRAIN LSTM ON CNN FEATURES
%% ============================================================
fprintf('========================================\n');
fprintf('STAGE 3: TRAINING LSTM CLASSIFIER\n');
fprintf('========================================\n\n');

lstmLayers = [
    sequenceInputLayer(256, 'Name', 'sequence_input')
    
    bilstmLayer(128, 'OutputMode', 'last', 'Name', 'bilstm1')
    dropoutLayer(0.5, 'Name', 'dropout1')
    
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.3, 'Name', 'dropout2')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

lstmOptions = trainingOptions('adam', ...
    'MaxEpochs', lstmEpochs, ...
    'MiniBatchSize', lstmBatchSize, ...
    'InitialLearnRate', lstmLearningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valFeatures, valLabelsFiltered}, ...
    'ValidationFrequency', floor(numel(trainFeatures) / lstmBatchSize), ...
    'ValidationPatience', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('Training LSTM on CNN features...\n\n');
tic;
trainedLSTM = trainNetwork(trainFeatures, trainLabelsFiltered, lstmLayers, lstmOptions);
lstmTrainingTime = toc;

fprintf('\nLSTM training completed in %.2f minutes\n', lstmTrainingTime/60);

%% ============================================================
% EVALUATE CNN-LSTM
%% ============================================================
fprintf('\n========================================\n');
fprintf('FINAL EVALUATION: CNN-LSTM\n');
fprintf('========================================\n\n');

predLabels = classify(trainedLSTM, testFeatures);
trueLabels = testLabelsFiltered;

accuracy = mean(predLabels == trueLabels);

fprintf('=== RESULTS COMPARISON ===\n');
fprintf('CNN-only Accuracy:  %.2f%%\n', cnnAccuracy*100);
fprintf('CNN-LSTM Accuracy:  %.2f%%\n', accuracy*100);
fprintf('Improvement:        %+.2f%%\n\n', (accuracy - cnnAccuracy)*100);

%% ============================================================
% DETAILED METRICS
%% ============================================================
[confMat, order] = confusionmat(trueLabels, predLabels);

precision = diag(confMat) ./ sum(confMat, 1)';
recall = diag(confMat) ./ sum(confMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1Score(isnan(f1Score)) = 0;

fprintf('========================================\n');
fprintf('PER-CLASS METRICS (CNN-LSTM)\n');
fprintf('========================================\n');
fprintf('%-25s %10s %10s %10s %10s\n', 'Class', 'Samples', 'Precision', 'Recall', 'F1-Score');
fprintf('%-25s %10s %10s %10s %10s\n', '-----', '-------', '---------', '------', '--------');

for i = 1:numel(classes)
    numSamples = sum(trueLabels == classes{i});
    fprintf('%-25s %10d %9.2f%% %9.2f%% %9.2f%%\n', ...
        classes{i}, numSamples, precision(i)*100, recall(i)*100, f1Score(i)*100);
end

fprintf('\n%-25s %10s %9.2f%% %9.2f%% %9.2f%%\n', ...
    'Macro Average', '', mean(precision)*100, mean(recall)*100, mean(f1Score)*100);
fprintf('========================================\n\n');

%% ============================================================
% CONFUSION MATRIX VISUALIZATION
%% ============================================================
figure('Position', [100, 100, 900, 800]);
cm = confusionchart(trueLabels, predLabels);
cm.Title = sprintf('CNN-LSTM Confusion Matrix (Acc: %.2f%%)', accuracy*100);
cm.FontSize = 10;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ============================================================
% SAVE MODELS
%% ============================================================
modelFilename = sprintf('model2_species_classifier_acc%.2f.mat', accuracy*100);
save(modelFilename, 'trainedCNN', 'trainedLSTM', 'featureNet', 'classes', ...
     'accuracy', 'cnnAccuracy', 'inputSize', 'numTimeWindows', 'windowOverlap');
fprintf('Model 2 saved as: %s\n', modelFilename);

fprintf('\n========================================\n');
fprintf('TWO-STAGE TRAINING COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total Training Time: %.2f minutes\n', (cnnTrainingTime + lstmTrainingTime)/60);
fprintf('CNN Baseline: %.2f%%\n', cnnAccuracy*100);
fprintf('CNN-LSTM Final: %.2f%%\n', accuracy*100);
fprintf('========================================\n');

%% ============================================================
% SUPPORT FUNCTIONS
%% ============================================================

function [paths, labels] = getValidFilesAndLabels(fileStruct)
    paths = {};
    labels = {};
    skipped = 0;
    backgroundSkipped = 0;
    
    for k = 1:numel(fileStruct)
        try
            fullPath = fullfile(fileStruct(k).folder, fileStruct(k).name);
            
            % Get class name from folder
            parts = split(fileStruct(k).folder, filesep);
            className = parts{end};
            
            % SKIP BACKGROUND CLASS FOR MODEL 2
            if strcmpi(className, 'background')
                backgroundSkipped = backgroundSkipped + 1;
                continue;
            end
            
            data = load(fullPath);
            if isfield(data,'melSpec') && ~isempty(data.melSpec)
                paths{end+1,1} = fullPath;
                labels{end+1,1} = className;
            else
                skipped = skipped + 1;
            end
        catch
            skipped = skipped + 1;
        end
    end
    
    if skipped > 0
        fprintf('  Warning: Skipped %d invalid files\n', skipped);
    end
    if backgroundSkipped > 0
        fprintf('  Note: Excluded %d background files (Model 2 trains on mosquitoes only)\n', backgroundSkipped);
    end
end

function [images, labels] = loadAllMelSpectrograms(filePaths, fileLabels, inputSize)
    N = numel(filePaths);
    images = zeros([inputSize(1:2), 1, N], 'single');
    labels = fileLabels;
    validIdx = true(N, 1);
    
    fprintf('  Loading %d mel spectrograms...\n', N);
    progressInterval = max(1, floor(N/20));
    
    for i = 1:N
        if mod(i, progressInterval) == 0 || i == N
            fprintf('    Progress: %d/%d (%.1f%%)\n', i, N, 100*i/N);
        end
        
        try
            loadedData = load(filePaths{i});
            mel = loadedData.melSpec;
            
            % Validate
            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(i) = false;
                continue;
            end
            
            % Normalize
            mel = rescale(mel);
            
            % Resize if needed
            if size(mel, 1) ~= inputSize(1) || size(mel, 2) ~= inputSize(2)
                mel = imresize(mel, inputSize(1:2));
            end
            
            % Ensure 2D
            if ndims(mel) > 2
                mel = mel(:, :, 1);
            end
            
            images(:, :, 1, i) = single(mel);
            
        catch ME
            warning('Error loading file %d: %s', i, ME.message);
            validIdx(i) = false;
        end
    end
    
    % Remove invalid samples
    images = images(:, :, :, validIdx);
    labels = labels(validIdx);
    
    fprintf('  Successfully loaded %d/%d images\n\n', sum(validIdx), N);
end

function [features, labels] = extractCNNLSTMFeatures(filePaths, fileLabels, ...
    featureNet, inputSize, numWindows, overlap)
    
    N = numel(filePaths);
    features = cell(N, 1);
    labels = fileLabels;
    
    fprintf('  Processing %d files...\n', N);
    progressInterval = max(1, floor(N/20));
    
    validIdx = true(N, 1);
    
    for i = 1:N
        if mod(i, progressInterval) == 0 || i == N
            fprintf('    Progress: %d/%d (%.1f%%)\n', i, N, 100*i/N);
        end
        
        try
            windows = loadMelWindows(filePaths{i}, inputSize, numWindows, overlap);
            
            if isempty(windows)
                validIdx(i) = false;
                continue;
            end
            
            numWin = numel(windows);
            featSeq = zeros(256, numWin, 'single');
            
            for w = 1:numWin
                img = dlarray(single(windows{w}), 'SSC');
                feat = forward(featureNet, img);
                featSeq(:, w) = gather(extractdata(feat));
            end
            
            features{i} = featSeq;
            
        catch ME
            warning('Error processing file %d: %s', i, ME.message);
            validIdx(i) = false;
        end
    end
    
    features = features(validIdx);
    labels = labels(validIdx);
    
    fprintf('  Extracted features from %d/%d files\n\n', sum(validIdx), N);
end

function windows = loadMelWindows(filepath, inputSize, numWindows, overlap)
    try
        data = load(filepath);
        mel = data.melSpec;
    catch
        windows = {};
        return;
    end
    
    if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
        windows = {};
        return;
    end
    
    mel = rescale(mel);
    [freqBins, timeBins] = size(mel);  % Should be [128, 122]
    
    % FIXED WINDOWING CALCULATION
    if overlap == 0
        % Non-overlapping windows
        windowSize = floor(timeBins / numWindows);
        stepSize = windowSize;
    else
        % Overlapping windows
        windowSize = ceil(timeBins / (numWindows - (numWindows-1) * overlap));
        stepSize = ceil(windowSize * (1 - overlap));
    end
    
    % Ensure minimum window size
    if windowSize < 10
        windows = {};
        return;
    end
    
    windows = cell(numWindows, 1);
    actualWindows = 0;
    
    for w = 1:numWindows
        startIdx = (w-1) * stepSize + 1;
        endIdx = min(startIdx + windowSize - 1, timeBins);
        
        if startIdx > timeBins
            break;
        end
        
        % Extract window
        window = mel(:, startIdx:endIdx);
        
        % Pad if last window is shorter
        if size(window, 2) < windowSize
            padSize = windowSize - size(window, 2);
            window = [window, repmat(window(:, end), 1, padSize)];
        end
        
        % Resize to CNN input size
        window = imresize(window, inputSize(1:2));
        
        % Ensure 2D
        if ndims(window) > 2
            window = window(:, :, 1);
        end
        
        actualWindows = actualWindows + 1;
        windows{actualWindows} = window;
    end
    
    % Trim to actual windows created
    windows = windows(1:actualWindows);
    
    % Validate we got expected number of windows
    if actualWindows < numWindows * 0.8
        windows = {};
    end
end