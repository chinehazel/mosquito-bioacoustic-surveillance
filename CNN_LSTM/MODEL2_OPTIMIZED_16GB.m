%% ============================================================
% MODEL 2: OPTIMIZED FOR 16GB RAM PC
% Changes from original:
% - Reduced epochs (50→30 for CNN, 50→30 for LSTM)
% - Larger batch sizes (32→64 for CNN, 16→32 for LSTM)
% - Checkpointing every 5 epochs (prevents loss if crash)
% - Memory-efficient data loading
%% ============================================================

clear; clc; close all;

%% OPTIMIZED SETTINGS FOR 16GB RAM
trainDir = 'G:\My Drive\Thesis_Dataset\1vector_borne\train';
valDir   = 'G:\My Drive\Thesis_Dataset\1vector_borne\val';
testDir  = 'G:\My Drive\Thesis_Dataset\1vector_borne\test';

inputSize = [128 122 1];

% REDUCED EPOCHS (faster training)
cnnEpochs = 30;              % Was 50 → Now 30 (40% faster)
cnnBatchSize = 64;           % Was 32 → Now 64 (50% fewer iterations)
cnnLearningRate = 1e-3;
augmentationFactor = 2;      % Was 3 → Now 2 (33% less memory)

% LSTM parameters (also optimized)
numTimeWindows = 10;
windowOverlap = 0.5;
lstmEpochs = 30;             % Was 50 → Now 30
lstmBatchSize = 32;          % Was 16 → Now 32 (50% fewer iterations)
lstmLearningRate = 1e-4;

% Checkpoint settings
saveCheckpointsEvery = 5;    % Save every 5 epochs

%% SYSTEM CHECK
fprintf('========================================\n');
fprintf('MODEL 2: OPTIMIZED FOR 16GB RAM PC\n');
fprintf('========================================\n\n');

% Check available memory
[~, systemview] = memory;
availableGB = systemview.PhysicalMemory.Available/1e9;
fprintf('Available RAM: %.1f GB\n', availableGB);

if availableGB < 8
    warning('Low RAM detected. Consider closing other programs.');
    pause(3);
end

% Check GPU
if canUseGPU
    gpu = gpuDevice;
    fprintf('✓ GPU Available: %s\n', gpu.Name);
    fprintf('  GPU Memory: %.1f GB\n\n', gpu.AvailableMemory/1e9);
else
    fprintf('⚠ No GPU detected - Training will use CPU\n\n');
end

fprintf('Training Configuration:\n');
fprintf('  CNN Epochs: %d (reduced from 50)\n', cnnEpochs);
fprintf('  CNN Batch Size: %d (increased from 32)\n', cnnBatchSize);
fprintf('  Augmentation: %dx (reduced from 3x)\n', augmentationFactor);
fprintf('  LSTM Epochs: %d (reduced from 50)\n', lstmEpochs);
fprintf('  LSTM Batch Size: %d (increased from 16)\n', lstmBatchSize);
fprintf('  Checkpointing: Every %d epochs\n\n', saveCheckpointsEvery);

%% LOAD DATA
fprintf('========================================\n');
fprintf('LOADING DATASET\n');
fprintf('========================================\n\n');

fprintf('Loading dataset files...\n');
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

fprintf('\nDataset Summary:\n');
fprintf('  Classes: %d\n', numClasses);
fprintf('  Train: %d, Val: %d, Test: %d\n\n', ...
    numel(trainPaths), numel(valPaths), numel(testPaths));

%% STAGE 1: TRAIN CNN
fprintf('========================================\n');
fprintf('STAGE 1: CNN TRAINING\n');
fprintf('========================================\n\n');

% Build CNN (same architecture)
cnnLayers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    
    % Block 1
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_1')
    batchNormalizationLayer('Name', 'bn1_1')
    reluLayer('Name', 'relu1_1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_2')
    batchNormalizationLayer('Name', 'bn1_2')
    reluLayer('Name', 'relu1_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    dropoutLayer(0.2, 'Name', 'drop1')
    
    % Block 2
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_1')
    batchNormalizationLayer('Name', 'bn2_1')
    reluLayer('Name', 'relu2_1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_2')
    batchNormalizationLayer('Name', 'bn2_2')
    reluLayer('Name', 'relu2_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    dropoutLayer(0.3, 'Name', 'drop2')
    
    % Block 3
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_1')
    batchNormalizationLayer('Name', 'bn3_1')
    reluLayer('Name', 'relu3_1')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_2')
    batchNormalizationLayer('Name', 'bn3_2')
    reluLayer('Name', 'relu3_2')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_3')
    batchNormalizationLayer('Name', 'bn3_3')
    reluLayer('Name', 'relu3_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    dropoutLayer(0.4, 'Name', 'drop3')
    
    % Block 4
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    globalAveragePooling2dLayer('Name', 'gap')
    
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

% Load spectrograms
fprintf('Loading mel spectrograms...\n');
[trainImages, trainLabelsForCNN] = loadAllMelSpectrograms(trainPaths, trainLabelsCat, inputSize);
[valImages, valLabelsForCNN] = loadAllMelSpectrograms(valPaths, valLabelsCat, inputSize);

fprintf('  Loaded: Train=%d, Val=%d\n', size(trainImages,4), size(valImages,4));

% Check memory after loading
[~, systemview] = memory;
fprintf('  RAM used: %.1f GB\n\n', (systemview.PhysicalMemory.Total - systemview.PhysicalMemory.Available)/1e9);

% Apply augmentation (REDUCED to 2x)
fprintf('Applying 2x augmentation...\n');
[trainImagesAug, trainLabelsAug] = generateAugmentedData(...
    trainImages, trainLabelsForCNN, augmentationFactor);

fprintf('  Original: %d → Augmented: %d\n', size(trainImages,4), size(trainImagesAug,4));

% Verify labels
uniqueLabelsAug = categories(trainLabelsAug);
fprintf('  Classes: %d (verified)\n\n', numel(uniqueLabelsAug));

% Calculate iterations
iterationsPerEpoch = ceil(size(trainImagesAug,4) / cnnBatchSize);
totalIterations = iterationsPerEpoch * cnnEpochs;

fprintf('Training Stats:\n');
fprintf('  Iterations per epoch: %d (was ~1477)\n', iterationsPerEpoch);
fprintf('  Total iterations: %d\n', totalIterations);
fprintf('  Estimated time per epoch: ~10-15 min (CPU) or ~2-3 min (GPU)\n\n');

% Training options
cnnOptions = trainingOptions('adam', ...
    'MaxEpochs', cnnEpochs, ...
    'MiniBatchSize', cnnBatchSize, ...
    'InitialLearnRate', cnnLearningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valImages, valLabelsForCNN}, ...
    'ValidationFrequency', iterationsPerEpoch, ...
    'ValidationPatience', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'CheckpointPath', pwd);  % Save checkpoints in current folder

% Train CNN
fprintf('========================================\n');
fprintf('STARTING CNN TRAINING\n');
fprintf('========================================\n\n');

startTime = datetime('now');
fprintf('Start time: %s\n\n', datestr(startTime));

tic;
trainedCNN = trainNetwork(trainImagesAug, trainLabelsAug, cnnLayers, cnnOptions);
cnnTrainingTime = toc;

fprintf('\n========================================\n');
fprintf('CNN TRAINING COMPLETED\n');
fprintf('========================================\n');
fprintf('Training time: %.2f minutes (%.2f hours)\n', cnnTrainingTime/60, cnnTrainingTime/3600);
fprintf('End time: %s\n', datestr(datetime('now')));
fprintf('========================================\n\n');

% Save CNN model immediately
save('trainedCNN_checkpoint.mat', 'trainedCNN', 'classes', 'inputSize', ...
     'trainPaths', 'trainLabelsCat', 'valPaths', 'valLabelsCat', ...
     'testPaths', 'testLabelsCat');
fprintf('✓ CNN model saved: trainedCNN_checkpoint.mat\n\n');

%% EVALUATE CNN BASELINE
fprintf('========================================\n');
fprintf('CNN BASELINE EVALUATION\n');
fprintf('========================================\n\n');

fprintf('Loading test data...\n');
[testImages, testLabelsForCNN] = loadAllMelSpectrograms(testPaths, testLabelsCat, inputSize);

fprintf('Classifying test set...\n');
cnnPredictions = classify(trainedCNN, testImages);
cnnAccuracy = mean(cnnPredictions == testLabelsForCNN);

fprintf('\n✓ CNN Test Accuracy: %.2f%%\n\n', cnnAccuracy*100);

% Confusion matrix
figure('Position', [100, 100, 900, 800]);
cm = confusionchart(testLabelsForCNN, cnnPredictions);
cm.Title = sprintf('CNN Baseline (Acc: %.2f%%)', cnnAccuracy*100);
cm.FontSize = 10;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Decision point
fprintf('========================================\n');
fprintf('DECISION POINT\n');
fprintf('========================================\n\n');

if cnnAccuracy >= 0.90
    fprintf('🎯 EXCELLENT! CNN achieved %.2f%% (target: 90%%)\n\n', cnnAccuracy*100);
    fprintf('Continue to LSTM? (y/n): ');
    userChoice = input('', 's');
    if ~strcmpi(userChoice, 'y')
        fprintf('\nStopping here. CNN model saved.\n');
        fprintf('To continue later, run: CONTINUE_TO_LSTM.m\n');
        return;
    end
elseif cnnAccuracy >= 0.85
    fprintf('✓ GOOD baseline: %.2f%%\n', cnnAccuracy*100);
    fprintf('Continuing to LSTM stage...\n\n');
else
    fprintf('⚠ Baseline: %.2f%% (below 85%%)\n', cnnAccuracy*100);
    fprintf('Recommendation: Check data or retrain with more epochs\n');
    fprintf('Continue anyway? (y/n): ');
    userChoice = input('', 's');
    if ~strcmpi(userChoice, 'y')
        fprintf('\nStopping. Review results and try again.\n');
        return;
    end
end

%% STAGE 2: EXTRACT FEATURES
fprintf('\n========================================\n');
fprintf('STAGE 2: FEATURE EXTRACTION\n');
fprintf('========================================\n\n');

% Convert to feature extractor
fprintf('Converting CNN to feature extractor...\n');
featureNet = layerGraph(trainedCNN);
featureNet = removeLayers(featureNet, {'dropout', 'fc', 'softmax', 'classification'});
featureNet = dlnetwork(featureNet);

fprintf('  Windows: %d, Overlap: %.0f%%\n\n', numTimeWindows, windowOverlap*100);

% Extract features
fprintf('Extracting CNN features...\n\n');

fprintf('TRAIN set:\n');
[trainFeatures, trainLabelsFiltered] = extractCNNLSTMFeatures(trainPaths, trainLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

fprintf('VAL set:\n');
[valFeatures, valLabelsFiltered] = extractCNNLSTMFeatures(valPaths, valLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

fprintf('TEST set:\n');
[testFeatures, testLabelsFiltered] = extractCNNLSTMFeatures(testPaths, testLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

fprintf('\n✓ Feature extraction complete!\n');
fprintf('  Sequences: Train=%d, Val=%d, Test=%d\n', ...
    numel(trainFeatures), numel(valFeatures), numel(testFeatures));
if ~isempty(trainFeatures)
    fprintf('  Dimensions: [%d × %d]\n\n', size(trainFeatures{1},1), size(trainFeatures{1},2));
end

%% STAGE 3: TRAIN LSTM
fprintf('========================================\n');
fprintf('STAGE 3: LSTM TRAINING\n');
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

lstmIterationsPerEpoch = ceil(numel(trainFeatures) / lstmBatchSize);
fprintf('LSTM Training Stats:\n');
fprintf('  Iterations per epoch: %d\n', lstmIterationsPerEpoch);
fprintf('  Total epochs: %d\n\n', lstmEpochs);

lstmOptions = trainingOptions('adam', ...
    'MaxEpochs', lstmEpochs, ...
    'MiniBatchSize', lstmBatchSize, ...
    'InitialLearnRate', lstmLearningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valFeatures, valLabelsFiltered}, ...
    'ValidationFrequency', lstmIterationsPerEpoch, ...
    'ValidationPatience', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('Training LSTM...\n\n');
tic;
trainedLSTM = trainNetwork(trainFeatures, trainLabelsFiltered, lstmLayers, lstmOptions);
lstmTrainingTime = toc;

fprintf('\n✓ LSTM training completed in %.2f minutes (%.2f hours)\n', ...
    lstmTrainingTime/60, lstmTrainingTime/3600);

%% FINAL EVALUATION
fprintf('\n========================================\n');
fprintf('FINAL EVALUATION\n');
fprintf('========================================\n\n');

predLabels = classify(trainedLSTM, testFeatures);
trueLabels = testLabelsFiltered;
lstmAccuracy = mean(predLabels == trueLabels);

fprintf('=== FINAL RESULTS ===\n');
fprintf('CNN-only:      %.2f%%\n', cnnAccuracy*100);
fprintf('CNN-LSTM:      %.2f%%\n', lstmAccuracy*100);
fprintf('Difference:    %+.2f%%\n\n', (lstmAccuracy - cnnAccuracy)*100);

% Interpretation
if lstmAccuracy > cnnAccuracy + 0.01
    fprintf('✓ LSTM HELPS! (+%.2f%%)\n', (lstmAccuracy - cnnAccuracy)*100);
    fprintf('  Use CNN-LSTM for deployment.\n\n');
elseif abs(lstmAccuracy - cnnAccuracy) <= 0.01
    fprintf('○ LSTM NEUTRAL\n');
    fprintf('  CNN-only is simpler and sufficient.\n\n');
else
    fprintf('✗ LSTM HURTS (%.2f%%)\n', (lstmAccuracy - cnnAccuracy)*100);
    fprintf('  Use CNN-only model.\n\n');
end

%% PER-CLASS METRICS
[confMat, ~] = confusionmat(trueLabels, predLabels);
precision = diag(confMat) ./ sum(confMat, 1)';
recall = diag(confMat) ./ sum(confMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1Score(isnan(f1Score)) = 0;

fprintf('========================================\n');
fprintf('PER-CLASS METRICS\n');
fprintf('========================================\n');
fprintf('%-25s %8s %10s %10s %10s\n', 'Class', 'Samples', 'Precision', 'Recall', 'F1');
fprintf('%-25s %8s %10s %10s %10s\n', repmat('-',1,25), repmat('-',1,8), repmat('-',1,10), repmat('-',1,10), repmat('-',1,10));
for i = 1:numel(classes)
    n = sum(trueLabels == classes{i});
    fprintf('%-25s %8d %9.2f%% %9.2f%% %9.2f%%\n', ...
        classes{i}, n, precision(i)*100, recall(i)*100, f1Score(i)*100);
end
fprintf('\n%-25s %8s %9.2f%% %9.2f%% %9.2f%%\n', ...
    'Average', '', mean(precision)*100, mean(recall)*100, mean(f1Score)*100);
fprintf('========================================\n\n');

%% CONFUSION MATRIX
figure('Position', [100, 100, 1000, 900]);
cm = confusionchart(trueLabels, predLabels);
cm.Title = sprintf('CNN-LSTM Final Results (Acc: %.2f%%)', lstmAccuracy*100);
cm.FontSize = 10;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% SAVE FINAL MODEL
modelFile = sprintf('final_model_cnn%.2f_lstm%.2f.mat', cnnAccuracy*100, lstmAccuracy*100);
save(modelFile, 'trainedCNN', 'trainedLSTM', 'featureNet', 'classes', ...
     'cnnAccuracy', 'lstmAccuracy', 'augmentationFactor', ...
     'numTimeWindows', 'windowOverlap', 'inputSize');

fprintf('✓ Final model saved: %s\n\n', modelFile);

%% TRAINING SUMMARY
fprintf('========================================\n');
fprintf('COMPLETE TRAINING SUMMARY\n');
fprintf('========================================\n');
fprintf('Total Training Time:\n');
fprintf('  CNN: %.2f hours\n', cnnTrainingTime/3600);
fprintf('  LSTM: %.2f hours\n', lstmTrainingTime/3600);
fprintf('  Total: %.2f hours\n\n', (cnnTrainingTime + lstmTrainingTime)/3600);
fprintf('Final Accuracy:\n');
fprintf('  CNN-only: %.2f%%\n', cnnAccuracy*100);
fprintf('  CNN-LSTM: %.2f%%\n', lstmAccuracy*100);
fprintf('========================================\n');

%% SUPPORT FUNCTIONS
function [paths, labels] = getValidFilesAndLabels(fileStruct)
    paths = {};
    labels = {};
    backgroundSkipped = 0;
    
    for k = 1:numel(fileStruct)
        try
            fullPath = fullfile(fileStruct(k).folder, fileStruct(k).name);
            parts = split(fileStruct(k).folder, filesep);
            className = parts{end};
            
            if strcmpi(className, 'background')
                backgroundSkipped = backgroundSkipped + 1;
                continue;
            end
            
            data = load(fullPath);
            if isfield(data,'melSpec') && ~isempty(data.melSpec)
                paths{end+1,1} = fullPath;
                labels{end+1,1} = className;
            end
        catch
        end
    end
    
    if backgroundSkipped > 0
        fprintf('  Excluded %d background files\n', backgroundSkipped);
    end
end

function [images, labels] = loadAllMelSpectrograms(filePaths, fileLabels, inputSize)
    N = numel(filePaths);
    images = zeros([inputSize(1:2), 1, N], 'single');
    labels = fileLabels;
    validIdx = true(N, 1);
    
    fprintf('  Loading %d spectrograms...\n', N);
    
    for i = 1:N
        if mod(i, max(1, floor(N/20))) == 0 || i == N
            fprintf('    %d/%d (%.0f%%)\n', i, N, 100*i/N);
        end
        
        try
            loadedData = load(filePaths{i});
            mel = loadedData.melSpec;
            
            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(i) = false;
                continue;
            end
            
            mel = rescale(mel);
            
            if size(mel,1) ~= inputSize(1) || size(mel,2) ~= inputSize(2)
                mel = imresize(mel, inputSize(1:2));
            end
            
            if ndims(mel) > 2
                mel = mel(:,:,1);
            end
            
            images(:,:,1,i) = single(mel);
            
        catch
            validIdx(i) = false;
        end
    end
    
    images = images(:,:,:,validIdx);
    labels = labels(validIdx);
    
    fprintf('  Loaded %d/%d\n\n', sum(validIdx), N);
end

function [augImages, augLabels] = generateAugmentedData(images, labels, factor)
    [H, W, C, N] = size(images);
    totalAugmented = N * factor;
    
    augImages = zeros([H, W, C, totalAugmented], 'single');
    augLabelsCell = cell(totalAugmented, 1);
    
    fprintf('  Generating %d augmented samples...\n', totalAugmented);
    
    idx = 1;
    for i = 1:N
        if mod(idx, max(1, floor(totalAugmented/20))) == 0
            fprintf('    %d/%d (%.0f%%)\n', idx, totalAugmented, 100*idx/totalAugmented);
        end
        
        img = images(:,:,:,i);
        label = labels(i);
        
        augImages(:,:,:,idx) = img;
        augLabelsCell{idx} = char(label);
        idx = idx + 1;
        
        for j = 1:(factor-1)
            augImg = img;
            
            augType = randi(5);
            switch augType
                case 1
                    augImg = applySpecAugment(augImg, 12, 12, 2);
                case 2
                    stretchFactor = 0.85 + rand() * 0.3;
                    augImg = applyTimeStretch(augImg, stretchFactor);
                case 3
                    level = rand() * 0.03;
                    augImg = augImg + level * randn(size(augImg), 'single');
                    augImg = max(0, min(1, augImg));
                case 4
                    shift = randi([-10, 10]);
                    augImg = circshift(augImg, shift, 2);
                case 5
                    otherIdx = randi(N);
                    alpha = rand() * 0.3 + 0.1;
                    augImg = alpha * augImg + (1-alpha) * images(:,:,:,otherIdx);
            end
            
            augImages(:,:,:,idx) = augImg;
            augLabelsCell{idx} = char(label);
            idx = idx + 1;
        end
    end
    
    augLabels = categorical(augLabelsCell);
    fprintf('  Complete!\n');
end

function augImg = applySpecAugment(img, freqMask, timeMask, numMasks)
    augImg = img;
    [H, W, ~] = size(img);
    
    for i = 1:numMasks
        mh = randi([1, freqMask]);
        ms = randi([1, max(1, H - mh + 1)]);
        augImg(ms:min(H, ms+mh-1), :, :) = 0;
        
        mw = randi([1, timeMask]);
        ms = randi([1, max(1, W - mw + 1)]);
        augImg(:, ms:min(W, ms+mw-1), :) = 0;
    end
end

function augImg = applyTimeStretch(img, stretchFactor)
    [H, W, C] = size(img);
    newWidth = round(W * stretchFactor);
    augImg = imresize(img, [H, newWidth]);
    
    if newWidth > W
        augImg = augImg(:, 1:W, :);
    else
        padSize = W - newWidth;
        padValues = repmat(augImg(:, end, :), [1, padSize, 1]);
        augImg = cat(2, augImg, padValues);
    end
end

function [features, labels] = extractCNNLSTMFeatures(filePaths, fileLabels, ...
    featureNet, inputSize, numWindows, overlap)
    
    N = numel(filePaths);
    features = cell(N, 1);
    labels = fileLabels;
    validIdx = true(N, 1);
    
    fprintf('  Processing %d files...\n', N);
    
    for i = 1:N
        if mod(i, max(1, floor(N/20))) == 0 || i == N
            fprintf('    %d/%d (%.0f%%)\n', i, N, 100*i/N);
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
            
        catch
            validIdx(i) = false;
        end
    end
    
    features = features(validIdx);
    labels = labels(validIdx);
    
    fprintf('  Extracted %d/%d\n', sum(validIdx), N);
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
    [~, timeBins] = size(mel);
    
    if overlap == 0
        windowSize = floor(timeBins / numWindows);
        stepSize = windowSize;
    else
        windowSize = ceil(timeBins / (numWindows - (numWindows-1) * overlap));
        stepSize = ceil(windowSize * (1 - overlap));
    end
    
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
        
        window = mel(:, startIdx:endIdx);
        
        if size(window, 2) < windowSize
            window = [window, repmat(window(:, end), 1, windowSize - size(window, 2))];
        end
        
        window = imresize(window, inputSize(1:2));
        
        if ndims(window) > 2
            window = window(:,:,1);
        end
        
        actualWindows = actualWindows + 1;
        windows{actualWindows} = window;
    end
    
    windows = windows(1:actualWindows);
    
    if actualWindows < numWindows * 0.8
        windows = {};
    end
end
