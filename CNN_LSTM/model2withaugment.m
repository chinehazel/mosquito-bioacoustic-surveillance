%% ============================================================
% OPTION B: PROPER CNN-LSTM WITH AUGMENTATION
% The RIGHT way: Better windowing + augmentation
% Combines: model2_cnn_with_augmentation.m + fixed LSTM approach
%% ============================================================

clear; clc; close all;

%% SETTINGS
trainDir = 'G:\My Drive\Thesis_Dataset\1vector_borne\train';
valDir   = 'G:\My Drive\Thesis_Dataset\1vector_borne\val';
testDir  = 'G:\My Drive\Thesis_Dataset\1vector_borne\test';

inputSize = [128 122 1];

% Stage 1: CNN training (WITH AUGMENTATION)
cnnEpochs = 50;
cnnBatchSize = 32;
cnnLearningRate = 1e-3;
augmentationFactor = 3;  % 3x augmented data

% Stage 2: LSTM training (IMPROVED WINDOWING)
numTimeWindows = 10;     % Changed from 4 → 10 (more time steps!)
windowOverlap = 0.5;     % Changed from 0.0 → 0.5 (smoother transitions!)
lstmEpochs = 50;
lstmBatchSize = 16;
lstmLearningRate = 1e-4;  % Lower LR for LSTM

%% ============================================================
% LOAD DATA
%% ============================================================
fprintf('========================================\n');
fprintf('MODEL 2: AUGMENTED CNN + PROPER LSTM\n');
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

%% ============================================================
% STAGE 1: TRAIN CNN WITH AUGMENTATION
%% ============================================================
fprintf('========================================\n');
fprintf('STAGE 1: TRAINING CNN (WITH AUGMENTATION)\n');
fprintf('========================================\n\n');

% Build improved CNN
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

fprintf('  Loaded %d training, %d validation images\n', ...
    size(trainImages, 4), size(valImages, 4));

% Apply augmentation to TRAINING data only
fprintf('\nApplying augmentation (factor=%d)...\n', augmentationFactor);
[trainImagesAug, trainLabelsAug] = generateAugmentedData(...
    trainImages, trainLabelsForCNN, augmentationFactor);

fprintf('  Original: %d → Augmented: %d (%.1fx)\n\n', ...
    size(trainImages, 4), size(trainImagesAug, 4), ...
    size(trainImagesAug, 4) / size(trainImages, 4));

% CNN training options
cnnOptions = trainingOptions('adam', ...
    'MaxEpochs', cnnEpochs, ...
    'MiniBatchSize', cnnBatchSize, ...
    'InitialLearnRate', cnnLearningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 15, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valImages, valLabelsForCNN}, ...
    'ValidationFrequency', floor(size(trainImagesAug, 4) / cnnBatchSize), ...
    'ValidationPatience', 15, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% Train CNN on AUGMENTED data
fprintf('Training CNN on augmented data...\n\n');
tic;
trainedCNN = trainNetwork(trainImagesAug, trainLabelsAug, cnnLayers, cnnOptions);
cnnTrainingTime = toc;

fprintf('\nCNN training completed in %.2f minutes\n', cnnTrainingTime/60);

% Evaluate CNN alone (baseline)
fprintf('\n========================================\n');
fprintf('CNN-ONLY BASELINE (WITH AUGMENTATION)\n');
fprintf('========================================\n\n');

[testImages, testLabelsForCNN] = loadAllMelSpectrograms(testPaths, testLabelsCat, inputSize);
cnnPredictions = classify(trainedCNN, testImages);
cnnAccuracy = mean(cnnPredictions == testLabelsForCNN);

fprintf('Augmented CNN Test Accuracy: %.2f%%\n', cnnAccuracy*100);

% CNN confusion matrix
figure('Position', [100, 100, 900, 800]);
cm = confusionchart(testLabelsForCNN, cnnPredictions);
cm.Title = sprintf('Augmented CNN Baseline (Acc: %.2f%%)', cnnAccuracy*100);
cm.FontSize = 10;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ============================================================
% STAGE 2: EXTRACT CNN FEATURES (IMPROVED WINDOWING)
%% ============================================================
fprintf('\n========================================\n');
fprintf('STAGE 2: EXTRACTING CNN FEATURES FOR LSTM\n');
fprintf('========================================\n\n');

% Convert to feature extractor
featureNet = layerGraph(trainedCNN);
featureNet = removeLayers(featureNet, {'dropout', 'fc', 'softmax', 'classification'});
featureNet = dlnetwork(featureNet);

fprintf('Improved windowing settings:\n');
fprintf('  Time windows: %d (was 4)\n', numTimeWindows);
fprintf('  Overlap: %.1f%% (was 0%%)\n', windowOverlap*100);
fprintf('  → More time steps for LSTM to learn from\n\n');

% Extract features with BETTER windowing
fprintf('Extracting features: TRAIN\n');
[trainFeatures, trainLabelsFiltered] = extractCNNLSTMFeatures(trainPaths, trainLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

fprintf('Extracting features: VAL\n');
[valFeatures, valLabelsFiltered] = extractCNNLSTMFeatures(valPaths, valLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

fprintf('Extracting features: TEST\n');
[testFeatures, testLabelsFiltered] = extractCNNLSTMFeatures(testPaths, testLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

fprintf('\nFeature extraction complete:\n');
fprintf('  Train: %d sequences\n', numel(trainFeatures));
fprintf('  Val: %d sequences\n', numel(valFeatures));
fprintf('  Test: %d sequences\n', numel(testFeatures));
if ~isempty(trainFeatures)
    fprintf('  Dimensions: [%d × %d] (features × time)\n\n', ...
        size(trainFeatures{1}, 1), size(trainFeatures{1}, 2));
end

%% ============================================================
% STAGE 3: TRAIN LSTM
%% ============================================================
fprintf('========================================\n');
fprintf('STAGE 3: TRAINING LSTM ON CNN FEATURES\n');
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
    'ValidationPatience', 15, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('Training LSTM...\n\n');
tic;
trainedLSTM = trainNetwork(trainFeatures, trainLabelsFiltered, lstmLayers, lstmOptions);
lstmTrainingTime = toc;

fprintf('\nLSTM training completed in %.2f minutes\n', lstmTrainingTime/60);

%% ============================================================
% FINAL EVALUATION
%% ============================================================
fprintf('\n========================================\n');
fprintf('FINAL EVALUATION\n');
fprintf('========================================\n\n');

predLabels = classify(trainedLSTM, testFeatures);
trueLabels = testLabelsFiltered;
lstmAccuracy = mean(predLabels == trueLabels);

fprintf('=== RESULTS ===\n');
fprintf('Augmented CNN:      %.2f%%\n', cnnAccuracy*100);
fprintf('Augmented CNN-LSTM: %.2f%%\n', lstmAccuracy*100);
fprintf('Difference:         %+.2f%%\n\n', (lstmAccuracy - cnnAccuracy)*100);

% Interpret results
if lstmAccuracy > cnnAccuracy + 0.01  % >1% improvement
    fprintf('✓ LSTM HELPS! Temporal patterns improve classification.\n');
elseif abs(lstmAccuracy - cnnAccuracy) <= 0.01  % Within 1%
    fprintf('○ LSTM NEUTRAL. No significant benefit from temporal modeling.\n');
else
    fprintf('✗ LSTM HURTS. Stick with CNN-only approach.\n');
end

%% Per-class metrics
[confMat, order] = confusionmat(trueLabels, predLabels);
precision = diag(confMat) ./ sum(confMat, 1)';
recall = diag(confMat) ./ sum(confMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1Score(isnan(f1Score)) = 0;

fprintf('\n========================================\n');
fprintf('PER-CLASS METRICS (CNN-LSTM)\n');
fprintf('========================================\n');
fprintf('%-25s %8s %10s %10s %10s\n', 'Class', 'Samples', 'Precision', 'Recall', 'F1');
fprintf('%-25s %8s %10s %10s %10s\n', '-----', '-------', '---------', '------', '--------');
for i = 1:numel(classes)
    numSamples = sum(trueLabels == classes{i});
    fprintf('%-25s %8d %9.2f%% %9.2f%% %9.2f%%\n', ...
        classes{i}, numSamples, precision(i)*100, recall(i)*100, f1Score(i)*100);
end
fprintf('\n%-25s %8s %9.2f%% %9.2f%% %9.2f%%\n', ...
    'Macro Average', '', mean(precision)*100, mean(recall)*100, mean(f1Score)*100);
fprintf('========================================\n\n');

%% Confusion matrix
figure('Position', [100, 100, 1000, 900]);
cm = confusionchart(trueLabels, predLabels);
cm.Title = sprintf('Augmented CNN-LSTM (Acc: %.2f%%)', lstmAccuracy*100);
cm.FontSize = 10;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% Save models
modelFile = sprintf('model2_augmented_cnn_lstm_acc%.2f.mat', lstmAccuracy*100);
save(modelFile, 'trainedCNN', 'trainedLSTM', 'featureNet', 'classes', ...
     'cnnAccuracy', 'lstmAccuracy', 'augmentationFactor', ...
     'numTimeWindows', 'windowOverlap', 'inputSize');
fprintf('Models saved as: %s\n\n', modelFile);

fprintf('========================================\n');
fprintf('TRAINING COMPLETE\n');
fprintf('========================================\n');
fprintf('Total time: %.2f minutes\n', (cnnTrainingTime + lstmTrainingTime)/60);
fprintf('Final result: %.2f%% (CNN) → %.2f%% (CNN-LSTM)\n', ...
    cnnAccuracy*100, lstmAccuracy*100);
fprintf('========================================\n');

%% ============================================================
% SUPPORT FUNCTIONS (Same as before, included here)
%% ============================================================

function [paths, labels] = getValidFilesAndLabels(fileStruct)
    paths = {};
    labels = {};
    skipped = 0;
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
            else
                skipped = skipped + 1;
            end
        catch
            skipped = skipped + 1;
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
            fprintf('    Progress: %d/%d (%.1f%%)\n', i, N, 100*i/N);
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
    augLabels = categorical(zeros(totalAugmented, 1));
    
    fprintf('  Generating %d augmented samples...\n', totalAugmented);
    
    idx = 1;
    for i = 1:N
        if mod(idx, max(1, floor(totalAugmented/20))) == 0
            fprintf('    Progress: %d/%d\n', idx, totalAugmented);
        end
        
        img = images(:,:,:,i);
        label = labels(i);
        
        augImages(:,:,:,idx) = img;
        augLabels(idx) = label;
        idx = idx + 1;
        
        for j = 1:(factor-1)
            augImg = img;
            
            % Random augmentation
            augType = randi(5);
            switch augType
                case 1  % SpecAugment
                    augImg = applySpecAugment(augImg, 12, 12, 2);
                case 2  % Time stretch
                    factor = 0.85 + rand() * 0.3;
                    augImg = applyTimeStretch(augImg, factor);
                case 3  % Noise
                    level = rand() * 0.03;
                    augImg = augImg + level * randn(size(augImg));
                    augImg = max(0, min(1, augImg));
                case 4  % Time shift
                    shift = randi([-10, 10]);
                    augImg = circshift(augImg, shift, 2);
                case 5  % Mixup
                    otherIdx = randi(N);
                    alpha = rand() * 0.3 + 0.1;
                    augImg = alpha * augImg + (1-alpha) * images(:,:,:,otherIdx);
            end
            
            augImages(:,:,:,idx) = augImg;
            augLabels(idx) = label;
            idx = idx + 1;
        end
    end
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
        augImg = padarray(augImg, [0, W-newWidth, 0], 'replicate', 'post');
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
            fprintf('    Progress: %d/%d\n', i, N);
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
    
    fprintf('  Extracted %d/%d\n\n', sum(validIdx), N);
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