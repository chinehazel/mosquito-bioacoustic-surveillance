clear; clc; close all;

%% ============================================================
% MODEL 1: MOSQUITO DETECTOR (BINARY CLASSIFICATION)
% Purpose: Detect presence/absence of mosquito in audio
% Classes: Background vs Mosquito (all species combined)
%
% Architecture mirrors Model 2: 4-block CNN with progressive
% dropout, gradient clipping, and heavier regularization.
% Includes checkpoint saving for safe resume on interruption.
%% ============================================================

%% SETTINGS
trainDir = 'G:\My Drive\Thesis_Dataset\1vector_borne\train';
valDir   = 'G:\My Drive\Thesis_Dataset\1vector_borne\val';
testDir  = 'G:\My Drive\Thesis_Dataset\1vector_borne\test';

inputSize = [128 122 1];

epochs              = 35;
batchSize           = 64;
initialLearningRate = 2e-3;

% Checkpoint directory
checkpointDir = 'G:\My Drive\Documentation (Training)\model1_checkpoints';

% ============================================================
% RESUME FLAG
% Set resumeTraining = true to continue from a saved checkpoint
% Set resumeTraining = false to start fresh
% ============================================================
resumeTraining = false;

%% LOAD DATA
fprintf('========================================\n');
fprintf('MODEL 1: MOSQUITO DETECTOR (BINARY)\n');
fprintf('========================================\n\n');

fprintf('Loading TRAIN set...\n');
[trainImages, trainLabels] = loadBinaryDataset(trainDir, inputSize);

fprintf('Loading VAL set...\n');
[valImages, valLabels] = loadBinaryDataset(valDir, inputSize);

fprintf('Loading TEST set...\n');
[testImages, testLabels] = loadBinaryDataset(testDir, inputSize);

%% CHECKPOINT DIRECTORY SETUP
if ~exist(checkpointDir, 'dir')
    mkdir(checkpointDir);
end
fprintf('Checkpoint directory: %s\n\n', checkpointDir);

%% BUILD OR LOAD MODEL
if resumeTraining
    fprintf('========================================\n');
    fprintf('RESUMING FROM CHECKPOINT\n');
    fprintf('========================================\n\n');

    checkpointFiles = dir(fullfile(checkpointDir, 'net_checkpoint__*.mat'));

    if isempty(checkpointFiles)
        error('No checkpoint files found in: %s\nSet resumeTraining = false to start fresh.', checkpointDir);
    end

    [~, latestIdx] = max([checkpointFiles.datenum]);
    latestFile = fullfile(checkpointDir, checkpointFiles(latestIdx).name);
    fprintf('Loading checkpoint: %s\n\n', checkpointFiles(latestIdx).name);

    checkpoint = load(latestFile);
    layersToUse = checkpoint.net.Layers;

    % Ask user how many epochs have already completed
    fprintf('----------------------------------------\n');
    fprintf('IMPORTANT: How many epochs already done?\n');
    fprintf('Check the checkpoint filename for the iteration number.\n');
    fprintf('Divide iterations by ceil(trainSize / batchSize) to get epochs.\n');
    fprintf('Then set remainingEpochs below accordingly.\n');
    fprintf('----------------------------------------\n\n');

    % EDIT THIS: set to (epochs - already completed epochs)
    remainingEpochs = 20;  % <-- adjust this before resuming
    fprintf('Remaining epochs to train: %d\n\n', remainingEpochs);
    epochsToRun = remainingEpochs;

else
    fprintf('========================================\n');
    fprintf('BUILDING MODEL 1 ARCHITECTURE\n');
    fprintf('========================================\n\n');

    layersToUse = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')

        % Initial convolution
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_init')
        batchNormalizationLayer('Name', 'bn_init')
        reluLayer('Name', 'relu_init')

        % Block 1 - 32 filters
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_1')
        batchNormalizationLayer('Name', 'bn1_1')
        reluLayer('Name', 'relu1_1')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_2')
        batchNormalizationLayer('Name', 'bn1_2')
        reluLayer('Name', 'relu1_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        dropoutLayer(0.15, 'Name', 'drop1')

        % Block 2 - 64 filters
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_1')
        batchNormalizationLayer('Name', 'bn2_1')
        reluLayer('Name', 'relu2_1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_2')
        batchNormalizationLayer('Name', 'bn2_2')
        reluLayer('Name', 'relu2_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        dropoutLayer(0.2, 'Name', 'drop2')

        % Block 3 - 128 filters (3 conv layers)
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
        dropoutLayer(0.25, 'Name', 'drop3')

        % Block 4 - 256 filters
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4_1')
        batchNormalizationLayer('Name', 'bn4_1')
        reluLayer('Name', 'relu4_1')
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4_2')
        batchNormalizationLayer('Name', 'bn4_2')
        reluLayer('Name', 'relu4_2')
        globalAveragePooling2dLayer('Name', 'gap')

        % Classification head
        dropoutLayer(0.4, 'Name', 'dropout_final')
        fullyConnectedLayer(2, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    epochsToRun = epochs;
    fprintf('Architecture built. Training from scratch for %d epochs.\n\n', epochsToRun);
end

%% TRAINING OPTIONS
iterationsPerEpoch = ceil(size(trainImages, 4) / batchSize);

options = trainingOptions('adam', ...
    'MaxEpochs', epochsToRun, ...
    'MiniBatchSize', batchSize, ...
    'InitialLearnRate', initialLearningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'L2Regularization', 2e-4, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valImages, valLabels}, ...
    'ValidationFrequency', iterationsPerEpoch, ...
    'ValidationPatience', 10, ...
    'CheckpointPath', checkpointDir, ...
    'CheckpointFrequency', 1, ...
    'CheckpointFrequencyUnit', 'epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% TRAINING INFO PRINTOUT
fprintf('========================================\n');
fprintf('TRAINING CONFIGURATION\n');
fprintf('========================================\n');
fprintf('  Epochs to run:    %d\n', epochsToRun);
fprintf('  Batch size:       %d\n', batchSize);
fprintf('  Initial LR:       %.4f\n', initialLearningRate);
fprintf('  LR schedule:\n');
fprintf('    Epochs  1-10:   LR = %.4f\n', initialLearningRate);
fprintf('    Epochs 11-20:   LR = %.4f\n', initialLearningRate * 0.5);
fprintf('    Epochs 21-30:   LR = %.4f\n', initialLearningRate * 0.25);
fprintf('    Epochs 31-35:   LR = %.4f\n', initialLearningRate * 0.125);
fprintf('  L2 regularization: 2e-4\n');
fprintf('  Gradient clipping: 1.0\n');
fprintf('  Early stopping:    10 epochs patience\n');
fprintf('  Checkpoints:       every epoch -> %s\n', checkpointDir);
fprintf('========================================\n\n');

%% TRAIN MODEL
fprintf('Starting training...\n\n');
tic;
model1_detector = trainNetwork(trainImages, trainLabels, layersToUse, options);
trainingTime = toc;

fprintf('\nTraining completed in %.2f minutes\n\n', trainingTime/60);

%% EVALUATE ON TEST SET
fprintf('========================================\n');
fprintf('EVALUATING MODEL 1 ON TEST SET\n');
fprintf('========================================\n\n');

[predictions, scores] = classify(model1_detector, testImages);

accuracy = mean(predictions == testLabels);
fprintf('Test Accuracy: %.2f%%\n\n', accuracy*100);

% Confusion matrix
figure('Position', [100, 100, 700, 600]);
cm = confusionchart(testLabels, predictions);
cm.Title = sprintf('Model 1: Mosquito Detector (Acc: %.2f%%)', accuracy*100);
cm.FontSize = 12;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Detailed metrics per class
[confMat, order] = confusionmat(testLabels, predictions);

fprintf('Per-Class Metrics:\n');
fprintf('%-15s %10s %10s %10s %10s\n', 'Class', 'Samples', 'Precision', 'Recall', 'F1-Score');
fprintf('%-15s %10s %10s %10s %10s\n', '-----', '-------', '---------', '------', '--------');

for i = 1:2
    className  = char(order(i));
    numSamples = sum(testLabels == order(i));

    TP = confMat(i, i);
    FN = sum(confMat(i, :)) - TP;
    FP = sum(confMat(:, i)) - TP;

    precision = TP / (TP + FP);
    recall    = TP / (TP + FN);
    f1        = 2 * precision * recall / (precision + recall);

    fprintf('%-15s %10d %9.2f%% %9.2f%% %9.2f%%\n', ...
        className, numSamples, precision*100, recall*100, f1*100);
end

fprintf('\n');

% Mosquito-specific performance
mosquitoIdx = find(order == 'Mosquito');
if ~isempty(mosquitoIdx)
    TP = confMat(mosquitoIdx, mosquitoIdx);
    FN = sum(confMat(mosquitoIdx, :)) - TP;
    FP = sum(confMat(:, mosquitoIdx)) - TP;

    mosquitoRecall    = TP / (TP + FN);
    mosquitoPrecision = TP / (TP + FP);

    fprintf('========================================\n');
    fprintf('MOSQUITO DETECTION PERFORMANCE\n');
    fprintf('========================================\n');
    fprintf('Precision: %.2f%%\n', mosquitoPrecision*100);
    fprintf('Recall:    %.2f%%\n', mosquitoRecall*100);
    fprintf('Missed mosquitoes: %d out of %d\n', FN, TP+FN);
    fprintf('False alarms: %d out of %d background samples\n', FP, sum(testLabels == 'Background'));
    fprintf('========================================\n\n');
end

%% THRESHOLD OPTIMIZATION
fprintf('Testing different classification thresholds...\n');
mosquitoScores = scores(:, mosquitoIdx);
thresholds = 0.3:0.05:0.9;

fprintf('%-12s %10s %10s %10s\n', 'Threshold', 'Precision', 'Recall', 'F1-Score');
fprintf('%-12s %10s %10s %10s\n', '---------', '---------', '------', '--------');

bestF1        = 0;
bestThreshold = 0.5;

for t = thresholds
    predAtThreshold = mosquitoScores > t;
    TP = sum(predAtThreshold & (testLabels == 'Mosquito'));
    FP = sum(predAtThreshold & (testLabels == 'Background'));
    FN = sum(~predAtThreshold & (testLabels == 'Mosquito'));

    prec = TP / (TP + FP);
    rec  = TP / (TP + FN);
    f1   = 2 * prec * rec / (prec + rec);

    fprintf('%-12.2f %9.2f%% %9.2f%% %9.2f%%\n', t, prec*100, rec*100, f1*100);

    if f1 > bestF1
        bestF1        = f1;
        bestThreshold = t;
    end
end

fprintf('\nBest threshold: %.2f (F1 = %.2f%%)\n', bestThreshold, bestF1*100);
fprintf('Note: Default threshold is 0.5\n\n');

%% SAVE FINAL MODEL
modelFilename = sprintf('model1_mosquito_detector_acc%.2f.mat', accuracy*100);
save(modelFilename, 'model1_detector', 'accuracy', 'inputSize', 'bestThreshold');
fprintf('Model 1 saved as: %s\n', modelFilename);

fprintf('\n========================================\n');
fprintf('MODEL 1 TRAINING COMPLETE!\n');
fprintf('Total Training Time: %.2f minutes\n', trainingTime/60);
fprintf('Final Test Accuracy: %.2f%%\n', accuracy*100);
fprintf('========================================\n');

%% ============================================================
% HELPER FUNCTION: Load Binary Dataset (no augmentation)
%% ============================================================
function [images, labels] = loadBinaryDataset(dataDir, inputSize)
    allFiles = dir(fullfile(dataDir, '**', '*.mat'));

    N      = length(allFiles);
    images = zeros([inputSize(1:2), 1, N], 'single');
    labels = categorical(zeros(N, 1), [0, 1], {'Background', 'Mosquito'});

    validIdx         = true(N, 1);
    progressInterval = max(1, floor(N/20));

    fprintf('  Total files found: %d\n', N);
    fprintf('  Loading mel spectrograms...\n');

    for i = 1:N
        fullPath = fullfile(allFiles(i).folder, allFiles(i).name);

        parts         = strsplit(allFiles(i).folder, filesep);
        speciesFolder = parts{end};

        if strcmpi(speciesFolder, 'background')
            labels(i) = 'Background';
        else
            labels(i) = 'Mosquito';
        end

        try
            data = load(fullPath);
            mel  = data.melSpec;

            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(i) = false;
                continue;
            end

            % Clip to valid [0, 1] range only; data is already normalized
            mel = max(0, min(1, mel));

            if size(mel, 1) ~= inputSize(1) || size(mel, 2) ~= inputSize(2)
                mel = imresize(mel, inputSize(1:2));
            end

            if ndims(mel) > 2
                mel = mel(:, :, 1);
            end

            images(:, :, 1, i) = single(mel);

        catch ME
            warning('Error loading file %d: %s', i, ME.message);
            validIdx(i) = false;
        end

        if mod(i, progressInterval) == 0 || i == N
            fprintf('    Progress: %d/%d (%.1f%%)\n', i, N, 100*i/N);
        end
    end

    images = images(:, :, :, validIdx);
    labels = labels(validIdx);

    numBackground = sum(labels == 'Background');
    numMosquito   = sum(labels == 'Mosquito');

    fprintf('  Successfully loaded: %d samples\n', length(labels));
    fprintf('    Background: %d (%.1f%%)\n', numBackground, 100*numBackground/length(labels));
    fprintf('    Mosquito:   %d (%.1f%%)\n\n', numMosquito, 100*numMosquito/length(labels));

    sampleMel = images(:, :, 1, 1);
    fprintf('  Data range verification:\n');
    fprintf('    Min: %.4f, Max: %.4f, Mean: %.4f\n', ...
        min(sampleMel(:)), max(sampleMel(:)), mean(sampleMel(:)));
    fprintf('    (Should be in [0, 1] range)\n\n');
end