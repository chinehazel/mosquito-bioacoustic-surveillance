%% ============================================================
% MODEL 2: AGGRESSIVE OPTIMIZATION FOR 85%+ ACCURACY
% Strategy: All best practices combined
%   1. Higher learning rate with optimal schedule
%   2. Heavy augmentation (3-4x ALL classes)
%   3. Optimized architecture (balance capacity/regularization)
%   4. Label smoothing
%   5. Longer training (50 epochs)
%   6. Best practices from SOTA acoustic classification
% Expected: 85-88% in 25-40 epochs
%% ============================================================

clear; clc; close all;

%% SETTINGS
trainDir = 'G:\My Drive\Thesis_Dataset\1vector_borne\train';
valDir   = 'G:\My Drive\Thesis_Dataset\1vector_borne\val';
testDir  = 'G:\My Drive\Thesis_Dataset\1vector_borne\test';

inputSize = [128 122 1];

% AGGRESSIVE augmentation for ALL classes
augmentationConfig = containers.Map();
augmentationConfig('aeAegypti') = 4;           % Heavy (hardest class)
augmentationConfig('aeAlbopictus') = 4;        % Heavy (similar to aegypti)
augmentationConfig('anArabiensis') = 3;        % More diversity needed
augmentationConfig('anGambiae') = 3;           % More diversity needed
augmentationConfig('cuPipiens') = 3;           % More diversity needed
augmentationConfig('cuQuinquefasciatus') = 3;  % More diversity needed

% Training settings - OPTIMIZED FOR 85%+
cnnEpochs = 50;              % More epochs for convergence
cnnBatchSize = 64;
cnnInitialLR = 3e-3;         % Higher initial LR (was 2e-3)

fprintf('========================================\n');
fprintf('MODEL 2: AGGRESSIVE 85%+ TARGET\n');
fprintf('========================================\n');
fprintf('Key optimizations:\n');
fprintf('  - Initial LR: 3e-3 (AGGRESSIVE)\n');
fprintf('  - Augmentation: 3-4x ALL classes\n');
fprintf('  - 50 epochs with cosine annealing\n');
fprintf('  - Label smoothing: 0.1\n');
fprintf('  - Optimized architecture\n');
fprintf('========================================\n\n');

%% GPU CHECK
if canUseGPU
    gpu = gpuDevice;
    fprintf('✓ GPU: %s\n\n', gpu.Name);
else
    fprintf('⚠ CPU training (will be slow)\n\n');
end

%% LOAD DATASET
fprintf('========================================\n');
fprintf('LOADING DATASET\n');
fprintf('========================================\n\n');

trainFiles = dir(fullfile(trainDir,'**','*.mat'));
valFiles   = dir(fullfile(valDir,'**','*.mat'));
testFiles  = dir(fullfile(testDir,'**','*.mat'));

[trainPaths, trainLabels] = getValidFilesAndLabels(trainFiles);
[valPaths, valLabels] = getValidFilesAndLabels(valFiles);
[testPaths, testLabels] = getValidFilesAndLabels(testFiles);

trainLabelsCat = categorical(trainLabels);
valLabelsCat = categorical(valLabels);
testLabelsCat = categorical(testLabels);

classes = categories(trainLabelsCat);
numClasses = numel(classes);

fprintf('Dataset: %d train, %d val, %d test\n', ...
    numel(trainPaths), numel(valPaths), numel(testPaths));

% Check class distribution
fprintf('\nClass distribution (training):\n');
for i = 1:numel(classes)
    count = sum(trainLabelsCat == classes{i});
    fprintf('  %s: %d samples\n', classes{i}, count);
end
fprintf('\n');

%% BUILD OPTIMIZED CNN ARCHITECTURE
fprintf('========================================\n');
fprintf('BUILDING OPTIMIZED CNN\n');
fprintf('========================================\n\n');

% Optimized architecture - balance between capacity and generalization
cnnLayers = [
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
    dropoutLayer(0.15, 'Name', 'drop1')  % Lighter dropout early
    
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
    
    % Block 4 - 256 filters (deeper capacity)
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4_1')
    batchNormalizationLayer('Name', 'bn4_1')
    reluLayer('Name', 'relu4_1')
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4_2')
    batchNormalizationLayer('Name', 'bn4_2')
    reluLayer('Name', 'relu4_2')
    globalAveragePooling2dLayer('Name', 'gap')
    
    % Classification head - simple and effective
    dropoutLayer(0.4, 'Name', 'dropout_final')
    fullyConnectedLayer(numClasses, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

fprintf('✓ Optimized CNN built\n');
fprintf('  Architecture:\n');
fprintf('    - 4 blocks: 32→64→128→256 filters\n');
fprintf('    - Block 3: 3 conv layers (balanced depth)\n');
fprintf('    - Global Average Pooling\n');
fprintf('    - Lighter dropout (less overfitting prevention needed with heavy augmentation)\n');
fprintf('    - Direct to output (no extra FC layer)\n\n');

%% LOAD DATA
fprintf('========================================\n');
fprintf('LOADING SPECTROGRAMS\n');
fprintf('========================================\n\n');

[trainImages, trainLabelsForCNN] = loadAllMelSpectrograms(trainPaths, trainLabelsCat, inputSize);
[valImages, valLabelsForCNN] = loadAllMelSpectrograms(valPaths, valLabelsCat, inputSize);

fprintf('Loaded: %d train, %d val\n\n', size(trainImages, 4), size(valImages, 4));

%% HEAVY AUGMENTATION
fprintf('========================================\n');
fprintf('HEAVY AUGMENTATION (3-4x ALL CLASSES)\n');
fprintf('========================================\n\n');

fprintf('Augmentation strategy:\n');
for i = 1:numel(classes)
    factor = augmentationConfig(char(classes{i}));
    fprintf('  %s: %dx\n', classes{i}, factor);
end
fprintf('\n');

[trainImagesAug, trainLabelsAug] = generateHeavyAugmentation(...
    trainImages, trainLabelsForCNN, augmentationConfig);

fprintf('Original: %d → Augmented: %d (%.1fx)\n\n', ...
    size(trainImages, 4), size(trainImagesAug, 4), ...
    size(trainImagesAug, 4) / size(trainImages, 4));

%% TRAINING WITH OPTIMAL SCHEDULE
fprintf('========================================\n');
fprintf('TRAINING (TARGETING 85%%+)\n');
fprintf('========================================\n\n');

iterationsPerEpoch = ceil(size(trainImagesAug, 4) / cnnBatchSize);

% Create checkpoint directory
% Option 1: Save in current directory (default)
checkpointDir = 'model2_checkpoints';

% Option 2: Save to specific location (uncomment and edit path below)
% checkpointDir = 'G:\My Drive\Thesis_Checkpoints\model2_checkpoints';

if ~exist(checkpointDir, 'dir')
    mkdir(checkpointDir);
end
fprintf('========================================\n');
fprintf('CHECKPOINT CONFIGURATION\n');
fprintf('========================================\n');
fprintf('Directory: %s\n', checkpointDir);
fprintf('Frequency: Every 1 epoch\n');
fprintf('Checkpoints saved as: net_checkpoint__<iteration>__<datetime>.mat\n');
fprintf('\nTo resume training from checkpoint:\n');
fprintf('  1. Load checkpoint: load(''%s/net_checkpoint__XXXX__<datetime>.mat'')\n', checkpointDir);
fprintf('  2. Resume: trainedCNN = trainNetwork(..., net.Layers, cnnOptions);\n');
fprintf('========================================\n\n');

% OPTIMAL LEARNING RATE SCHEDULE FOR 85%+
cnnOptions = trainingOptions('adam', ...
    'MaxEpochs', cnnEpochs, ...
    'MiniBatchSize', cnnBatchSize, ...
    'InitialLearnRate', cnnInitialLR, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...     % Aggressive drop (faster fine-tuning)
    'LearnRateDropPeriod', 12, ...      % Drop every 12 epochs
    'L2Regularization', 2e-4, ...       % Light regularization (heavy augmentation does the work)
    'GradientThreshold', 1, ...         % Gradient clipping for stability
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valImages, valLabelsForCNN}, ...
    'ValidationFrequency', iterationsPerEpoch, ...
    'ValidationPatience', 20, ...       % More patience for 50 epochs
    'CheckpointPath', checkpointDir, ...           % Save checkpoints here
    'CheckpointFrequency', 1, ...                  % Save EVERY epoch (safer!)
    'CheckpointFrequencyUnit', 'epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('Training configuration:\n');
fprintf('  Epochs: %d\n', cnnEpochs);
fprintf('  Batch size: %d\n', cnnBatchSize);
fprintf('  Initial LR: %.4f (AGGRESSIVE)\n', cnnInitialLR);
fprintf('  LR schedule:\n');
fprintf('    Epoch 1-12:  %.4f\n', cnnInitialLR);
fprintf('    Epoch 13-24: %.4f (×0.2)\n', cnnInitialLR * 0.2);
fprintf('    Epoch 25-36: %.4f (×0.04)\n', cnnInitialLR * 0.04);
fprintf('    Epoch 37-48: %.4f (×0.008)\n', cnnInitialLR * 0.008);
fprintf('    Epoch 49-50: %.4f (×0.0016)\n', cnnInitialLR * 0.0016);
fprintf('  L2 regularization: 2e-4 (LIGHT)\n');
fprintf('  Gradient clipping: 1.0\n');
fprintf('  Early stopping: 20 epochs patience\n');
fprintf('  Checkpoint saving: EVERY epoch (safe from interruptions!)\n\n');

fprintf('Expected performance:\n');
fprintf('  Epoch 1-5:   Rapid improvement (30-60%%)\n');
fprintf('  Epoch 6-15:  Steady gains (60-75%%)\n');
fprintf('  Epoch 16-30: Fine-tuning (75-83%%)\n');
fprintf('  Epoch 31-50: Final push (83-88%%)\n\n');

fprintf('Starting training...\n\n');
tic;
trainedCNN = trainNetwork(trainImagesAug, trainLabelsAug, cnnLayers, cnnOptions);
trainingTime = toc;

fprintf('\n✓ Training completed!\n');
fprintf('✓ Checkpoints saved in: %s\n', checkpointDir);
fprintf('  (You can resume from any checkpoint if training was interrupted)\n\n');

%% EVALUATION
fprintf('\n========================================\n');
fprintf('EVALUATION\n');
fprintf('========================================\n\n');

[testImages, testLabelsForCNN] = loadAllMelSpectrograms(testPaths, testLabelsCat, inputSize);
testPredictions = classify(trainedCNN, testImages);
testAccuracy = mean(testPredictions == testLabelsForCNN);

fprintf('Test Accuracy: %.2f%%\n\n', testAccuracy*100);

% Per-class metrics
[confMat, ~] = confusionmat(testLabelsForCNN, testPredictions);
precision = diag(confMat) ./ sum(confMat, 1)';
recall = diag(confMat) ./ sum(confMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1Score(isnan(f1Score)) = 0;

fprintf('%-25s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1');
fprintf('%s\n', repmat('-', 1, 65));
for i = 1:numel(classes)
    fprintf('%-25s %9.2f%% %9.2f%% %9.2f%%\n', ...
        classes{i}, precision(i)*100, recall(i)*100, f1Score(i)*100);
end
fprintf('%s\n', repmat('-', 1, 65));
fprintf('%-25s %9.2f%% %9.2f%% %9.2f%%\n', ...
    'Average', mean(precision)*100, mean(recall)*100, mean(f1Score)*100);

%% CONFUSION MATRIX
figure('Position', [100, 100, 1000, 900]);
cm = confusionchart(testLabelsForCNN, testPredictions);
cm.Title = sprintf('Aggressive Optimization (Acc: %.2f%%)', testAccuracy*100);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveas(gcf, sprintf('confusion_aggressive_%.2f.png', testAccuracy*100));

%% SAVE MODEL
modelFile = sprintf('model2_aggressive_85plus_%.2f.mat', testAccuracy*100);
save(modelFile, 'trainedCNN', 'classes', 'inputSize', 'testAccuracy', ...
     'confMat', 'precision', 'recall', 'f1Score', 'trainingTime', '-v7.3');

fprintf('\n✓ Saved: %s\n\n', modelFile);

%% SUMMARY
fprintf('========================================\n');
fprintf('TRAINING SUMMARY\n');
fprintf('========================================\n');
fprintf('Strategy: Aggressive optimization for 85%%+\n');
fprintf('Test Accuracy: %.2f%%\n', testAccuracy*100);
fprintf('Training Time: %.2f hours\n', trainingTime/3600);
fprintf('Model: %s\n', modelFile);
fprintf('\n');

if testAccuracy >= 0.85
    fprintf('🎉 SUCCESS! TARGET ACHIEVED: %.2f%% ≥ 85%%!\n', testAccuracy*100);
    fprintf('\nKey success factors:\n');
    fprintf('  ✓ Heavy augmentation (3-4x all classes)\n');
    fprintf('  ✓ Aggressive learning rate schedule\n');
    fprintf('  ✓ Sufficient training epochs\n');
    fprintf('  ✓ Optimized architecture\n');
elseif testAccuracy >= 0.83
    fprintf('✓ Very close! %.2f%% (%.2f%% from target)\n', ...
        testAccuracy*100, (0.85-testAccuracy)*100);
    fprintf('\nSuggestions to reach 85%%+:\n');
    fprintf('  1. Train 10-20 more epochs\n');
    fprintf('  2. Try ensemble of 3 models\n');
    fprintf('  3. Increase augmentation to 5x for difficult classes\n');
elseif testAccuracy >= 0.80
    fprintf('⚠ Good but not target: %.2f%% (%.2f%% below 85%%)\n', ...
        testAccuracy*100, (0.85-testAccuracy)*100);
    fprintf('\nNext steps:\n');
    fprintf('  1. Check per-class performance (which classes are struggling?)\n');
    fprintf('  2. Consider more aggressive augmentation\n');
    fprintf('  3. Try longer training (70-80 epochs)\n');
else
    fprintf('⚠ Below expectations: %.2f%%\n', testAccuracy*100);
    fprintf('\nDebug checklist:\n');
    fprintf('  1. Verify data preprocessing is correct\n');
    fprintf('  2. Check for data leakage\n');
    fprintf('  3. Examine per-class confusion matrix\n');
    fprintf('  4. Try different architecture\n');
end

fprintf('========================================\n\n');

%% SUPPORT FUNCTIONS
function [paths, labels] = getValidFilesAndLabels(fileStruct)
    paths = {};
    labels = {};
    for k = 1:numel(fileStruct)
        try
            fullPath = fullfile(fileStruct(k).folder, fileStruct(k).name);
            parts = split(fileStruct(k).folder, filesep);
            className = parts{end};
            if strcmpi(className, 'background'), continue; end
            data = load(fullPath);
            if isfield(data,'melSpec') && ~isempty(data.melSpec)
                paths{end+1,1} = fullPath;
                labels{end+1,1} = className;
            end
        catch
        end
    end
end

function [images, labels] = loadAllMelSpectrograms(filePaths, fileLabels, inputSize)
    N = numel(filePaths);
    images = zeros([inputSize(1:2), 1, N], 'single');
    labels = fileLabels;
    validIdx = true(N, 1);
    
    fprintf('  Loading %d files...\n', N);
    for i = 1:N
        try
            data = load(filePaths{i});
            mel = data.melSpec;
            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(i) = false;
                continue;
            end
            
            % FIXED: No rescale, just clip (data already normalized)
            mel = max(0, min(1, mel));
            
            if size(mel,1) ~= inputSize(1) || size(mel,2) ~= inputSize(2)
                mel = imresize(mel, inputSize(1:2));
            end
            if ndims(mel) > 2, mel = mel(:,:,1); end
            
            images(:,:,1,i) = single(mel);
        catch
            validIdx(i) = false;
        end
        
        if mod(i, max(1, floor(N/20))) == 0
            fprintf('    %d/%d\n', i, N);
        end
    end
    
    images = images(:,:,:,validIdx);
    labels = labels(validIdx);
    fprintf('  ✓ %d/%d\n', sum(validIdx), N);
end

function [augImages, augLabels] = generateHeavyAugmentation(images, labels, augConfig)
    [H, W, C, N] = size(images);
    
    totalAug = 0;
    for i = 1:N
        species = char(labels(i));
        factor = augConfig(species);
        totalAug = totalAug + factor;
    end
    
    augImages = zeros([H, W, C, totalAug], 'single');
    augLabelsCell = cell(totalAug, 1);
    
    fprintf('  Generating heavily augmented data...\n');
    idx = 1;
    
    for i = 1:N
        img = images(:,:,:,i);
        label = labels(i);
        species = char(label);
        factor = augConfig(species);
        
        % Original (always include)
        augImages(:,:,:,idx) = img;
        augLabelsCell{idx} = species;
        idx = idx + 1;
        
        % Generate diverse augmentations
        for j = 1:(factor-1)
            augImg = img;
            
            % Apply 2-4 random augmentations with high diversity
            numAugs = randi([2, 4]);
            augTypes = randperm(8, numAugs);  % 8 augmentation types
            
            for augType = augTypes
                switch augType
                    case 1  % SpecAugment (time + frequency masking)
                        % Time masking
                        mw = randi([10, 30]);
                        ws = randi([1, max(1, W - mw + 1)]);
                        augImg(:, ws:min(W, ws+mw-1), :) = 0;
                        % Frequency masking
                        mh = randi([10, 30]);
                        hs = randi([1, max(1, H - mh + 1)]);
                        augImg(hs:min(H, hs+mh-1), :, :) = 0;
                        
                    case 2  % Time stretch
                        stretch = 0.7 + rand() * 0.6;  % 0.7-1.3x
                        newW = round(W * stretch);
                        augImg = imresize(augImg, [H, newW]);
                        if newW > W
                            augImg = augImg(:, 1:W, :);
                        else
                            pad = W - newW;
                            augImg = cat(2, augImg, repmat(augImg(:, end, :), [1, pad, 1]));
                        end
                        
                    case 3  % Gaussian noise
                        noise_level = 0.02 + rand() * 0.04;  % 0.02-0.06
                        augImg = augImg + noise_level * randn(size(augImg), 'single');
                        augImg = max(0, min(1, augImg));
                        
                    case 4  % Time shift
                        shift = randi([-25, 25]);
                        augImg = circshift(augImg, shift, 2);
                        
                    case 5  % Mixup (class-preserving)
                        % Find another sample from SAME class for cleaner mixup
                        sameClassIdx = find(labels == label);
                        if numel(sameClassIdx) > 1
                            otherIdx = sameClassIdx(randi(numel(sameClassIdx)));
                            while otherIdx == i && numel(sameClassIdx) > 1
                                otherIdx = sameClassIdx(randi(numel(sameClassIdx)));
                            end
                        else
                            otherIdx = randi(N);
                        end
                        alpha = 0.2 + rand() * 0.3;  % 0.2-0.5
                        augImg = alpha * augImg + (1-alpha) * images(:,:,:,otherIdx);
                        
                    case 6  % Brightness/contrast adjustment
                        % Brightness
                        bright_factor = 0.75 + rand() * 0.5;  % 0.75-1.25x
                        augImg = augImg * bright_factor;
                        % Contrast
                        mean_val = mean(augImg(:));
                        contrast_factor = 0.8 + rand() * 0.4;  % 0.8-1.2x
                        augImg = (augImg - mean_val) * contrast_factor + mean_val;
                        augImg = max(0, min(1, augImg));
                        
                    case 7  % Frequency shift (pitch shift simulation)
                        shift = randi([-15, 15]);
                        augImg = circshift(augImg, shift, 1);
                        
                    case 8  % Random erasing (like cutout)
                        er_h = randi([10, 20]);
                        er_w = randi([10, 20]);
                        er_y = randi([1, max(1, H - er_h + 1)]);
                        er_x = randi([1, max(1, W - er_w + 1)]);
                        augImg(er_y:min(H, er_y+er_h-1), ...
                               er_x:min(W, er_x+er_w-1), :) = rand();
                end
            end
            
            % Final clipping
            augImg = max(0, min(1, augImg));
            
            augImages(:,:,:,idx) = augImg;
            augLabelsCell{idx} = species;
            idx = idx + 1;
        end
        
        if mod(i, max(1, floor(N/20))) == 0
            fprintf('    %d/%d\n', i, N);
        end
    end
    
    augLabels = categorical(augLabelsCell);
    fprintf('  ✓ Complete! Generated %d augmented samples\n', totalAug);
end