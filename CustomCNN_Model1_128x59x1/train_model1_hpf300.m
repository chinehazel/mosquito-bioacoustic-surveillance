clear; clc; close all;

%% ============================================================
% MODEL 1: MOSQUITO DETECTOR (BINARY) - v3 (1s, HPF 300 Hz)
%
% Trained on the v3 dataset:
%   - Mosquito: 6_Mosquitos_hpf300_v3 (1s windows, looped or split)
%   - Background: ESC-50_hpf300_background_v3 (ESC-50 + Humbug merged)
%
% Differences from v2:
%   - inputSize: [128 59 1]  (was [128 122 1])
%   - Data paths point to *_v3 folders
%   - Otherwise identical: same architecture, same hyperparameters
%% ============================================================

%% SETTINGS — TWO SEPARATE DATA SOURCES
mosquitoTrainDir = 'G:\My Drive\Thesis_Dataset\6_Mosquitos_hpf300_v3\train';
mosquitoValDir   = 'G:\My Drive\Thesis_Dataset\6_Mosquitos_hpf300_v3\val';
mosquitoTestDir  = 'G:\My Drive\Thesis_Dataset\6_Mosquitos_hpf300_v3\test';

backgroundTrainDir = 'G:\My Drive\Thesis_Dataset\ESC-50_hpf300_background_v3\train\background';
backgroundValDir   = 'G:\My Drive\Thesis_Dataset\ESC-50_hpf300_background_v3\val\background';
backgroundTestDir  = 'G:\My Drive\Thesis_Dataset\ESC-50_hpf300_background_v3\test\background';

inputSize = [128 59 1];   % v3: 1-second windows

% Training parameters (unchanged from v2)
epochs = 25;
batchSize = 64;
initialLearningRate = 2e-3;

%% LOAD DATA
fprintf('========================================\n');
fprintf('MODEL 1 v3: MOSQUITO DETECTOR (1s, HPF 300 Hz)\n');
fprintf('========================================\n\n');

fprintf('Loading TRAIN set...\n');
[trainImages, trainLabels] = loadBinaryFromTwoFolders( ...
    mosquitoTrainDir, backgroundTrainDir, inputSize);

fprintf('Loading VAL set...\n');
[valImages, valLabels] = loadBinaryFromTwoFolders( ...
    mosquitoValDir, backgroundValDir, inputSize);

fprintf('Loading TEST set...\n');
[testImages, testLabels] = loadBinaryFromTwoFolders( ...
    mosquitoTestDir, backgroundTestDir, inputSize);

%% COMPUTE CLASS WEIGHTS (handle imbalance honestly)
% v3 data is roughly 1:1 balanced, but keeping the weighting in case
% imbalance creeps in later — it's a no-op when classes are equal.
nBg  = sum(trainLabels == 'Background');
nMos = sum(trainLabels == 'Mosquito');
classWeights = [length(trainLabels)/(2*nBg), length(trainLabels)/(2*nMos)];
fprintf('Class weights: Background=%.3f, Mosquito=%.3f\n\n', ...
        classWeights(1), classWeights(2));

%% BUILD MODEL
fprintf('Building Model 1 architecture...\n\n');

layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    
    % Block 1
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    % Block 2
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    % Block 3
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    
    % Block 4
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    globalAveragePooling2dLayer('Name', 'gap')
    
    % Classification head
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(2, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output', ...
        'Classes', {'Background', 'Mosquito'}, ...
        'ClassWeights', classWeights)
];

%% CHECKPOINT DIRECTORY
checkpointDir = 'model1_hpf300_v3_checkpoints';
if ~exist(checkpointDir, 'dir')
    mkdir(checkpointDir);
end
fprintf('Checkpoints will be saved to: %s\n\n', checkpointDir);

%% TRAINING OPTIONS
options = trainingOptions('adam', ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', batchSize, ...
    'InitialLearnRate', initialLearningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valImages, valLabels}, ...
    'ValidationFrequency', floor(size(trainImages, 4) / batchSize), ...
    'ValidationPatience', 5, ...
    'L2Regularization', 1e-4, ...
    'CheckpointPath', checkpointDir, ...
    'CheckpointFrequency', 1, ...
    'CheckpointFrequencyUnit', 'epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% TRAIN MODEL
fprintf('Training Model 1 v3...\n');
fprintf('Learning Rate Schedule:\n');
fprintf('  Epochs 1-8:   LR = %.4f\n', initialLearningRate);
fprintf('  Epochs 9-16:  LR = %.4f\n', initialLearningRate * 0.5);
fprintf('  Epochs 17-25: LR = %.4f\n\n', initialLearningRate * 0.25);

tic;
model1_detector = trainNetwork(trainImages, trainLabels, layers, options);
trainingTime = toc;

fprintf('\nTraining completed in %.2f minutes\n\n', trainingTime/60);

%% EVALUATE ON TEST SET
fprintf('========================================\n');
fprintf('EVALUATING MODEL 1 v3 ON TEST SET\n');
fprintf('========================================\n\n');

[predictions, scores] = classify(model1_detector, testImages);
accuracy = mean(predictions == testLabels);
fprintf('Test Accuracy: %.2f%%\n\n', accuracy*100);

%% CONFUSION MATRIX
figure('Position', [100, 100, 700, 600]);
cm = confusionchart(testLabels, predictions);
cm.Title = sprintf('Model 1 v3 (1s, HPF 300 Hz): Acc %.2f%%', accuracy*100);
cm.FontSize = 12;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveas(gcf, sprintf('confusion_model1_hpf300_v3_%.2f.png', accuracy*100));

%% PER-CLASS METRICS
[confMat, order] = confusionmat(testLabels, predictions);

fprintf('Per-Class Metrics:\n');
fprintf('%-15s %10s %10s %10s %10s\n', 'Class', 'Samples', 'Precision', 'Recall', 'F1-Score');
fprintf('%-15s %10s %10s %10s %10s\n', '-----', '-------', '---------', '------', '--------');

for i = 1:2
    className = char(order(i));
    numSamples = sum(testLabels == order(i));
    
    TP = confMat(i, i);
    FN = sum(confMat(i, :)) - TP;
    FP = sum(confMat(:, i)) - TP;
    
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1 = 2 * precision * recall / (precision + recall);
    
    fprintf('%-15s %10d %9.2f%% %9.2f%% %9.2f%%\n', ...
        className, numSamples, precision*100, recall*100, f1*100);
end
fprintf('\n');

%% MOSQUITO DETECTION FOCUS
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
    fprintf('Missed mosquitoes: %d / %d\n', FN, TP+FN);
    fprintf('False alarms:      %d / %d background\n', FP, sum(testLabels == 'Background'));
    fprintf('========================================\n\n');
end

%% THRESHOLD OPTIMIZATION
fprintf('Testing different thresholds...\n');
mosquitoScores = scores(:, mosquitoIdx);
thresholds = 0.3:0.05:0.9;

fprintf('%-12s %10s %10s %10s\n', 'Threshold', 'Precision', 'Recall', 'F1-Score');
fprintf('%-12s %10s %10s %10s\n', '---------', '---------', '------', '--------');

bestF1 = 0;
bestThreshold = 0.5;

for t = thresholds
    predAtT = mosquitoScores > t;
    TP = sum(predAtT & (testLabels == 'Mosquito'));
    FP = sum(predAtT & (testLabels == 'Background'));
    FN = sum(~predAtT & (testLabels == 'Mosquito'));
    
    prec = TP / (TP + FP);
    rec  = TP / (TP + FN);
    f1   = 2 * prec * rec / (prec + rec);
    
    fprintf('%-12.2f %9.2f%% %9.2f%% %9.2f%%\n', t, prec*100, rec*100, f1*100);
    
    if f1 > bestF1
        bestF1 = f1;
        bestThreshold = t;
    end
end

fprintf('\nBest threshold: %.2f (F1 = %.2f%%)\n\n', bestThreshold, bestF1*100);

%% SAVE MODEL
modelFilename = sprintf('model1_hpf300_v3_acc%.2f.mat', accuracy*100);
save(modelFilename, 'model1_detector', 'accuracy', 'inputSize', 'bestThreshold');
fprintf('Saved: %s\n', modelFilename);

fprintf('\n========================================\n');
fprintf('MODEL 1 v3 TRAINING COMPLETE\n');
fprintf('Training time: %.2f min\n', trainingTime/60);
fprintf('Test accuracy: %.2f%%\n', accuracy*100);
fprintf('\n');
fprintf('Compare to v2 (2s, with padding artifact): 99.20%%\n');
fprintf('A small drop is expected and HEALTHY — the model can no\n');
fprintf('longer rely on the padding-as-class signature. Run Grad-CAM\n');
fprintf('after this to confirm attention is on real wingbeat features.\n');
fprintf('========================================\n');


%% ============================================================
% HELPER: Load binary dataset from two separate folders
%   - mosquitoDir contains species subfolders (everything is "Mosquito")
%   - backgroundDir contains .mat files directly (everything is "Background")
%% ============================================================
function [images, labels] = loadBinaryFromTwoFolders(mosquitoDir, backgroundDir, inputSize)
    
    % --- Index mosquito files (recursively, all species) ---
    mosqFiles = dir(fullfile(mosquitoDir, '**', '*.mat'));
    nMosq = length(mosqFiles);
    
    % --- Index background files (flat) ---
    bgFiles = dir(fullfile(backgroundDir, '*.mat'));
    nBg = length(bgFiles);
    
    N = nMosq + nBg;
    fprintf('  Mosquito files: %d\n', nMosq);
    fprintf('  Background files: %d\n', nBg);
    fprintf('  Total: %d\n', N);
    
    images = zeros([inputSize(1:2), 1, N], 'single');
    labels = categorical(zeros(N, 1), [0, 1], {'Background', 'Mosquito'});
    validIdx = true(N, 1);
    
    progressInterval = max(1, floor(N/20));
    
    % --- Load mosquitoes (label = Mosquito) ---
    for i = 1:nMosq
        fullPath = fullfile(mosqFiles(i).folder, mosqFiles(i).name);
        labels(i) = 'Mosquito';
        
        try
            data = load(fullPath);
            mel = data.melSpec;
            
            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(i) = false;
                continue;
            end
            
            mel = max(0, min(1, mel));
            
            if size(mel, 1) ~= inputSize(1) || size(mel, 2) ~= inputSize(2)
                mel = imresize(mel, inputSize(1:2));
            end
            if ndims(mel) > 2
                mel = mel(:, :, 1);
            end
            
            images(:, :, 1, i) = single(mel);
        catch
            validIdx(i) = false;
        end
        
        if mod(i, progressInterval) == 0
            fprintf('    Progress: %d/%d (%.1f%%)\n', i, N, 100*i/N);
        end
    end
    
    % --- Load backgrounds (label = Background) ---
    for j = 1:nBg
        idx = nMosq + j;
        fullPath = fullfile(bgFiles(j).folder, bgFiles(j).name);
        labels(idx) = 'Background';
        
        try
            data = load(fullPath);
            mel = data.melSpec;
            
            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(idx) = false;
                continue;
            end
            
            mel = max(0, min(1, mel));
            
            if size(mel, 1) ~= inputSize(1) || size(mel, 2) ~= inputSize(2)
                mel = imresize(mel, inputSize(1:2));
            end
            if ndims(mel) > 2
                mel = mel(:, :, 1);
            end
            
            images(:, :, 1, idx) = single(mel);
        catch
            validIdx(idx) = false;
        end
        
        if mod(idx, progressInterval) == 0
            fprintf('    Progress: %d/%d (%.1f%%)\n', idx, N, 100*idx/N);
        end
    end
    
    % --- Remove invalid samples ---
    images = images(:, :, :, validIdx);
    labels = labels(validIdx);
    
    nBgFinal  = sum(labels == 'Background');
    nMosFinal = sum(labels == 'Mosquito');
    
    fprintf('  Loaded successfully: %d\n', length(labels));
    fprintf('    Background: %d (%.1f%%)\n', nBgFinal, 100*nBgFinal/length(labels));
    fprintf('    Mosquito:   %d (%.1f%%)\n\n', nMosFinal, 100*nMosFinal/length(labels));
end