`clear; clear; close all;

%% ============================================================
% SETTINGS
%% ============================================================
trainDir = "G:\My Drive\Thesis_Dataset\1vector_borne\train";
valDir   = "G:\My Drive\Thesis_Dataset\1vector_borne\val";
testDir  = "G:\My Drive\Thesis_Dataset\1vector_borne\test";

inputSize = [128 128 1];

% Stage 1: CNN training
cnnEpochs = 30;
cnnBatchSize = 32;
cnnLearningRate = 1e-3;

% Stage 2: LSTM training
numTimeWindows = 10;
windowOverlap = 0.5;
lstmEpochs = 50;
lstmBatchSize = 16;
lstmLearningRate = 1e-3;

%% ============================================================
% LOAD DATA
%% ============================================================
fprintf("Loading dataset files...\n");

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

disp("========================================");
disp("TWO-STAGE CNN-LSTM TRAINING");
disp("========================================");
fprintf("  Classes: %d\n", numClasses);
fprintf("  Train samples: %d\n", numel(trainPaths));
fprintf("  Val samples: %d\n", numel(valPaths));
fprintf("  Test samples: %d\n", numel(testPaths));
disp("========================================");

%% ============================================================
% STAGE 1: TRAIN CNN CLASSIFIER
%% ============================================================
disp(" ");
disp("========================================");
disp("STAGE 1: TRAINING CNN FEATURE EXTRACTOR");
disp("========================================");

% Build complete CNN for classification
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
disp("Loading and preprocessing data for CNN training...");
[trainImages, trainLabelsForCNN] = loadAllMelSpectrograms(trainPaths, trainLabelsCat, inputSize);
[valImages, valLabelsForCNN] = loadAllMelSpectrograms(valPaths, valLabelsCat, inputSize);

fprintf("  Loaded %d training images\n", size(trainImages, 4));
fprintf("  Loaded %d validation images\n", size(valImages, 4));

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
fprintf("\nTraining CNN classifier...\n");
tic;
trainedCNN = trainNetwork(trainImages, trainLabelsForCNN, cnnLayers, cnnOptions);
cnnTrainingTime = toc;

fprintf("CNN training completed in %.2f minutes\n", cnnTrainingTime/60);

% Evaluate CNN alone
disp(" ");
disp("Evaluating CNN performance (baseline)...");
[testImages, testLabelsForCNN] = loadAllMelSpectrograms(testPaths, testLabelsCat, inputSize);
cnnPredictions = classify(trainedCNN, testImages);
cnnAccuracy = mean(cnnPredictions == testLabelsForCNN);
fprintf("CNN-only Test Accuracy: %.2f%%\n", cnnAccuracy*100);

%% ============================================================
% STAGE 2: EXTRACT CNN FEATURES FOR LSTM
%% ============================================================
disp(" ");
disp("========================================");
disp("STAGE 2: EXTRACTING CNN FEATURES FOR LSTM");
disp("========================================");

% Convert trained CNN to feature extractor (remove classification layers)
featureNet = layerGraph(trainedCNN);
featureNet = removeLayers(featureNet, {'dropout', 'fc', 'softmax', 'classification'});
featureNet = dlnetwork(featureNet);

fprintf("Feature extraction layer: 'gap' (256 features)\n");

% Extract features from time windows
disp(" ");
disp("Extracting features: TRAIN");
[trainFeatures, trainLabelsFiltered] = extractCNNLSTMFeatures(trainPaths, trainLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

disp("Extracting features: VAL");
[valFeatures, valLabelsFiltered] = extractCNNLSTMFeatures(valPaths, valLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

disp("Extracting features: TEST");
[testFeatures, testLabelsFiltered] = extractCNNLSTMFeatures(testPaths, testLabelsCat, ...
    featureNet, inputSize, numTimeWindows, windowOverlap);

% Validate features
fprintf("\nFeature Extraction Summary:\n");
fprintf("  Train sequences: %d\n", numel(trainFeatures));
fprintf("  Val sequences: %d\n", numel(valFeatures));
fprintf("  Test sequences: %d\n", numel(testFeatures));

if ~isempty(trainFeatures)
    fprintf("  Feature dimensions: [%d × %d] (features × time)\n", ...
        size(trainFeatures{1}, 1), size(trainFeatures{1}, 2));
    
    % Check feature quality
    allFeats = cat(2, trainFeatures{:});
    featStd = std(allFeats(:));
    fprintf("  Feature std deviation: %.4f\n", featStd);
    
    if featStd < 0.01
        warning("Low feature variance! Features may not be discriminative.");
    end
end

%% ============================================================
% STAGE 3: TRAIN LSTM ON CNN FEATURES
%% ============================================================
disp(" ");
disp("========================================");
disp("STAGE 3: TRAINING LSTM CLASSIFIER");
disp("========================================");

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

fprintf("\nTraining LSTM on CNN features...\n");
tic;
trainedLSTM = trainNetwork(trainFeatures, trainLabelsFiltered, lstmLayers, lstmOptions);
lstmTrainingTime = toc;

fprintf("LSTM training completed in %.2f minutes\n", lstmTrainingTime/60);

%% ============================================================
% EVALUATE CNN-LSTM
%% ============================================================
disp(" ");
disp("========================================");
disp("FINAL EVALUATION");
disp("========================================");

predLabels = classify(trainedLSTM, testFeatures);
trueLabels = testLabelsFiltered;

accuracy = mean(predLabels == trueLabels);
fprintf("\n=== RESULTS COMPARISON ===\n");
fprintf("CNN-only Accuracy:  %.2f%%\n", cnnAccuracy*100);
fprintf("CNN-LSTM Accuracy:  %.2f%%\n", accuracy*100);
fprintf("Improvement:        %.2f%%\n\n", (accuracy - cnnAccuracy)*100);

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

fprintf("Per-Class Metrics (CNN-LSTM):\n");
fprintf("%-25s %10s %10s %10s %10s\n", "Class", "Samples", "Precision", "Recall", "F1-Score");
fprintf("%-25s %10s %10s %10s %10s\n", "-----", "-------", "---------", "------", "--------");

for i = 1:numel(classes)
    numSamples = sum(trueLabels == classes{i});
    fprintf("%-25s %10d %9.2f%% %9.2f%% %9.2f%%\n", ...
        classes{i}, numSamples, precision(i)*100, recall(i)*100, f1Score(i)*100);
end

fprintf("\n%-25s %10s %9.2f%% %9.2f%% %9.2f%%\n", ...
    "Macro Average", "", mean(precision)*100, mean(recall)*100, mean(f1Score)*100);

%% ============================================================
% VISUALIZATION
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
modelFilename = sprintf("mosquito_two_stage_cnn_lstm_acc%.2f.mat", accuracy*100);
save(modelFilename, "trainedCNN", "trainedLSTM", "featureNet", "classes", ...
     "accuracy", "cnnAccuracy", "inputSize", "numTimeWindows", "windowOverlap");
fprintf("\nModels saved as: %s\n", modelFilename);

disp(" ");
disp("========================================");
disp("TWO-STAGE TRAINING COMPLETE!");
fprintf("Total Training Time: %.2f minutes\n", (cnnTrainingTime + lstmTrainingTime)/60);
fprintf("Final CNN-LSTM Accuracy: %.2f%%\n", accuracy*100);
disp("========================================");

%% ============================================================
% SUPPORT FUNCTIONS
%% ============================================================

function [paths, labels] = getValidFilesAndLabels(fileStruct)
    paths = {};
    labels = {};
    skipped = 0;
    
    for k = 1:numel(fileStruct)
        try
            fullPath = fullfile(fileStruct(k).folder, fileStruct(k).name);
            data = load(fullPath);
            if isfield(data,'melSpec') && ~isempty(data.melSpec)
                paths{end+1,1} = fullPath;
                parts = split(fileStruct(k).folder, filesep);
                labels{end+1,1} = parts{end};
            else
                skipped = skipped + 1;
            end
        catch
            skipped = skipped + 1;
        end
    end
    
    if skipped > 0
        fprintf("  Warning: Skipped %d files\n", skipped);
    end
end

function [images, labels] = loadAllMelSpectrograms(filePaths, fileLabels, inputSize)
    % Load all mel spectrograms into a 4D array for CNN training
    N = numel(filePaths);
    images = zeros([inputSize(1:2), 1, N], 'single');
    labels = fileLabels;
    validIdx = true(N, 1);
    
    fprintf("  Loading %d mel spectrograms...\n", N);
    progressInterval = max(1, floor(N/20));
    
    for i = 1:N
        if mod(i, progressInterval) == 0 || i == N
            fprintf("    Progress: %d/%d (%.1f%%)\n", i, N, 100*i/N);
        end
        
        try
            loadedData = load(filePaths{i});
            mel = loadedData.melSpec;
            
            % Check for valid data
            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(i) = false;
                continue;
            end
            
            % Normalize
            mel = rescale(mel);
            mel = imresize(mel, inputSize(1:2));
            
            % Ensure 2D
            if ndims(mel) > 2
                mel = mel(:, :, 1);
            end
            
            images(:, :, 1, i) = single(mel);
            
        catch ME
            warning("Error loading file %d: %s", i, ME.message);
            validIdx(i) = false;
        end
    end
    
    % Remove invalid samples
    images = images(:, :, :, validIdx);
    labels = labels(validIdx);
    
    fprintf("  Successfully loaded %d/%d images\n", sum(validIdx), N);
end

function [features, labels] = extractCNNLSTMFeatures(filePaths, fileLabels, ...
    featureNet, inputSize, numWindows, overlap)
    
    N = numel(filePaths);
    features = cell(N, 1);
    labels = fileLabels;
    
    fprintf("  Processing %d files...\n", N);
    progressInterval = max(1, floor(N/20));
    
    validIdx = true(N, 1);
    
    for i = 1:N
        if mod(i, progressInterval) == 0 || i == N
            fprintf("    Progress: %d/%d (%.1f%%)\n", i, N, 100*i/N);
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
            warning("Error processing file %d: %s", i, ME.message);
            validIdx(i) = false;
        end
    end
    
    features = features(validIdx);
    labels = labels(validIdx);
    
    fprintf("  Extracted features from %d/%d files\n\n", sum(validIdx), N);
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
    
    if timeBins < numWindows
        windows = {};
        return;
    end
    
    windowSize = floor(timeBins / (numWindows - (numWindows-1)*overlap));
    stepSize = floor(windowSize * (1 - overlap));
    windowSize = max(windowSize, 10);
    stepSize = max(stepSize, 5);
    
    windows = cell(numWindows, 1);
    
    for w = 1:numWindows
        startIdx = (w-1)*stepSize + 1;
        endIdx = min(startIdx + windowSize - 1, timeBins);
        
        if startIdx > timeBins
            windows = {};
            return;
        end
        
        window = mel(:, startIdx:endIdx);
        
        if size(window, 2) < windowSize
            window = padarray(window, [0 windowSize-size(window,2)], 'replicate', 'post');
        end
        
        window = imresize(window, inputSize(1:2));
        
        if ndims(window) > 2
            window = window(:, :, 1);
        end
        
        windows{w} = window;
    end
end