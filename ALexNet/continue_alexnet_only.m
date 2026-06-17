clear; clc; close all;

%% ============================================================
% CONTINUE TRAINING: AlexNet Only
% Loads existing ResNet18 and Custom CNN models
% Trains only AlexNet, then creates ensemble
%% ============================================================

fprintf('========================================\n');
fprintf('CONTINUE TRAINING: AlexNet Only\n');
fprintf('========================================\n\n');

%% STEP 1: LOAD EXISTING MODELS
fprintf('Step 1: Loading existing models...\n');

% Load the backup file you saved earlier
backupFile = 'G:\My Drive\Thesis_Dataset\models_backup_20251127_130230.mat';

if ~exist(backupFile, 'file')
    error('Backup file not found! Please check the path:\n%s', backupFile);
end

load(backupFile);
fprintf('✅ Loaded existing models:\n');
fprintf('   - model1_resnet18 (Accuracy: %.2f%%)\n', accuracy_resnet*100);
fprintf('   - model2_custom (Accuracy: %.2f%%)\n\n', accuracy_custom*100);

%% STEP 2: SETUP FOR ALEXNET
trainDir = 'G:\My Drive\Thesis_Dataset\1vector_borne\train';
valDir   = 'G:\My Drive\Thesis_Dataset\1vector_borne\val';
testDir  = 'G:\My Drive\Thesis_Dataset\1vector_borne\test';

% Input sizes
inputSizeResNet = [224 224 3];    % For ResNet18 (already trained)
inputSizeAlexNet = [227 227 3];   % For AlexNet (MUST BE 227x227!)
inputSizeCustom = [128 122 1];    % For custom CNN (already trained)

% Training parameters for AlexNet
maxEpochs = 30;
miniBatchSize = 32;
initialLearningRate = 1e-4;

%% STEP 3: LOAD DATASET
fprintf('Step 2: Loading dataset...\n');

trainFiles = dir(fullfile(trainDir,'**','*.mat'));
valFiles   = dir(fullfile(valDir,'**','*.mat'));
testFiles  = dir(fullfile(testDir,'**','*.mat'));

[trainPaths, trainLabels] = getValidFilesAndLabels(trainFiles);
[valPaths, valLabels] = getValidFilesAndLabels(valFiles);
[testPaths, testLabels] = getValidFilesAndLabels(testFiles);

trainLabels = categorical(trainLabels);
valLabels = categorical(valLabels);
testLabels = categorical(testLabels);

% Update classes variable if it doesn't exist
if ~exist('classes', 'var')
    classes = categories(trainLabels);
end
numClasses = numel(classes);

fprintf('\nDataset loaded:\n');
fprintf('  Classes: %d\n', numClasses);
fprintf('  Train: %d samples\n', numel(trainPaths));
fprintf('  Val: %d samples\n', numel(valPaths));
fprintf('  Test: %d samples\n\n', numel(testPaths));

%% STEP 4: TRAIN ALEXNET (227x227x3)
fprintf('========================================\n');
fprintf('MODEL 3: AlexNet (227x227x3)\n');
fprintf('========================================\n\n');

fprintf('Loading images for AlexNet (227x227x3)...\n');
[trainImagesAlexNet, trainLabelsAlexNet] = loadImagesForTransfer(trainPaths, trainLabels, inputSizeAlexNet);
[valImagesAlexNet, valLabelsAlexNet] = loadImagesForTransfer(valPaths, valLabels, inputSizeAlexNet);
[testImagesAlexNet, testLabelsAlexNet] = loadImagesForTransfer(testPaths, testLabels, inputSizeAlexNet);

fprintf('  Train: %d | Val: %d | Test: %d\n\n', ...
    size(trainImagesAlexNet,4), size(valImagesAlexNet,4), size(testImagesAlexNet,4));

% Load pre-trained AlexNet
fprintf('Loading pre-trained AlexNet...\n');
netAlex = alexnet;
lgraphAlex = layerGraph(netAlex);

% Replace final fully connected layers
newFC6 = fullyConnectedLayer(4096, 'Name', 'fc6_new', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraphAlex = replaceLayer(lgraphAlex, 'fc6', newFC6);

newFC7 = fullyConnectedLayer(4096, 'Name', 'fc7_new', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraphAlex = replaceLayer(lgraphAlex, 'fc7', newFC7);

newFC8 = fullyConnectedLayer(numClasses, 'Name', 'fc8_new', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraphAlex = replaceLayer(lgraphAlex, 'fc8', newFC8);

% Replace classification layer
newClassLayerAlex = classificationLayer('Name', 'output_new');
lgraphAlex = replaceLayer(lgraphAlex, 'output', newClassLayerAlex);

% Training options
optionsAlex = trainingOptions('sgdm', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', initialLearningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valImagesAlexNet, valLabelsAlexNet}, ...
    'ValidationFrequency', floor(size(trainImagesAlexNet, 4) / miniBatchSize), ...
    'ValidationPatience', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('Training AlexNet...\n\n');
tic;
model3_alexnet = trainNetwork(trainImagesAlexNet, trainLabelsAlexNet, lgraphAlex, optionsAlex);
alexnetTrainingTime = toc;

fprintf('\nEvaluating AlexNet...\n');
predictions_alexnet = classify(model3_alexnet, testImagesAlexNet);
accuracy_alexnet = mean(predictions_alexnet == testLabelsAlexNet);

fprintf('AlexNet Accuracy: %.2f%%\n', accuracy_alexnet*100);
fprintf('Training time: %.2f minutes\n\n', alexnetTrainingTime/60);

% Confusion matrix
figure('Position', [100, 100, 800, 700]);
cm3 = confusionchart(testLabelsAlexNet, predictions_alexnet);
cm3.Title = sprintf('Model 3: AlexNet (Acc: %.2f%%)', accuracy_alexnet*100);
cm3.FontSize = 10;
cm3.RowSummary = 'row-normalized';
cm3.ColumnSummary = 'column-normalized';

% Save AlexNet
save('checkpoint_model3.mat', 'model3_alexnet', 'accuracy_alexnet', '-v7.3');
fprintf('✅ Checkpoint: Model 3 saved\n\n');

%% STEP 5: LOAD TEST DATA FOR OTHER MODELS
fprintf('========================================\n');
fprintf('PREPARING ENSEMBLE\n');
fprintf('========================================\n\n');

fprintf('Loading test data for ResNet18 and Custom CNN...\n');

% Load test data for ResNet18 (224x224x3)
[testImagesResNet, testLabelsResNet] = loadImagesForTransfer(testPaths, testLabels, inputSizeResNet);

% Load test data for Custom CNN (128x122x1)
[testImagesCustom, testLabelsCustom] = loadImagesCustom(testPaths, testLabels, inputSizeCustom);

fprintf('Done loading test data.\n\n');

%% STEP 6: CREATE ENSEMBLE
fprintf('Getting predictions from all models...\n');

% Get probability scores from each model
[predictions_resnet, scores_resnet] = classify(model1_resnet18, testImagesResNet);
[predictions_custom, scores_custom] = classify(model2_custom, testImagesCustom);
[~, scores_alexnet] = classify(model3_alexnet, testImagesAlexNet);

fprintf('✅ Got predictions from all 3 models\n\n');

% Method 1: Simple Average
fprintf('Method 1: Simple Average Ensemble\n');
avgScores = (scores_resnet + scores_custom + scores_alexnet) / 3;
[~, ensemble_predictions_avg] = max(avgScores, [], 2);
ensemble_predictions_avg = classes(ensemble_predictions_avg);
accuracy_ensemble_avg = mean(ensemble_predictions_avg == testLabels);
fprintf('  Simple Average Accuracy: %.2f%%\n\n', accuracy_ensemble_avg*100);

% Method 2: Weighted Average (optimize on validation set)
fprintf('Method 2: Weighted Average Ensemble\n');
fprintf('  Loading validation data for optimization...\n');

% Load validation data
[valImagesResNet, valLabelsResNet] = loadImagesForTransfer(valPaths, valLabels, inputSizeResNet);
[valImagesCustom, valLabelsCustom] = loadImagesCustom(valPaths, valLabels, inputSizeCustom);
% AlexNet val data already loaded above

fprintf('  Optimizing weights on validation set...\n');

% Get validation scores
[~, val_scores_resnet] = classify(model1_resnet18, valImagesResNet);
[~, val_scores_custom] = classify(model2_custom, valImagesCustom);
[~, val_scores_alexnet] = classify(model3_alexnet, valImagesAlexNet);

% Grid search for best weights
bestAccuracy = 0;
bestWeights = [0.33, 0.33, 0.34];

for w1 = 0.2:0.1:0.6
    for w2 = 0.2:0.1:0.6
        w3 = 1 - w1 - w2;
        if w3 >= 0.2 && w3 <= 0.6
            weightedScores = w1*val_scores_resnet + w2*val_scores_custom + w3*val_scores_alexnet;
            [~, predictions] = max(weightedScores, [], 2);
            predictions = classes(predictions);
            acc = mean(predictions == valLabels);
            
            if acc > bestAccuracy
                bestAccuracy = acc;
                bestWeights = [w1, w2, w3];
            end
        end
    end
end

fprintf('  Best weights: [%.2f, %.2f, %.2f] (ResNet, Custom, AlexNet)\n', bestWeights);
fprintf('  Validation accuracy: %.2f%%\n', bestAccuracy*100);

% Apply best weights to test set
weightedScores = bestWeights(1)*scores_resnet + bestWeights(2)*scores_custom + bestWeights(3)*scores_alexnet;
[~, ensemble_predictions_weighted] = max(weightedScores, [], 2);
ensemble_predictions_weighted = classes(ensemble_predictions_weighted);
accuracy_ensemble_weighted = mean(ensemble_predictions_weighted == testLabels);

fprintf('  Weighted Ensemble Test Accuracy: %.2f%%\n\n', accuracy_ensemble_weighted*100);

%% STEP 7: FINAL RESULTS
fprintf('========================================\n');
fprintf('FINAL RESULTS SUMMARY\n');
fprintf('========================================\n\n');

fprintf('Individual Models:\n');
fprintf('  Model 1 (ResNet18):   %.2f%%\n', accuracy_resnet*100);
fprintf('  Model 2 (Custom CNN): %.2f%%\n', accuracy_custom*100);
fprintf('  Model 3 (AlexNet):    %.2f%%\n\n', accuracy_alexnet*100);

fprintf('Ensemble Results:\n');
fprintf('  Simple Average:       %.2f%%\n', accuracy_ensemble_avg*100);
fprintf('  Weighted Average:     %.2f%% ✅\n\n', accuracy_ensemble_weighted*100);

bestIndividual = max([accuracy_resnet, accuracy_custom, accuracy_alexnet]);
fprintf('Improvement:\n');
fprintf('  Best individual:      %.2f%%\n', bestIndividual*100);
fprintf('  Ensemble:             %.2f%%\n', accuracy_ensemble_weighted*100);
fprintf('  Gain:                 +%.2f%%\n\n', ...
    (accuracy_ensemble_weighted - bestIndividual)*100);

% Final confusion matrix
figure('Position', [100, 100, 900, 800]);
cm_final = confusionchart(testLabels, ensemble_predictions_weighted);
cm_final.Title = sprintf('Final Ensemble (Acc: %.2f%%)', accuracy_ensemble_weighted*100);
cm_final.FontSize = 10;
cm_final.RowSummary = 'row-normalized';
cm_final.ColumnSummary = 'column-normalized';

% Per-class metrics
[confMat, order] = confusionmat(testLabels, ensemble_predictions_weighted);

precision = diag(confMat) ./ sum(confMat, 1)';
recall = diag(confMat) ./ sum(confMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1Score(isnan(f1Score)) = 0;

fprintf('========================================\n');
fprintf('PER-CLASS METRICS (ENSEMBLE)\n');
fprintf('========================================\n');
fprintf('%-25s %10s %10s %10s %10s\n', 'Class', 'Samples', 'Precision', 'Recall', 'F1-Score');
fprintf('%-25s %10s %10s %10s %10s\n', '-----', '-------', '---------', '------', '--------');

for i = 1:numel(classes)
    numSamples = sum(testLabels == classes{i});
    fprintf('%-25s %10d %9.2f%% %9.2f%% %9.2f%%\n', ...
        classes{i}, numSamples, precision(i)*100, recall(i)*100, f1Score(i)*100);
end

fprintf('\n%-25s %10s %9.2f%% %9.2f%% %9.2f%%\n', ...
    'Macro Average', '', mean(precision)*100, mean(recall)*100, mean(f1Score)*100);
fprintf('========================================\n\n');

%% STEP 8: SAVE FINAL MODELS
fprintf('Saving final ensemble...\n');

% Save to your thesis folder
saveLocation = 'G:\My Drive\Thesis_Dataset\';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = fullfile(saveLocation, sprintf('ensemble_complete_%s.mat', timestamp));

save(filename, ...
    'model1_resnet18', 'model2_custom', 'model3_alexnet', ...
    'bestWeights', 'classes', ...
    'accuracy_resnet', 'accuracy_custom', 'accuracy_alexnet', ...
    'accuracy_ensemble_weighted', 'accuracy_ensemble_avg', ...
    'inputSizeResNet', 'inputSizeAlexNet', 'inputSizeCustom', ...
    'predictions_resnet', 'predictions_custom', 'predictions_alexnet', ...
    'ensemble_predictions_weighted', ...
    'testLabels', ...
    '-v7.3');

fprintf('✅ Final ensemble saved to:\n   %s\n\n', filename);

fprintf('========================================\n');
fprintf('TRAINING COMPLETE!\n');
fprintf('========================================\n');
fprintf('AlexNet training time: %.2f minutes\n', alexnetTrainingTime/60);
fprintf('Final ensemble accuracy: %.2f%%\n', accuracy_ensemble_weighted*100);

if accuracy_ensemble_weighted >= 0.90
    fprintf('\n🎉 SUCCESS! Achieved 90%%+ accuracy goal!\n');
else
    fprintf('\n⚠️  Result: %.2f%% (%.2f%% below 90%% target)\n', ...
        accuracy_ensemble_weighted*100, (0.90-accuracy_ensemble_weighted)*100);
    fprintf('Recommendation: Consider adding data augmentation for +2-3%% boost\n');
end

fprintf('========================================\n');

%% ============================================================
% SUPPORT FUNCTIONS
%% ============================================================

function [paths, labels] = getValidFilesAndLabels(fileStruct)
    paths = {};
    labels = {};
    
    for k = 1:numel(fileStruct)
        try
            fullPath = fullfile(fileStruct(k).folder, fileStruct(k).name);
            parts = split(fileStruct(k).folder, filesep);
            className = parts{end};
            
            % Skip background
            if strcmpi(className, 'background')
                continue;
            end
            
            data = load(fullPath);
            if isfield(data,'melSpec') && ~isempty(data.melSpec)
                paths{end+1,1} = fullPath;
                labels{end+1,1} = className;
            end
        catch
            continue;
        end
    end
end

function [images, labels] = loadImagesForTransfer(filePaths, fileLabels, inputSize)
    % Load images for transfer learning (224x224x3 or 227x227x3)
    N = numel(filePaths);
    images = zeros([inputSize(1:2), 3, N], 'single');
    labels = fileLabels;
    validIdx = true(N, 1);
    
    fprintf('  Loading %d images (%dx%dx%d)...\n', N, inputSize(1), inputSize(2), inputSize(3));
    progressInterval = max(1, floor(N/20));
    
    for i = 1:N
        try
            data = load(filePaths{i});
            mel = data.melSpec;
            
            if isempty(mel) || any(isnan(mel(:))) || any(isinf(mel(:)))
                validIdx(i) = false;
                continue;
            end
            
            % Normalize
            mel = rescale(mel);
            
            % Resize to target size (224x224 or 227x227)
            mel = imresize(mel, inputSize(1:2));
            
            % Replicate to 3 channels (grayscale -> RGB)
            mel3 = repmat(mel, [1, 1, 3]);
            
            images(:, :, :, i) = single(mel3);
            
        catch
            validIdx(i) = false;
        end
        
        if mod(i, progressInterval) == 0 || i == N
            fprintf('    Progress: %d/%d (%.1f%%)\n', i, N, 100*i/N);
        end
    end
    
    images = images(:, :, :, validIdx);
    labels = labels(validIdx);
    
    fprintf('  Loaded %d/%d images\n', sum(validIdx), N);
end

function [images, labels] = loadImagesCustom(filePaths, fileLabels, inputSize)
    % Load images for custom CNN (128x122x1)
    N = numel(filePaths);
    images = zeros([inputSize(1:2), 1, N], 'single');
    labels = fileLabels;
    validIdx = true(N, 1);
    
    fprintf('  Loading %d images (%dx%dx%d)...\n', N, inputSize(1), inputSize(2), inputSize(3));
    progressInterval = max(1, floor(N/20));
    
    for i = 1:N
        try
            data = load(filePaths{i});
            mel = data.melSpec;
            
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
            
        catch
            validIdx(i) = false;
        end
        
        if mod(i, progressInterval) == 0 || i == N
            fprintf('    Progress: %d/%d (%.1f%%)\n', i, N, 100*i/N);
        end
    end
    
    images = images(:, :, :, validIdx);
    labels = labels(validIdx);
    
    fprintf('  Loaded %d/%d images\n', sum(validIdx), N);
end
