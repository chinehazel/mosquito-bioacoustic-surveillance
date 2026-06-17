%% MEMORY-EFFICIENT: Problem Trio Training
clear; clc; close all;

trainDir = 'G:\My Drive\Thesis_Dataset\6_Mosquitos_Mel_Fixed\train';
valDir   = 'G:\My Drive\Thesis_Dataset\6_Mosquitos_Mel_Fixed\val';
testDir  = 'G:\My Drive\Thesis_Dataset\6_Mosquitos_Mel_Fixed\test';

checkpointDir = 'model_problem_trio_fixed';
if ~exist(checkpointDir, 'dir'), mkdir(checkpointDir); end

fprintf('=== MEMORY-EFFICIENT TRAINING FOR PROBLEM TRIO ===\n\n');

%% Load data
trainFiles = dir(fullfile(trainDir, '**', '*.mat'));
valFiles = dir(fullfile(valDir, '**', '*.mat'));
testFiles = dir(fullfile(testDir, '**', '*.mat'));

[trainPaths, trainLabels] = getFilesAndLabels(trainFiles);
[valPaths, valLabels] = getFilesAndLabels(valFiles);
[testPaths, testLabels] = getFilesAndLabels(testFiles);

trainLabelsCat = categorical(trainLabels);
valLabelsCat = categorical(valLabels);
testLabelsCat = categorical(testLabels);

classes = categories(trainLabelsCat);
numClasses = numel(classes);

fprintf('Loading all data...\n');
trainImages = loadAllImages(trainPaths);
valImages = loadAllImages(valPaths);
testImages = loadAllImages(testPaths);

fprintf('\nOriginal training size: %d\n', size(trainImages, 4));

%% MEMORY-EFFICIENT: Only 2x (not 3x) for problem trio
fprintf('\nApplying 2x augmentation to problem trio (memory-efficient)...\n');

problemSpecies = {'aeAegypti', 'anArabiensis', 'cuPipiens'};

% Pre-allocate for efficiency
totalAugmented = 0;
for sp = 1:length(problemSpecies)
    speciesIdx = trainLabelsCat == problemSpecies{sp};
    totalAugmented = totalAugmented + sum(speciesIdx);
end

newSize = size(trainImages, 4) + totalAugmented;
augImages = zeros([224, 224, 3, newSize], 'single');
augLabelsCell = cell(newSize, 1);

fprintf('Pre-allocating %d total samples...\n', newSize);

% Copy original data
augImages(:,:,:,1:size(trainImages,4)) = trainImages;
for i = 1:length(trainLabelsCat)
    augLabelsCell{i} = char(trainLabelsCat(i));
end

idx = size(trainImages, 4) + 1;

% Add augmented samples
for sp = 1:length(problemSpecies)
    speciesName = problemSpecies{sp};
    fprintf('  %s: ', speciesName);
    
    speciesIdx = trainLabelsCat == speciesName;
    speciesImages = trainImages(:,:,:,speciesIdx);
    numSpecies = size(speciesImages, 4);
    
    % Generate 1x more (2x total including original)
    for i = 1:numSpecies
        img = speciesImages(:,:,:,i);
        
        % Time shift augmentation
        shift = randi([-30, 30]);
        if shift > 0
            augImg = [zeros(224, shift, 3, 'single'), img(:, 1:end-shift, :)];
        elseif shift < 0
            augImg = [img(:, -shift+1:end, :), zeros(224, -shift, 3, 'single')];
        else
            augImg = img;
        end
        
        augImages(:,:,:,idx) = augImg;
        augLabelsCell{idx} = speciesName;
        idx = idx + 1;
    end
    
    fprintf('%d -> %d (2x)\n', numSpecies, numSpecies * 2);
end

trainImages = augImages;
trainLabelsCat = categorical(augLabelsCell);

fprintf('\nNew training size: %d\n', size(trainImages, 4));
fprintf('Memory usage: %.2f GB\n\n', numel(trainImages)*4/1e9);

% Clear large temporary variables
clear augImages augLabelsCell speciesImages;

%% ResNet-18
fprintf('Loading ResNet-18...\n');
net = resnet18;
lgraph = layerGraph(net);

newFC = fullyConnectedLayer(numClasses, ...
    'Name', 'fc_mosquito', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

lgraph = replaceLayer(lgraph, 'fc1000', newFC);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', ...
    classificationLayer('Name', 'classoutput_mosquito'));

fprintf('Network ready\n\n');

%% Training Options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 60, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 5e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 15, ...
    'Momentum', 0.95, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valImages, valLabelsCat}, ...
    'ValidationFrequency', floor(size(trainImages,4)/32), ...
    'ValidationPatience', 15, ...
    'CheckpointPath', checkpointDir, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('Training configuration:\n');
fprintf('  Epochs: 60\n');
fprintf('  Batch: 32\n');
fprintf('  Initial LR: 0.0005\n');
fprintf('  Strategy: 2x augmentation for problem trio\n\n');

fprintf('Starting training...\n\n');

tic;
netTrained = trainNetwork(trainImages, trainLabelsCat, lgraph, options);
trainingTime = toc;

fprintf('\nTraining complete in %.2f hours\n\n', trainingTime/3600);

%% Evaluation
fprintf('=== EVALUATION ===\n\n');

valPreds = classify(netTrained, valImages);
valAccuracy = mean(valPreds == valLabelsCat) * 100;

testPreds = classify(netTrained, testImages);
testAccuracy = mean(testPreds == testLabelsCat) * 100;

fprintf('Overall:\n');
fprintf('  Validation: %.2f%%\n', valAccuracy);
fprintf('  Test:       %.2f%%\n\n', testAccuracy);

fprintf('Per-class accuracy:\n');
fprintf('%-20s %10s %10s\n', 'Species', 'Accuracy', 'vs Previous');
fprintf('%s\n', repmat('-', 1, 50));

previousAccs = struct(...
    'aeAegypti', 70.92, ...
    'aeAlbopictus', 88.33, ...
    'anArabiensis', 81.94, ...
    'anGambiae', 82.47, ...
    'cuPipiens', 83.81, ...
    'cuQuinquefasciatus', 99.49);

for i = 1:numClasses
    className = char(classes(i));
    classIdx = testLabelsCat == classes(i);
    classAcc = mean(testPreds(classIdx) == testLabelsCat(classIdx)) * 100;
    
    if isfield(previousAccs, className)
        prevAcc = previousAccs.(className);
        change = classAcc - prevAcc;
        changeStr = sprintf('%+.2f%%', change);
    else
        changeStr = 'N/A';
    end
    
    fprintf('%-20s %9.2f%% %10s', className, classAcc, changeStr);
    
    if ismember(className, problemSpecies)
        fprintf(' [PROBLEM]');
    end
    
    fprintf('\n');
end

fprintf('\n');

%% Confusion Matrix
figure('Position', [100, 100, 1400, 600]);

subplot(1,2,1);
confusionchart(valLabelsCat, valPreds);
title(sprintf('Validation (%.2f%%)', valAccuracy));

subplot(1,2,2);
confusionchart(testLabelsCat, testPreds);
title(sprintf('Test (%.2f%%)', testAccuracy));

saveas(gcf, fullfile(checkpointDir, sprintf('confusion_val%.2f_test%.2f.png', valAccuracy, testAccuracy)));

%% Save
modelFile = fullfile(checkpointDir, sprintf('model_val%.2f_test%.2f.mat', valAccuracy, testAccuracy));
save(modelFile, 'netTrained', 'classes', 'valAccuracy', 'testAccuracy', 'trainingTime', '-v7.3');

fprintf('Model saved: %s\n\n', modelFile);

if testAccuracy >= 85.0
    fprintf('SUCCESS! %.2f%% >= 85%%\n', testAccuracy);
else
    fprintf('Result: %.2f%% (need %.2f%% more)\n', testAccuracy, 85 - testAccuracy);
end

%% Helpers
function [paths, labels] = getFilesAndLabels(fileStruct)
    paths = {};
    labels = {};
    for k = 1:numel(fileStruct)
        fullPath = fullfile(fileStruct(k).folder, fileStruct(k).name);
        parts = split(fileStruct(k).folder, filesep);
        className = parts{end};
        paths{end+1,1} = fullPath;
        labels{end+1,1} = className;
    end
end

function images = loadAllImages(filePaths)
    N = numel(filePaths);
    images = zeros([224, 224, 3, N], 'single');
    fprintf('  Loading %d files', N);
    for i = 1:N
        data = load(filePaths{i});
        img = data.img;
        img = single(img);
        img = max(0, min(1, img));
        images(:,:,:,i) = img;
        if mod(i, 1000) == 0, fprintf('.'); end
    end
    fprintf(' done!\n');
end