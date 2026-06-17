load('model_val93.95_test85.43.mat');

% Load test data
testDir = 'G:\My Drive\Thesis_Dataset\6_Mosquitos_Mel_Fixed\test';
testFiles = dir(fullfile(testDir, '**', '*.mat'));

[testPaths, testLabels] = getFilesAndLabels(testFiles);
testLabelsCat = categorical(testLabels);
testImages = loadAllImages(testPaths);

% Get predictions
testPreds = classify(netTrained, testImages);

% Per-class metrics
classes = categories(testLabelsCat);
fprintf('\n%-22s %8s %10s %10s %10s %10s\n', ...
    'Class', 'Samples', 'Precision', 'Recall', 'F1', 'Accuracy');
fprintf('%s\n', repmat('-', 1, 75));

[confMat, ~] = confusionmat(testLabelsCat, testPreds);

for i = 1:numel(classes)
    className = char(classes(i));
    numSamples = sum(testLabelsCat == classes(i));

    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;

    precision = TP / (TP + FP);
    recall    = TP / (TP + FN);
    f1        = 2 * precision * recall / (precision + recall);
    acc       = TP / sum(confMat(i, :));

    fprintf('%-22s %8d %9.2f%% %9.2f%% %9.2f%% %9.2f%%\n', ...
        className, numSamples, precision*100, recall*100, f1*100, acc*100);
end

% ============================================
% HELPER FUNCTIONS
% ============================================
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