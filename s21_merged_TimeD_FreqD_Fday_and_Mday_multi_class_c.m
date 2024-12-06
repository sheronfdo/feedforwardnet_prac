clc;
clear;

%% Load dataset
load('preprocessed_data/merge_TimeD_FreqD_Fday_and_Mday.mat');

% Transpose data for convenience (samples as rows)
data = mergedData';

%% Detect and Replace Outliers
zScores = zscore(data);
outliers = abs(zScores) > 3; % Define outliers as points with |Z| > 3

% Replace outliers with the median of their respective feature
for col = 1:size(data, 2)
    colData = data(:, col);
    colData(outliers(:, col)) = median(colData(~outliers(:, col)));
    data(:, col) = colData;
end

% Transpose back
mergedData = data';

%% Separate Features and Labels
X = mergedData(:, 1:end-1); % Features
Y = mergedData(:, end);     % Labels

% Normalize features
X = normalize(X);

%% Handle Missing Values
nanMask = isnan(X); % Check for NaNs
nanColumns = any(nanMask, 1);
fprintf('Columns with NaN values: %s\n', mat2str(find(nanColumns)));

% Remove problematic columns (hardcoded or dynamically)
columnsToRemove = [111, 121]; % Replace with dynamic criteria if possible
X(:, columnsToRemove) = [];

% Ensure no NaN or Inf values remain
if any(isnan(X), 'all') || any(isinf(X), 'all')
    error('Data contains NaN or infinite values. Clean the data before proceeding.');
end

%% Dimensionality Reduction (PCA)
[coeff, score, latent, tsquared, explained] = pca(X);

% Display explained variance
disp('Explained Variance by Each Component:');
disp(explained);

% Select components to retain 90% variance
explainedVariance = cumsum(explained);
numComponents = find(explainedVariance >= 90, 1);

% Reduce dimensionality
X_reduced = score(:, 1:numComponents);
fprintf('Original number of features: %d\n', size(X, 2));
fprintf('Reduced number of features: %d\n', size(X_reduced, 2));

% Transpose X and Y to match neural network input format
X = X_reduced';
Y = Y';

%% Define Feedforward Neural Network
hiddenLayerSizes = [30, 30]; % Two hidden layers with 30 neurons each
net = feedforwardnet(hiddenLayerSizes);

% Configure for classification
net.layers{end}.transferFcn = 'softmax'; % Use softmax for multi-class problems
net.performFcn = 'crossentropy';        % Cross-entropy loss function

% Training Parameters
net.trainParam.epochs = 1000;  % Number of epochs
net.trainParam.goal = 1e-6;    % Performance goal
net.trainParam.lr = 0.01;      % Learning rate
net.trainParam.show = 25;      % Show progress every 25 iterations
net.trainParam.max_fail = 10;  % Early stopping (max validation failures)

% Randomize initial weights
net.IW{1,1} = randn(size(net.IW{1,1})) * 0.01;
net.LW{2,1} = randn(size(net.LW{2,1})) * 0.01;
net.b{1} = randn(size(net.b{1})) * 0.01;
net.b{2} = randn(size(net.b{2})) * 0.01;

% Split data into training, validation, and testing sets
net.divideParam.trainRatio = 0.8; % 80% for training
net.divideParam.valRatio = 0.1;   % 10% for validation
net.divideParam.testRatio = 0.1;  % 10% for testing

%% Train the Network
[net, tr] = train(net, X, Y);

% Debugging
fprintf('Training stopped due to: %s\n', tr.stop);

% Ensure indices are within bounds
numSamples = size(X, 2);
if max(tr.trainInd) > numSamples || max(tr.valInd) > numSamples || max(tr.testInd) > numSamples
    error('Index out of bounds. Check training, validation, and testing indices.');
end

%% Evaluate Network Performance
% Performance metrics
trainPerformance = perform(net, Y(tr.trainInd), net(X(:, tr.trainInd)));
valPerformance = perform(net, Y(tr.valInd), net(X(:, tr.valInd)));
testPerformance = perform(net, Y(tr.testInd), net(X(:, tr.testInd)));

% Classification accuracy
trainPredictions = round(net(X(:, tr.trainInd)));
trainAccuracy = sum(trainPredictions == Y(tr.trainInd)) / length(tr.trainInd) * 100;

valPredictions = round(net(X(:, tr.valInd)));
valAccuracy = sum(valPredictions == Y(tr.valInd)) / length(tr.valInd) * 100;

testPredictions = round(net(X(:, tr.testInd)));
testAccuracy = sum(testPredictions == Y(tr.testInd)) / length(tr.testInd) * 100;

% ROC and AUC for the test set
[~, ~, ~, AUC] = perfcurve(Y(tr.testInd)', net(X(:, tr.testInd)), 1);

% Display results
fprintf('Training Performance (MSE): %.6f\n', trainPerformance);
fprintf('Validation Performance (MSE): %.6f\n', valPerformance);
fprintf('Testing Performance (MSE): %.6f\n', testPerformance);
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);
fprintf('Testing Accuracy: %.2f%%\n', testAccuracy);
fprintf('AUC (Test Set): %.2f\n', AUC);

%% Visualization
% 1. Training Performance
figure;
plotperform(tr);
title('Performance');
grid on;

% 2. Confusion Matrix
figure;
predictedLabels = round(net(X));
confusionchart(Y, predictedLabels, 'Title', 'Confusion Matrix', ...
    'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
grid on;

% 3. Regression Plots
figure;
plotregression(Y(tr.trainInd), net(X(:, tr.trainInd)), 'Training');
figure;
plotregression(Y(tr.valInd), net(X(:, tr.valInd)), 'Validation');
figure;
plotregression(Y(tr.testInd), net(X(:, tr.testInd)), 'Testing');

% 4. ROC Curve
figure;
[Xroc, Yroc, Troc, AUC] = perfcurve(Y', net(X), 1);
plot(Xroc, Yroc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
grid on;

% Save model
save('models/merged_model.mat', 'net', 'tr', 'trainPerformance', ...
     'valPerformance', 'testPerformance');
