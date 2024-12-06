clc;
clear;

% Load dataset
load('preprocessed_data/merge_TimeD_FreqD_Fday_and_Mday.mat');

% Transpose data for convenience (samples as rows)
data = mergedData';

% Detect outliers based on Z-scores for the entire dataset
zScores = zscore(data);
outliers = abs(zScores) > 3; % Define outliers as points with |Z| > 3

% Replace outliers with the median of the respective feature/column
for col = 1:size(data, 2)
    colData = data(:, col);
    colData(outliers(:, col)) = median(colData(~outliers(:, col)));
    data(:, col) = colData; % Update with cleaned column
end

% Transpose back 
mergedData = data';

% Separate features and labels
X = mergedData(:, 1:end-1); % All columns except the last one are features
Y = mergedData(:, end);     % The last column is the label

% Normalize the features for better training performance
X = normalize(X);

% Ensure that X and Y have the same number of samples
if size(X, 1) ~= length(Y)
    error('The number of samples in X and Y do not match.');
end

% Convert labels to categorical for multi-class classification
Y = categorical(Y);

% Split data into training, validation, and testing sets
numSamples = size(X, 1);
trainRatio = 0.8;
valRatio = 0.1;
testRatio = 0.1;

idx = randperm(numSamples); % Randomize indices
trainIdx = idx(1:round(trainRatio*numSamples));
valIdx = idx(round(trainRatio*numSamples)+1:round((trainRatio+valRatio)*numSamples));
testIdx = idx(round((trainRatio+valRatio)*numSamples)+1:end);

X_train = X(trainIdx, :);
Y_train = Y(trainIdx);
X_val = X(valIdx, :);
Y_val = Y(valIdx);
X_test = X(testIdx, :);
Y_test = Y(testIdx);

% Transpose data for neural network input
X_train = X_train';
X_val = X_val';
X_test = X_test';
Y_train = onehotencode(Y_train, 2); % Convert to one-hot encoding
Y_val = onehotencode(Y_val, 2);
Y_test = onehotencode(Y_test, 2);

% Define the Feedforward Neural Network
hiddenLayerSizes = [10, 10]; % Hidden layers configuration
net = feedforwardnet(hiddenLayerSizes, 'trainscg'); % Use scaled conjugate gradient for classification

% Configure training parameters
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-4;
net.trainParam.lr = 0.01;
net.trainParam.show = 25;
net.trainParam.max_fail = 10;

% Update output layer to use softmax activation for classification
net.layers{end}.transferFcn = 'softmax';

% Split data for training, validation, and testing
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.1;

% Train the network
[net, tr] = train(net, X_train, Y_train);

% Evaluate performance
trainPredictions = net(X_train);
valPredictions = net(X_val);
testPredictions = net(X_test);

% Convert predictions back from one-hot encoding
trainLabels = vec2ind(trainPredictions);
valLabels = vec2ind(valPredictions);
testLabels = vec2ind(testPredictions);

% Calculate classification accuracies
trainAccuracy = sum(trainLabels' == vec2ind(Y_train)) / length(trainIdx) * 100;
valAccuracy = sum(valLabels' == vec2ind(Y_val)) / length(valIdx) * 100;
testAccuracy = sum(testLabels' == vec2ind(Y_test)) / length(testIdx) * 100;

% Display classification accuracies
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);
fprintf('Testing Accuracy: %.2f%%\n', testAccuracy);

% Plot Confusion Matrix
figure;
plotconfusion(categorical(vec2ind(Y_train)), categorical(trainLabels'), 'Training');
figure;
plotconfusion(categorical(vec2ind(Y_val)), categorical(valLabels'), 'Validation');
figure;
plotconfusion(categorical(vec2ind(Y_test)), categorical(testLabels'), 'Testing');

% Save the trained model
save('models/multi_class_classification_model.mat', 'net', 'tr');
