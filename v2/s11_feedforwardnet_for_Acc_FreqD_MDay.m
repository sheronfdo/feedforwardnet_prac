clc;
clear;

% Load dataset
load('preprocessed_data/mergedData_Acc_FreqD_MDay.mat');

% Separate features and labels
X = mergedData{:, 1:end-1}; % All columns except the last one are features
userIDs = mergedData.UserID; % The last column is the user ID

% Convert user IDs to categorical labels
Y = categorical(userIDs); % Use user IDs as categorical labels

% Normalize the features for better training performance
X = normalize(X);

% Ensure that X and Y have the same number of samples
if size(X, 1) ~= length(Y)
    error('The number of samples in X and Y do not match.');
end

% Convert X to an array and transpose it to match the expected input format for the neural network
X = X';
Y = Y';

% Convert labels to one-hot encoding for multi-class classification
numClasses = length(categories(Y));
Y_onehot = full(ind2vec(double(Y), numClasses)); % Convert categorical to numeric for one-hot encoding

% Define the Feedforward Neural Network
hiddenLayerSizes = [10, 10]; % Example: 2 hidden layers with 10 neurons each
net = feedforwardnet(hiddenLayerSizes);

% Configure the network for multi-class classification
net.layers{end}.transferFcn = 'softmax'; % Use softmax for multi-class classification
net.performFcn = 'crossentropy';         % Cross-entropy loss for classification

% Configure training parameters
net.trainParam.epochs = 100; % Increase epochs for better convergence
net.trainParam.goal = 1e-6;  % Performance goal
net.trainParam.lr = 0.01;    % Learning rate
net.trainParam.show = 25;    % Show training updates every 25 iterations
net.trainParam.max_fail = 10; % Maximum validation failures

% Split data into training, validation, and testing sets
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train the network
[net, tr] = train(net, X, Y_onehot); % Use one-hot encoded labels

% Print maximum indices for debugging
fprintf('Maximum train index: %d\n', max(tr.trainInd));
fprintf('Maximum validation index: %d\n', max(tr.valInd));
fprintf('Maximum test index: %d\n', max(tr.testInd));

% Ensure indices are within bounds
numSamples = size(X, 2); % Number of samples
if max(tr.trainInd) > numSamples || max(tr.valInd) > numSamples || max(tr.testInd) > numSamples
    error('Index out of bounds. Check the sizes of training, validation, and test indices.');
end

% Evaluate network performance
% Training, validation, and test errors
trainPerformance = perform(net, Y_onehot(:, tr.trainInd), net(X(:, tr.trainInd)));
valPerformance = perform(net, Y_onehot(:, tr.valInd), net(X(:, tr.valInd)));
testPerformance = perform(net, Y_onehot(:, tr.testInd), net(X(:, tr.testInd)));

% Convert one-hot predictions back to class labels
trainPredictions = vec2ind(net(X(:, tr.trainInd)));
valPredictions = vec2ind(net(X(:, tr.valInd)));
testPredictions = vec2ind(net(X(:, tr.testInd)));

% Convert numeric predictions back to categorical
trainPredLabels = categorical(categories(Y));
valPredLabels = categorical(categories(Y));
testPredLabels = categorical(categories(Y));

% Calculate classification accuracy
trainAccuracy = sum(trainPredLabels(trainPredictions)' == Y(tr.trainInd)) / length(tr.trainInd) * 100;
valAccuracy = sum(valPredLabels(valPredictions)' == Y(tr.valInd)) / length(tr.valInd) * 100;
testAccuracy = sum(testPredLabels(testPredictions)' == Y(tr.testInd)) / length(tr.testInd) * 100;

% Display performance results
fprintf('Training Performance (Cross-Entropy Loss): %.6f\n', trainPerformance);
fprintf('Validation Performance (Cross-Entropy Loss): %.6f\n', valPerformance);
fprintf('Testing Performance (Cross-Entropy Loss): %.6f\n', testPerformance);

fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);
fprintf('Testing Accuracy: %.2f%%\n', testAccuracy);

% 1. Performance plot
figure;
plotperform(tr);
title('Performance');
grid on;

% 2. Confusion matrix for test set only
figure;
confusionchart(Y(tr.testInd), categorical(testPredLabels(testPredictions)), ...
    'Title', 'Confusion Matrix (Test Set)', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
grid on;

% Actual labels
testTargets = vec2ind(Y_onehot(:, tr.testInd));

% ROC Curve for each class
figure;
hold on;
for c = 1:numClasses
    trueBinaryLabels = (testTargets == c); % Binary labels for current class
    predictedScores = net(X(:, tr.testInd)); % Get predicted probabilities
    [rocX, rocY, ~, auc] = perfcurve(trueBinaryLabels, predictedScores(c, :), 1);
    plot(rocX, rocY, 'DisplayName', sprintf('Class %d (AUC = %.2f)', c, auc));
end
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve (One-vs-All)');
legend show;
grid on;
hold off;

% Function to compute precision, recall, and F1-score
function [precision, recall, f1Score] = precisionRecallF1(trueLabels, predictedLabels, numClasses)
    precision = zeros(1, numClasses);
    recall = zeros(1, numClasses);
    f1Score = zeros(1, numClasses);

    for c = 1:numClasses
        tp = sum((predictedLabels == c) & (trueLabels == c));
        fp = sum((predictedLabels == c) & (trueLabels ~= c));
        fn = sum((predictedLabels ~= c) & (trueLabels == c));

        precision(c) = tp / (tp + fp + eps); % Avoid division by zero
        recall(c) = tp / (tp + fn + eps);
        f1Score(c) = 2 * (precision(c) * recall(c)) / (precision(c) + recall(c) + eps);
    end
end

% Save trained model and results
save('models\trained_feedforwardnet_for_Data_Acc_FreqD_FDay.mat', 'net', 'tr', 'trainPerformance', 'valPerformance', 'testPerformance');
