clc;
clear;

% Load dataset
load('preprocessed_data/mergedData_Acc_TimeD_FDay.mat'); % Replace with your dataset file

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

% Columns to remove (Nan Columns)
columnsToRemove = [67, 68, 69, 77, 78, 79, 88]; 
% Remove specified columns from X 
X(:, columnsToRemove) = [];

if any(isnan(X), 'all') || any(isinf(X), 'all') 
    error('Data contains NaN or infinite values. Clean the data before PCA.'); 
end

% Perform PCA on the features 
[coeff, score, latent, tsquared, explained] = pca(X);

% Display explained variance
disp('Explained Variance by Each Component:'); 
disp(explained);

% Select the number of components to keep (e.g., 95% variance) 
explainedVariance = cumsum(explained); 
numComponents = find(explainedVariance >= 95, 1);

% Reduce the feature set to the selected components 
X_reduced = score(:, 1:numComponents);

fprintf('Original number of features: %d\n', size(X, 2));
fprintf('Reduced number of features: %d\n', size(X_reduced, 2));

% Transpose X and Y to match the expected input format for the neural network
X = X_reduced'; 
Y = Y';

% Convert labels to one-hot encoding for multi-class classification
numClasses = length(categories(Y));
Y_onehot = full(ind2vec(double(Y), numClasses)); % Convert categorical to numeric for one-hot encoding

% Define the Feedforward Neural Network
hiddenLayerSizes = [30, 30]; % Increased number of neurons 
net = feedforwardnet(hiddenLayerSizes, 'trainbr'); % Use Bayesian Regularization training

% Configure training parameters
net.trainParam.epochs = 1000; % Increased epochs for better convergence
net.trainParam.goal = 1e-6;   % Performance goal (MSE)
net.trainParam.lr = 0.01;     % Learning rate
net.trainParam.show = 25;     % Show training updates every 25 iterations
net.trainParam.max_fail = 10; % Maximum validation failures

% Initialize network weights to small random values 
net = init(net);
net.IW{1,1} = randn(size(net.IW{1,1}))*0.01;
net.LW{2,1} = randn(size(net.LW{2,1}))*0.01;
net.b{1} = randn(size(net.b{1}))*0.01;
net.b{2} = randn(size(net.b{2}))*0.01;

% Split data into training, validation, and testing sets
net.divideParam.trainRatio = 0.8; % Use 80% for training
net.divideParam.valRatio = 0.1;   % Use 10% for validation
net.divideParam.testRatio = 0.1;  % Use 10% for testing

% Train the network
[net, tr] = train(net, X, Y_onehot);

% Debug: Check stopping reason
fprintf('Training stopped due to: %s\n', tr.stop);

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

% Save the trained model and results
save('models/trained_feedforwardnet_for_TimeD_FDay.mat', 'net', 'tr', ...
    'trainPerformance', 'valPerformance', 'testPerformance');
