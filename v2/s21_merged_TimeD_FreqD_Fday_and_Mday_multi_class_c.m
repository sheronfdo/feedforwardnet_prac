clc;
clear;

% Load dataset
load('preprocessed_data/merge_TimeD_FreqD_Fday_and_Mday.mat');

% Separate features and labels
X = mergedData{:, 1:end-1}; % All columns except the last one are features
userIDs = mergedData.UserID; % The last column is the user ID

% Convert user IDs to categorical labels for multi-class classification
Y = categorical(userIDs); 

% Normalize the features for better training performance
X = normalize(X);

% Ensure that X and Y have the same number of samples
if size(X, 1) ~= length(Y)
    error('The number of samples in X and Y do not match.');
end

% Check for NaN values in the entire dataset 
nanMask = isnan(X); % Identify columns with NaN values 
nanColumns = any(nanMask, 1);
fprintf('Columns with NaN values: %s\n', mat2str(find(nanColumns)));

% Columns to remove 
columnsToRemove = [111, 121]; 
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
hiddenLayerSizes = [30, 30]; % Example: 2 hidden layers with 30 neurons each
net = feedforwardnet(hiddenLayerSizes, 'trainbr'); % Use Bayesian Regularization training

% Configure the network for multi-class classification
net.layers{end}.transferFcn = 'softmax'; % Use softmax for multi-class classification
net.performFcn = 'crossentropy';         % Use cross-entropy loss for classification

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
trainPredLabels = categories(Y);
valPredLabels = categories(Y);
testPredLabels = categories(Y);

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
confusionchart(categorical(Y(tr.testInd)), categorical(testPredLabels(testPredictions)), ...
    'Title', 'Confusion Matrix (Test Set)', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
grid on;

% Precision, Recall, and F1-score
[precision, recall, f1Score] = precisionRecallF1(testTargets, testPredictions, numClasses);
fprintf('Class-wise Precision: %s\n', mat2str(precision));
fprintf('Class-wise Recall: %s\n', mat2str(recall));
fprintf('Class-wise F1-Score: %s\n', mat2str(f1Score));

% ROC Curve
[rocX, rocY, rocT, auc] = perfcurve(double(Y(tr.testInd)), ...
    net(X(:, tr.testInd)), 1:numClasses);

% Plot ROC Curve
figure;
plot(rocX, rocY);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');

% Save the trained model and results
save('models/merged_TimeD_FreqD_Fday_and_Mday_model.mat', 'net', 'tr', ...
    'trainPerformance', 'valPerformance', 'testPerformance');
