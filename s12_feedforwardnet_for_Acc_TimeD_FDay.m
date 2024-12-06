clc;
clear;

% Load dataset
load('preprocessed_data/mergedData_Acc_TimeD_FDay.mat'); % Replace with your dataset file

% Separate features and labels
X = mergedData(:, 1:end-1); % All columns except the last one are features
Y = mergedData(:, end);     % The last column is the label

% Normalize the features for better training performance
X = normalize(X);

% Ensure that X and Y have the same number of samples
if size(X, 1) ~= length(Y)
    error('The number of samples in X and Y do not match.');
end

% Columns to remove (Nan Colomns)
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

% Define the Feedforward Neural Network
hiddenLayerSizes = [30, 30]; % Increased number of neurons 
net = feedforwardnet(hiddenLayerSizes, 'trainbr'); % Use Bayesian Regularization training 
% Initialize network weights to small random values 
net = init(net);
net.IW{1,1} = randn(size(net.IW{1,1}))*0.01;
net.LW{2,1} = randn(size(net.LW{2,1}))*0.01;
net.b{1} = randn(size(net.b{1}))*0.01;
net.b{2} = randn(size(net.b{2}))*0.01

% Split data into training, validation, and testing sets
net.divideParam.trainRatio = 0.8; % Use 80% for training
net.divideParam.valRatio = 0.1;   % Use 10% for validation
net.divideParam.testRatio = 0.1;  % Use 10% for testing

% Train the network
[net, tr] = train(net, X, Y);

% Debug: Check stopping reason
fprintf('Training stopped due to: %s\n', tr.stop);

% Evaluate network performance
% Training, validation, and test errors
trainPerformance = perform(net, Y(tr.trainInd), net(X(:, tr.trainInd)));
valPerformance = perform(net, Y(tr.valInd), net(X(:, tr.valInd)));
testPerformance = perform(net, Y(tr.testInd), net(X(:, tr.testInd)));

% Classification accuracy (for classification problems)
trainPredictions = round(net(X(:, tr.trainInd)));
trainAccuracy = sum(trainPredictions == Y(tr.trainInd)) / length(tr.trainInd) * 100;

valPredictions = round(net(X(:, tr.valInd)));
valAccuracy = sum(valPredictions == Y(tr.valInd)) / length(tr.valInd) * 100;

testPredictions = round(net(X(:, tr.testInd)));
testAccuracy = sum(testPredictions == Y(tr.testInd)) / length(tr.testInd) * 100;

% Regression metrics (for regression problems)
rmse = sqrt(mean((Y - net(X)).^2)); % Root Mean Squared Error
r_squared = 1 - sum((Y - net(X)).^2) / sum((Y - mean(Y)).^2); % R^2 metric

% Display performance results
fprintf('Training Performance (MSE): %.6f\n', trainPerformance);
fprintf('Validation Performance (MSE): %.6f\n', valPerformance);
fprintf('Testing Performance (MSE): %.6f\n', testPerformance);
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);
fprintf('Testing Accuracy: %.2f%%\n', testAccuracy);
fprintf('Root Mean Squared Error (RMSE): %.2f\n', rmse);
fprintf('R^2 (Coefficient of Determination): %.2f\n', r_squared);

% 1. Plot Training Performance
figure;
plotperform(tr);
title('Performance');
grid on;

% 2. Regression Plots
figure;
plotregression(Y(tr.trainInd), net(X(:, tr.trainInd)), 'Training');
title('Regression: Training');
grid on;

figure;
plotregression(Y(tr.valInd), net(X(:, tr.valInd)), 'Validation');
title('Regression: Validation');
grid on;

figure;
plotregression(Y(tr.testInd), net(X(:, tr.testInd)), 'Testing');
title('Regression: Testing');
grid on;

% 3. Confusion Matrix (for classification problems)
figure;
predictedLabels = round(net(X)); % Use `round` for binary classification
confusionchart(Y, predictedLabels, ...
    'Title', 'Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
grid on;

% 4. Actual vs. Predicted Values Plot (for regression problems)
figure;
plot(Y, net(X), 'o');
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs Predicted Values');
grid on;

% Save the trained model and results
save('models/trained_feedforwardnet_for_TimeD_FDay.mat', 'net', 'tr', ...
    'trainPerformance', 'valPerformance', 'testPerformance');
