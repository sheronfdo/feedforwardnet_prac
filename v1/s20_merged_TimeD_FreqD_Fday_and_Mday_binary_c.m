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

% Check for NaN values in the entire dataset 
nanMask = isnan(X); % Identify columns with NaN values 
nanColumns = any(nanMask, 1);
fprintf('Columns with NaN values: %s\n', mat2str(find(nanColumns)));

% Columns to remove 
columnsToRemove = [ 111, 121]; 
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
numComponents = find(explainedVariance >= 80, 1);

% Reduce the feature set to the selected components 
X_reduced = score(:, 1:numComponents);

fprintf('Original number of features: %d\n', size(X, 2));
fprintf('Reduced number of features: %d\n', size(X_reduced, 2));

% Transpose X and Y to match the expected input format for the neural network
X = X_reduced'; 
Y = Y';

% Define the Feedforward Neural Network
hiddenLayerSizes = [50, 100]; % Increased number of neurons 
net = feedforwardnet(hiddenLayerSizes, 'trainbr'); 

% Configure the network for binary classification 
net.layers{end}.transferFcn = 'logsig'; % Sigmoid activation for output layer 
net.performFcn = 'crossentropy'; % Cross-entropy loss function

% Configure training parameters
net.trainParam.epochs = 1000; % Increased epochs for better convergence
net.trainParam.goal = 1e-6;   % Performance goal (MSE)
net.trainParam.lr = 0.01;     % Learning rate
net.trainParam.show = 25;     % Show training updates every 25 iterations
net.trainParam.max_fail = 10; % Maximum validation failures

% Initialize network weights to small random values 
% net = init(net);

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

% AUC for test set 
[~,~,~,AUC] = perfcurve(Y(tr.testInd)', net(X(:, tr.testInd)), 1);

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
fprintf('AUC (Test Set): %.2f\n', AUC);

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

% ROC Curve 
figure; 
[Xroc, Yroc, Troc, AUC] = perfcurve(Y', net(X), 1); 
plot(Xroc, Yroc); 
xlabel('False Positive Rate'); 
ylabel('True Positive Rate'); 
title('ROC Curve'); 
grid on;

% Save the trained model and results
save('models/merged_TimeD_FreqD_Fday_and_Mday_model.mat', 'net', 'tr', ...
    'trainPerformance', 'valPerformance', 'testPerformance');
