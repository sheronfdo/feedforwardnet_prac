clc; 
clear; 

% Load dataset 
load('preprocessed_data/mergedData_Acc_FreqD_FDay.mat'); 

% Separate features and labels 
X = mergedData{:, 1:end-1}; % All columns except the last one are features 
userIDs = mergedData.UserID; % The last column is the user ID 

% Convert user IDs to numeric labels 
Y = grp2idx(userIDs); % Convert user IDs to numeric labels 

% Normalize the features for better training performance
X = normalize(X); 

% Ensure that X and Y have the same number of samples 
if size(X, 1) ~= length(Y) 
    error('The number of samples in X and Y do not match.'); 
end 

% Convert X to an array and transpose it to match the expected input format for the neural network
X = X'; 
Y = Y';

% Define the Feedforward Neural Network
hiddenLayerSizes = [10, 10]; % Example: 2 hidden layers with 10 neurons each
net = feedforwardnet(hiddenLayerSizes);

% Configure training parameters
net.trainParam.epochs = 100; % Increased epochs for better convergence
net.trainParam.goal = 1e-6;   % Performance goal (MSE)
net.trainParam.lr = 0.01;     % Learning rate
net.trainParam.show = 25;     % Show training updates every 25 iterations
net.trainParam.max_fail = 10; % Maximum validation failures

% Split data into training, validation, and testing sets
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train the network
[net, tr] = train(net, X, Y); % Y is a row vector

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

% Calculate classification accuracy (for classification problems)
trainPredictions = round(net(X(:, tr.trainInd)));
trainAccuracy = sum(trainPredictions == Y(tr.trainInd)) / length(tr.trainInd) * 100;

valPredictions = round(net(X(:, tr.valInd)));
valAccuracy = sum(valPredictions == Y(tr.valInd)) / length(tr.valInd) * 100;

testPredictions = round(net(X(:, tr.testInd)));
testAccuracy = sum(testPredictions == Y(tr.testInd)) / length(tr.testInd) * 100;

% Calculate regression metrics (for regression problems)
rmse = sqrt(mean((Y - net(X)).^2));
r_squared = 1 - sum((Y - net(X)).^2) / sum((Y - mean(Y)).^2);

% Display performance results
fprintf('Training Performance (MSE): %.6f\n', trainPerformance);
fprintf('Validation Performance (MSE): %.6f\n', valPerformance);
fprintf('Testing Performance (MSE): %.6f\n', testPerformance);

fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);
fprintf('Testing Accuracy: %.2f%%\n', testAccuracy);

fprintf('Root Mean Squared Error (RMSE): %.2f\n', rmse);
fprintf('R^2 (Coefficient of Determination): %.2f\n', r_squared);

% Add annotations to plots
annotationText = sprintf(['Training Performance (MSE): %.6f\nValidation Performance (MSE): %.6f\nTesting Performance (MSE): %.6f\n\n' ... 
    'Training Accuracy: %.2f%%\nValidation Accuracy: %.2f%%\nTesting Accuracy: %.2f%%\n\n' ...
    'Root Mean Squared Error (RMSE): %.2f\nR^2 (Coefficient of Determination): %.2f'], ...
    trainPerformance, valPerformance, testPerformance, trainAccuracy, valAccuracy, testAccuracy, rmse, r_squared);

% 1. Performance plot
figure;
plotperform(tr);
title('Performance');
grid on;
dim = [0.2 0.5 0.3 0.3];
annotation('textbox', dim, 'String', annotationText, 'FitBoxToText','on');

% 2. Regression plots
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

% 3. Confusion matrix (for classification problems)
% Predict class labels using the network
predictedLabels = round(net(X)); % Use `round` for binary classification or adjust for multi-class
actualLabels = Y; % Ensure labels are compatible

% Plot confusion matrix
figure;
confusionchart(actualLabels, predictedLabels, ...
    'Title', 'Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
grid on;

% 4. Actual vs. Predicted values plot (for regression problems)
figure;
plot(Y, net(X), 'o');
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs Predicted Values');
grid on;

% Save trained model and results
save('models\trained_feedforwardnet_for_Data_Acc_FreqD_FDay.mat', 'net', 'tr', 'trainPerformance', 'valPerformance', 'testPerformance');
