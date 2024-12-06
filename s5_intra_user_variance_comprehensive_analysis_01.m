% Define the user ID and corresponding dataset field names
user_id = 'U01'; % Change this for other users if needed
fields = {'Acc_FreqD_FDay', 'Acc_FreqD_MDay', 'Acc_TimeD_FDay', ...
          'Acc_TimeD_MDay', 'Acc_TimeD_FreqD_FDay', 'Acc_TimeD_FreqD_MDay'};
datasetNames = strcat(user_id, '_', fields);

% Check if all required datasets exist in all_data
missingFields = ~isfield(all_data, datasetNames);
if any(missingFields)
    error('Missing datasets: %s', strjoin(datasetNames(missingFields), ', '));
end

% Load the datasets dynamically
datasets = cellfun(@(field) all_data.(field), datasetNames, 'UniformOutput', false);

% Initialize figure
figure('Name', sprintf('Comprehensive Analysis for %s', user_id), ...
       'NumberTitle', 'off', 'Position', [100, 100, 1600, 900]);

% Subplot 1: Structural Comparison
subplot(2, 3, 1);
sizes = cellfun(@size, datasets, 'UniformOutput', false);
sizes = cell2mat(sizes(:));
bar(sizes(:, 2)); % Plot number of features (columns)
set(gca, 'XTick', 1:numel(fields), 'XTickLabel', fields, 'XTickLabelRotation', 45);
title('Number of Features in Each Dataset');
xlabel('Datasets');
ylabel('Number of Features');
grid on;

% Subplot 2: Descriptive Statistics (Mean & Std) for the first dataset
subplot(2, 3, 2);
meanStats = mean(datasets{1}, 1);
stdStats = std(datasets{1}, 0, 1);

bar([meanStats; stdStats]', 'grouped');
legend({'Mean', 'Std'});
title(sprintf('Descriptive Statistics (%s)', fields{1}));
xlabel('Features');
ylabel('Values');
grid on;

% Subplot 3: Difference Matrix Between FreqD_FDay and FreqD_MDay
subplot(2, 3, 3);
diffMatrix = datasets{1} - datasets{2};
imagesc(diffMatrix);
colorbar;
title('Difference Matrix (FreqD_FDay - FreqD_MDay)');
xlabel('Features');
ylabel('Samples');
grid on;

% Subplot 4: T-test Results Between FreqD_FDay and FreqD_MDay
subplot(2, 3, 4);
[numRows, numCols] = size(datasets{1});
pValues = zeros(1, numCols);
for i = 1:numCols
    [~, pValues(i)] = ttest(datasets{1}(:, i), datasets{2}(:, i));
end
stem(pValues, 'Marker', 'o');
hold on;
yline(0.05, 'r--', 'LineWidth', 1.5); % Significance threshold
hold off;
title('T-test P-values (FreqD_FDay vs. FreqD_MDay)');
xlabel('Features');
ylabel('P-values');
grid on;

% Subplot 5: Correlation Difference Between FreqD_FDay and FreqD_MDay
subplot(2, 3, 5);
corr_FDay = corr(datasets{1});
corr_MDay = corr(datasets{2});
corrDiff = corr_FDay - corr_MDay;
imagesc(corrDiff);
colorbar;
title('Correlation Difference Matrix (FreqD_FDay - FreqD_MDay)');
xlabel('Features');
ylabel('Features');
grid on;

% Subplot 6: PCA Comparison for FreqD_FDay and FreqD_MDay
subplot(2, 3, 6);
[coeffFDay, scoreFDay] = pca(datasets{1});
[coeffMDay, scoreMDay] = pca(datasets{2});
scatter(scoreFDay(:, 1), scoreFDay(:, 2), 'filled', 'DisplayName', 'FreqD_FDay');
hold on;
scatter(scoreMDay(:, 1), scoreMDay(:, 2), 'filled', 'DisplayName', 'FreqD_MDay');
legend;
title('PCA Comparison (FreqD_FDay vs. FreqD_MDay)');
xlabel('PC1');
ylabel('PC2');
grid on;

% Adjust layout for better clarity
sgtitle(sprintf('Comprehensive Analysis of Datasets for User %s', user_id));
