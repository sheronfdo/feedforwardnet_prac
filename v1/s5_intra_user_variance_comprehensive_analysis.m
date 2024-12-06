% Define the user ID
user_id = 'U01'; % Change this for other users as needed

% Define dataset pairs for comparison
datasetPairs = {
    [user_id, '_Acc_FreqD_FDay'], [user_id, '_Acc_FreqD_MDay'];
    [user_id, '_Acc_TimeD_FDay'], [user_id, '_Acc_TimeD_MDay'];
    [user_id, '_Acc_TimeD_FreqD_FDay'], [user_id, '_Acc_TimeD_FreqD_MDay']
};

% Loop through each dataset pair
for pairIdx = 1:size(datasetPairs, 1)
    % Extract field names for current pair
    dataset1_field = datasetPairs{pairIdx, 1};
    dataset2_field = datasetPairs{pairIdx, 2};

    % Check if the datasets exist in all_data
    if ~isfield(all_data, dataset1_field) || ~isfield(all_data, dataset2_field)
        warning('Skipping comparison: %s or %s is missing in all_data.', dataset1_field, dataset2_field);
        continue;
    end

    % Load the datasets
    dataset1 = all_data.(dataset1_field);
    dataset2 = all_data.(dataset2_field);

    % Ensure datasets have the same dimensions
    if ~isequal(size(dataset1), size(dataset2))
        warning('Skipping comparison: %s and %s have mismatched dimensions.', dataset1_field, dataset2_field);
        continue;
    end

    % Initialize figure for this dataset pair
    figure('Name', sprintf('Comparison of %s and %s', dataset1_field, dataset2_field), ...
           'NumberTitle', 'off', 'Position', [100, 100, 1400, 800]);

    % 1. Descriptive Statistics (Subplot 1)
    subplot(2, 3, 1);
    mean1 = mean(dataset1);
    mean2 = mean(dataset2);
    bar([mean1; mean2]', 'grouped');
    legend({sprintf('%s Mean', dataset1_field), sprintf('%s Mean', dataset2_field)}, 'Location', 'northeast');
    title('Mean Comparison');
    xlabel('Columns');
    ylabel('Mean');
    grid on;

    % 2. Difference Matrix (Subplot 2)
    subplot(2, 3, 2);
    diffMatrix = dataset1 - dataset2;
    imagesc(diffMatrix);
    colorbar;
    title('Difference Matrix');
    xlabel('Columns');
    ylabel('Rows');
    set(gca, 'XTick', 1:size(dataset1, 2), 'YTick', 1:size(dataset1, 1));

    % 3. Column-wise Mean Differences (Subplot 3)
    subplot(2, 3, 3);
    meanDiff = mean(diffMatrix);
    bar(meanDiff);
    title('Column-wise Mean Differences');
    xlabel('Columns');
    ylabel('Mean Difference');
    grid on;

    % 4. T-test Results (Subplot 4)
    subplot(2, 3, 4);
    [numRows, numCols] = size(dataset1);
    pValues = zeros(1, numCols);
    for i = 1:numCols
        [~, pValues(i)] = ttest(dataset1(:, i), dataset2(:, i));
    end
    stem(pValues, 'Marker', 'o');
    hold on;
    yline(0.05, 'r--', 'LineWidth', 1.5); % Significance threshold
    hold off;
    title('T-test P-values');
    xlabel('Columns');
    ylabel('P-value');
    grid on;

    % 5. Correlation Comparison (Subplot 5)
    subplot(2, 3, 5);
    corr1 = corr(dataset1);
    corr2 = corr(dataset2);
    corrDiff = corr1 - corr2;
    imagesc(corrDiff);
    colorbar;
    title('Correlation Difference Matrix');
    xlabel('Columns');
    ylabel('Columns');

    % 6. PCA Comparison (Subplot 6)
    subplot(2, 3, 6);
    [coeff1, score1] = pca(dataset1);
    [coeff2, score2] = pca(dataset2);
    scatter(score1(:, 1), score1(:, 2), 'filled', 'DisplayName', sprintf('%s', dataset1_field));
    hold on;
    scatter(score2(:, 1), score2(:, 2), 'filled', 'DisplayName', sprintf('%s', dataset2_field));
    legend;
    title('PCA Comparison (PC1 vs. PC2)');
    xlabel('PC1');
    ylabel('PC2');
    grid on;

    % Adjust layout
    sgtitle(sprintf('Comprehensive Analysis of %s and %s', dataset1_field, dataset2_field));
end
