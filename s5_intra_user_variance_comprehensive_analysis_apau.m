% Define user IDs
user_ids = {'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10'};

% Define dataset pairs for each user
datasetPairSuffixes = {
    '_Acc_FreqD_FDay', '_Acc_FreqD_MDay';
    '_Acc_TimeD_FDay', '_Acc_TimeD_MDay';
    '_Acc_TimeD_FreqD_FDay', '_Acc_TimeD_FreqD_MDay'
};

% Number of dataset pairs
numPairs = size(datasetPairSuffixes, 1);

% Loop through each user
for userIdx = 1:length(user_ids)
    user_id = user_ids{userIdx};
    
    % Create a new figure for the current user
    figure('Name', sprintf('Analysis for %s', user_id), ...
           'NumberTitle', 'off', 'Position', [100, 100, 2000, 1500]);

    % Loop through each dataset pair
    for pairIdx = 1:numPairs
        % Generate dataset field names for the current user
        dataset1_field = [user_id, datasetPairSuffixes{pairIdx, 1}];
        dataset2_field = [user_id, datasetPairSuffixes{pairIdx, 2}];

        % Check if the datasets exist in all_data
        if ~isfield(all_data, dataset1_field) || ~isfield(all_data, dataset2_field)
            warning('Skipping comparison for %s: %s or %s is missing in all_data.', user_id, dataset1_field, dataset2_field);
            continue;
        end

        % Load the datasets
        dataset1 = all_data.(dataset1_field);
        dataset2 = all_data.(dataset2_field);

        % Ensure datasets have the same dimensions
        if ~isequal(size(dataset1), size(dataset2))
            warning('Skipping comparison for %s: %s and %s have mismatched dimensions.', user_id, dataset1_field, dataset2_field);
            continue;
        end

        % Base index for subplots (6 subplots per pair)
        baseIdx = (pairIdx - 1) * 6;

        % Plot descriptive statistics
        subplot(numPairs, 6, baseIdx + 1);
        mean1 = mean(dataset1);
        mean2 = mean(dataset2);
        bar([mean1; mean2]', 'grouped');
        legend({sprintf('%s Mean', dataset1_field), sprintf('%s Mean', dataset2_field)}, 'Location', 'northeast');
        title('Mean Comparison');
        xlabel('Columns');
        ylabel('Mean');
        grid on;

        % Plot difference matrix
        subplot(numPairs, 6, baseIdx + 2);
        diffMatrix = dataset1 - dataset2;
        imagesc(diffMatrix);
        colorbar;
        title('Difference Matrix');
        xlabel('Columns');
        ylabel('Rows');
        set(gca, 'XTick', 1:size(dataset1, 2), 'YTick', 1:size(dataset1, 1));

        % Plot column-wise mean differences
        subplot(numPairs, 6, baseIdx + 3);
        meanDiff = mean(diffMatrix);
        bar(meanDiff);
        title('Column-wise Mean Differences');
        xlabel('Columns');
        ylabel('Mean Difference');
        grid on;

        % Plot T-test results
        subplot(numPairs, 6, baseIdx + 4);
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

        % Plot correlation comparison
        subplot(numPairs, 6, baseIdx + 5);
        corr1 = corr(dataset1);
        corr2 = corr(dataset2);
        corrDiff = corr1 - corr2;
        imagesc(corrDiff);
        colorbar;
        title('Correlation Difference Matrix');
        xlabel('Columns');
        ylabel('Columns');

        % Plot PCA comparison
        subplot(numPairs, 6, baseIdx + 6);
        [coeff1, score1] = pca(dataset1);
        [coeff2, score2] = pca(dataset2);
        scatter(score1(:, 1), score1(:, 2), 'filled', 'DisplayName', 'Dataset 1');
        hold on;
        scatter(score2(:, 1), score2(:, 2), 'filled', 'DisplayName', 'Dataset 2');
        legend;
        title('PCA Comparison (PC1 vs. PC2)');
        xlabel('PC1');
        ylabel('PC2');
        grid on;
    end

    % Add a super title for the figure
    sgtitle(sprintf('Comprehensive Analysis for %s', user_id));
end
