% Define the user IDs
user_ids = {'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10'};

% Define cross-dataset pairs for comparison
crossDatasetPairs = {
    '_Acc_FreqD_FDay', '_Acc_TimeD_FDay';
    '_Acc_FreqD_MDay', '_Acc_TimeD_MDay';
    '_Acc_FreqD_FDay', '_Acc_TimeD_FreqD_FDay';
    '_Acc_FreqD_MDay', '_Acc_TimeD_FreqD_MDay'
};

% Loop through each user
for userIdx = 1:length(user_ids)
    user_id = user_ids{userIdx};

    % Initialize figure for the current user
    figure('Name', sprintf('User %s: Cross-Dataset Analysis', user_id), ...
           'NumberTitle', 'off', 'Position', [100, 100, 1600, 900]);

    % Loop through each cross-dataset pair
    for pairIdx = 1:size(crossDatasetPairs, 1)
        % Generate dataset field names for the current pair
        dataset1_field = [user_id, crossDatasetPairs{pairIdx, 1}];
        dataset2_field = [user_id, crossDatasetPairs{pairIdx, 2}];

        % Check if the datasets exist in all_data
        if ~isfield(all_data, dataset1_field) || ~isfield(all_data, dataset2_field)
            warning('Skipping comparison for %s: %s or %s is missing in all_data.', user_id, dataset1_field, dataset2_field);
            continue;
        end

        % Load the datasets
        dataset1 = all_data.(dataset1_field);
        dataset2 = all_data.(dataset2_field);

        % Plot each analysis as a separate subplot
        subplot(size(crossDatasetPairs, 1), 6, (pairIdx - 1) * 6 + 1);
        bar([size(dataset1, 2), size(dataset2, 2)]);
        set(gca, 'XTickLabel', {dataset1_field, dataset2_field});
        title('Number of Features Comparison');
        ylabel('Number of Features');
        grid on;

        subplot(size(crossDatasetPairs, 1), 6, (pairIdx - 1) * 6 + 2);
        mean1 = mean(dataset1, 1);
        mean2 = mean(dataset2, 1);
        std1 = std(dataset1, 0, 1);
        std2 = std(dataset2, 0, 1);

        hold on;
        plot(mean1, 'DisplayName', sprintf('Mean %s', dataset1_field));
        plot(mean2, 'DisplayName', sprintf('Mean %s', dataset2_field));
        errorbar(mean1, std1, 'LineStyle', 'none', 'Color', 'blue', 'DisplayName', 'Std Error Dataset 1');
        errorbar(mean2, std2, 'LineStyle', 'none', 'Color', 'red', 'DisplayName', 'Std Error Dataset 2');
        hold off;

        legend;
        title('Feature-wise Mean and Std Comparison');
        xlabel('Feature Index');
        ylabel('Values');
        grid on;

        subplot(size(crossDatasetPairs, 1), 6, (pairIdx - 1) * 6 + 3);
        corrMatrix = corr(dataset1, dataset2);
        imagesc(corrMatrix);
        colorbar;
        title('Cross-Dataset Correlation');
        xlabel(sprintf('%s Features', dataset2_field));
        ylabel(sprintf('%s Features', dataset1_field));

        subplot(size(crossDatasetPairs, 1), 6, (pairIdx - 1) * 6 + 4);
        minCols = min(size(dataset1, 2), size(dataset2, 2));
        pValues = zeros(1, minCols);
        for i = 1:minCols
            [~, pValues(i)] = ttest2(dataset1(:, i), dataset2(:, i));
        end
        stem(pValues, 'Marker', 'o');
        hold on;
        yline(0.05, 'r--', 'LineWidth', 1.5); % Significance threshold
        hold off;
        title('T-test P-values on Matching Features');
        xlabel('Feature Index');
        ylabel('P-value');
        grid on;

        subplot(size(crossDatasetPairs, 1), 6, (pairIdx - 1) * 6 + 5);
        combinedData = [dataset1, dataset2];
        [coeff, score] = pca(combinedData);
        scatter(score(:, 1), score(:, 2), 'filled');
        title('PCA of Combined Datasets');
        xlabel('PC1');
        ylabel('PC2');
        grid on;

        subplot(size(crossDatasetPairs, 1), 6, (pairIdx - 1) * 6 + 6);
        combinedFeatures = [mean1, mean2];
        imagesc(combinedFeatures);
        colorbar;
        title('Heatmap of Feature Means');
        xlabel('Combined Features');
        ylabel('Datasets');
        set(gca, 'XTick', 1:length(combinedFeatures), 'XTickLabel', [1:length(mean1), 1:length(mean2)]);
        grid on;
    end

    % Add a global title for the figure
    sgtitle(sprintf('Cross-Dataset Analysis for User %s', user_id));
end
