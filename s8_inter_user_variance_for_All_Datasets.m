% User IDs
user_ids = {'U01', 'U02', 'U03','U04', 'U05', 'U06','U07', 'U08', 'U09', 'U10'}; % Add more user IDs as needed

% Feature Domains to Analyze
feature_domains = {'Acc_FreqD_FDay', 'Acc_TimeD_FDay', 'Acc_TimeD_FreqD_FDay','Acc_FreqD_MDay', 'Acc_TimeD_MDay', 'Acc_TimeD_FreqD_MDay'};

% Loop through each feature domain
for domainIdx = 1:length(feature_domains)
    feature_domain = feature_domains{domainIdx};
    
    % Collect datasets for all users for the current domain
    datasets = cell(1, length(user_ids));
    for userIdx = 1:length(user_ids)
        fieldName = [user_ids{userIdx}, '_', feature_domain];
        if ~isfield(all_data, fieldName)
            warning('Dataset %s is missing.', fieldName);
            datasets{userIdx} = [];
        else
            datasets{userIdx} = all_data.(fieldName);
        end
    end

    % Remove empty datasets
    validDatasets = ~cellfun('isempty', datasets);
    datasets = datasets(validDatasets);
    validUserIds = user_ids(validDatasets);

    % Ensure there are at least two users to compare
    if length(datasets) < 2
        warning('Not enough datasets for domain %s to perform inter-user variance analysis.', feature_domain);
        continue;
    end

    % Initialize figure for inter-user variance analysis
    figure('Name', sprintf('Inter-User Variance: %s', feature_domain), ...
           'NumberTitle', 'off', 'Position', [100, 100, 1400, 800]);

    % 1. Dimensional Comparison (Subplot 1)
    subplot(2, 3, 1);
    featureCounts = cellfun(@(x) size(x, 2), datasets);
    bar(featureCounts);
    set(gca, 'XTickLabel', validUserIds, 'XTickLabelRotation', 45);
    title('Number of Features for Each User');
    xlabel('Users');
    ylabel('Number of Features');
    grid on;

    % 2. Mean and Std Deviation Across Users (Subplot 2)
    subplot(2, 3, 2);
    means = cellfun(@(x) mean(mean(x, 1)), datasets);
    stds = cellfun(@(x) mean(std(x, 1)), datasets);
    bar([means; stds]', 'grouped');
    legend({'Mean', 'Std'}, 'Location', 'northeast');
    set(gca, 'XTickLabel', validUserIds, 'XTickLabelRotation', 45);
    title('Mean and Std Across Users');
    xlabel('Users');
    ylabel('Values');
    grid on;

    % 3. Pairwise Correlation Comparison (Subplot 3)
    subplot(2, 3, 3);
    userPairwiseCorr = zeros(length(datasets));
    for i = 1:length(datasets)
        for j = 1:length(datasets)
            if i ~= j
                userPairwiseCorr(i, j) = mean(diag(corr(datasets{i}, datasets{j})));
            end
        end
    end
    imagesc(userPairwiseCorr);
    colorbar;
    title('Pairwise Correlation Between Users');
    set(gca, 'XTick', 1:length(validUserIds), 'XTickLabel', validUserIds, ...
             'YTick', 1:length(validUserIds), 'YTickLabel', validUserIds);
    xlabel('Users');
    ylabel('Users');
    grid on;

    % 4. T-test for Matching Features Across Users (Subplot 4)
    subplot(2, 3, 4);
    numFeatures = min(cellfun(@(x) size(x, 2), datasets));
    pValues = zeros(length(datasets), length(datasets), numFeatures);
    for i = 1:length(datasets)
        for j = i+1:length(datasets)
            for k = 1:numFeatures
                [~, pValues(i, j, k)] = ttest2(datasets{i}(:, k), datasets{j}(:, k));
            end
        end
    end
    avgPValues = squeeze(mean(pValues, 3)); % Average p-values across features
    imagesc(avgPValues);
    colorbar;
    title('Average T-test P-values Between Users');
    set(gca, 'XTick', 1:length(validUserIds), 'XTickLabel', validUserIds, ...
             'YTick', 1:length(validUserIds), 'YTickLabel', validUserIds);
    xlabel('Users');
    ylabel('Users');
    grid on;

    % 5. PCA of All Users (Subplot 5)
    subplot(2, 3, 5);
    combinedData = cell2mat(datasets');
    [coeff, score, ~, ~, explained] = pca(combinedData);
    gscatter(score(:, 1), score(:, 2), repelem(1:length(datasets), cellfun(@(x) size(x, 1), datasets)), ...
             lines(length(datasets)), 'o', 8);
    legend(validUserIds, 'Location', 'best');
    title('PCA Across Users');
    xlabel(sprintf('PC1 (%.2f%%)', explained(1)));
    ylabel(sprintf('PC2 (%.2f%%)', explained(2)));
    grid on;

    % 6. Heatmap of Feature Means for All Users (Subplot 6)
    subplot(2, 3, 6);
    featureMeans = cellfun(@(x) mean(x, 1), datasets, 'UniformOutput', false);
    featureMeans = cell2mat(featureMeans');
    imagesc(featureMeans);
    colorbar;
    title('Feature Means Across Users');
    xlabel('Features');
    ylabel('Users');
    set(gca, 'YTick', 1:length(validUserIds), 'YTickLabel', validUserIds);
    grid on;

    % Adjust layout
    sgtitle(sprintf('Inter-User Variance Analysis: %s', feature_domain));
end
