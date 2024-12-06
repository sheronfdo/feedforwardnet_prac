% Define user IDs and datasets
user_ids = {'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10'};
datasets_template = { 
    '_Acc_FreqD_FDay', 'Frequency domain-based (Same-day)';
    '_Acc_TimeD_FDay', 'Time domain-based (Same-day)';
    '_Acc_TimeD_FreqD_FDay', 'Combined domain (Same-day)';
    '_Acc_FreqD_MDay', 'Frequency domain-based (Cross-day)';
    '_Acc_TimeD_MDay', 'Time domain-based (Cross-day)';
    '_Acc_TimeD_FreqD_MDay', 'Combined domain (Cross-day)';
};

% Target number of features for alignment
target_features = 131; % Choose the maximum or desired feature size

% Loop through each user
for user_idx = 1:length(user_ids)
    user_id = user_ids{user_idx};
    
    % Initialize storage for aligned variances
    aligned_variances = [];
    labels = {};
    
    % Loop through each dataset template
    for i = 1:size(datasets_template, 1)
        dataset_name = [user_id, datasets_template{i, 1}];
        dataset_label = datasets_template{i, 2};
        
        % Check if dataset exists in all_data
        if isfield(all_data, dataset_name)
            data = all_data.(dataset_name);
            num_features = size(data, 2);
            
            % Resample or interpolate to target size
            new_data = zeros(size(data, 1), target_features);
            for j = 1:size(data, 1)
                new_data(j, :) = interp1(1:num_features, data(j, :), linspace(1, num_features, target_features), 'linear');
            end
            
            % Compute variance
            feature_variances = var(new_data, 0, 1);
            
            % Store results
            aligned_variances = [aligned_variances; feature_variances];
            labels{i} = dataset_label;
        else
            fprintf('Dataset %s not found. Skipping...\n', dataset_name);
        end
    end
    
    % Check if any data was processed for the current user
    if isempty(aligned_variances)
        fprintf('No valid data found for User %s. Skipping...\n', user_id);
        continue;
    end
    
    % Compute correlation matrix
    corr_matrix = corr(aligned_variances');
    
    % Create a figure for the current user
    figure;
    
    % **BAR CHART: Aligned Variances**
    subplot(2, 2, 1); % Top-left
    bar(aligned_variances', 'grouped');
    xlabel('Aligned Feature Index');
    ylabel('Variance');
    title(sprintf('Aligned Variances (Bar Chart) - %s', user_id));
    legend(labels, 'Location', 'northeastoutside');
    grid on;
    
    % **HEATMAP: Feature Variances Across Datasets**
    subplot(2, 2, 2); % Top-right
    imagesc(aligned_variances); % Heatmap of variances
    colorbar;
    xlabel('Aligned Feature Index');
    ylabel('Dataset');
    yticks(1:size(labels, 2));
    yticklabels(labels);
    title(sprintf('Feature Variance Heatmap - %s', user_id));
    grid on;
    
    % **CORRELATION MATRIX HEATMAP**
    subplot(2, 2, [3, 4]); % Bottom (spans two columns)
    imagesc(corr_matrix); % Correlation matrix
    colorbar;
    xlabel('Dataset Index');
    ylabel('Dataset Index');
    xticks(1:size(labels, 2));
    yticks(1:size(labels, 2));
    xticklabels(labels);
    yticklabels(labels);
    title(sprintf('Correlation of Feature Variances - %s', user_id));
    grid on;
    
    % Add an overall title for the figure
    sgtitle(sprintf('Variance and Correlation Analysis for %s', user_id));
end
