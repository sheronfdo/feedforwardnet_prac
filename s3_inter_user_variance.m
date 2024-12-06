% Assume 'all_data' contains all datasets, already loaded.
% Define the list of user IDs you want to analyze
user_ids = {'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10'};

% Create a figure to hold all 10 plots
figure;
num_users = length(user_ids);

% Loop through each user and create a subplot for each
for j = 1:num_users
    user_id = user_ids{j};
    
    % Get datasets for the current user
    user_fields = fieldnames(all_data); % Get all dataset names
    user_datasets = user_fields(contains(user_fields, user_id)); % Filter datasets for the current user
    
    % Initialize combined data and feature names
    combined_data = [];
    common_columns = [];
    
    % Process each dataset for the current user
    for i = 1:length(user_datasets)
        current_data = all_data.(user_datasets{i}); % Load current dataset
    
         % Debug: Print dataset name and size
        fprintf('Processing dataset: %s, Size: [%d, %d]\n', user_datasets{i}, size(current_data, 1), size(current_data, 2));
    
        if isempty(combined_data)
            % First dataset: Initialize combined data
            combined_data = current_data;
            common_columns = 1:size(current_data, 2);
        else
            % Align dimensions by using common features
            min_columns = min(size(combined_data, 2), size(current_data, 2));
            combined_data = [combined_data(:, 1:min_columns); current_data(:, 1:min_columns)];
        end
    end
    
    % Check if data exists for User
    if isempty(combined_data)
        error('No valid data found for User: %s', user_id);
    end

    % Compute feature-wise variance
    feature_variances = var(combined_data, 0, 1); % Variance along rows

    % Display Intra-Feature Variance for User 01
    fprintf('Intra-Feature Variances for User %s:\n', user_id);
    disp(feature_variances);
    
    % Create a subplot for the current user
    subplot(2, 5, j);  % 2 rows and 5 columns of subplots
    bar(feature_variances);
    xlabel('Feature Index');
    ylabel('Variance');
    title(sprintf('User %s', user_id));
    grid on;
end

% Adjust layout for better spacing
sgtitle('Inter-Feature Variances for 10 Users');  % Add a common title for the entire figure
