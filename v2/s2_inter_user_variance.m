% Initialize a structure for storing intra-user variances
intra_user_variances = struct();

% Loop through all users (assuming 10 users U01 to U10)
for user_id = 1:10
    % Generate the user prefix (e.g., U01, U02)
    user_prefix = sprintf('U%02d', user_id);
    
    % Extract all datasets for this user
    user_datasets = fieldnames(all_data); % Assuming all_data contains loaded datasets
    user_datasets = user_datasets(contains(user_datasets, user_prefix));
    
    % Combine all datasets into one matrix (rows: samples, cols: features)
    combined_data = [];
    for j = 1:length(user_datasets)
        % Standardize the number of columns for concatenation
        dataset = all_data.(user_datasets{j});
        if isempty(combined_data)
            % Set the initial reference column size
            num_columns = size(dataset, 2);
        elseif size(dataset, 2) ~= num_columns
            % Adjust the dataset to match the column size
            min_columns = min(size(dataset, 2), num_columns);
            dataset = dataset(:, 1:min_columns); % Trim extra columns
            combined_data = combined_data(:, 1:min_columns); % Align combined data
        end
        
        combined_data = [combined_data; dataset];
    end
    
    % Compute feature-wise variance
    feature_variance = var(combined_data, 0, 1); % Variance across rows (samples)
    
    % Store results in the structure
    intra_user_variances.(user_prefix).feature_variance = feature_variance;
    intra_user_variances.(user_prefix).mean_variance = mean(feature_variance); % Mean variance for all features
end

% Display mean variances for all users
disp('Intra-User Mean Variance:');
user_ids = fieldnames(intra_user_variances);
for i = 1:length(user_ids)
    fprintf('%s: %.4f\n', user_ids{i}, intra_user_variances.(user_ids{i}).mean_variance);
end

% Prepare data for plotting
mean_variances = [];
for i = 1:length(user_ids)
    mean_variances = [mean_variances; intra_user_variances.(user_ids{i}).mean_variance];
end

% Bar plot for intra-user variances
figure;
bar(mean_variances);
xticks(1:length(user_ids));
xticklabels(user_ids);
xlabel('User ID');
ylabel('Mean Variance');
title('Inter-User Variance');

% Advanced Metrics (Optional)
disp('Feature-wise Standard Deviations and Coefficient of Variation:');
for user_id = 1:10
    user_prefix = sprintf('U%02d', user_id);
    combined_data = [];
    
    % Extract and combine datasets
    user_datasets = fieldnames(all_data);
    user_datasets = user_datasets(contains(user_datasets, user_prefix));
    for j = 1:length(user_datasets)
        dataset = all_data.(user_datasets{j});
        if isempty(combined_data)
            num_columns = size(dataset, 2);
        elseif size(dataset, 2) ~= num_columns
            min_columns = min(size(dataset, 2), num_columns);
            dataset = dataset(:, 1:min_columns);
            combined_data = combined_data(:, 1:min_columns);
        end
        combined_data = [combined_data; dataset];
    end
    
    % Compute standard deviation and coefficient of variation
    std_variance = std(combined_data, 0, 1); % Feature-wise standard deviation
    mean_combined = mean(combined_data, 1);
    cv = std_variance ./ mean_combined; % Coefficient of Variation
    
    % Display results
    fprintf('%s Standard Deviation (Mean): %.4f\n', user_prefix, mean(std_variance));
    fprintf('%s Coefficient of Variation (Mean): %.4f\n', user_prefix, mean(cv));
end
