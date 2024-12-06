% Define user IDs and feature domains
user_ids = {'U01', 'U02', 'U03','U04', 'U05', 'U06','U07', 'U08', 'U09', 'U10'}; % Add more user IDs as needed
feature_domains = {'Acc_FreqD_FDay', 'Acc_TimeD_FDay', 'Acc_TimeD_FreqD_FDay', 'Acc_FreqD_MDay', 'Acc_TimeD_MDay', 'Acc_TimeD_FreqD_MDay'};

% Iterate through each feature domain
for featureIdx = 1:length(feature_domains)
    feature_domain = feature_domains{featureIdx};
    
    % Initialize an empty array for merged data
    mergedData = [];
    mergedLabels = [];

    % Iterate through each user
    for userIdx = 1:length(user_ids)
        user_id = user_ids{userIdx};
        datasetField = [user_id, '_', feature_domain];      % e.g., U01_Acc_FreqD_FDay
        labelField = [datasetField, '_Labels'];            % e.g., U01_Acc_FreqD_FDay_Labels
        
        % Check if the dataset exist
        if isfield(all_data, datasetField)
            userData = all_data.(datasetField);
            userLabels = userIdx;
            
            % Skip if dataset or labels are empty
            if isempty(userData) || isempty(userLabels)
                fprintf('Warning: Dataset or labels for %s are empty. Skipping.\n', datasetField);
                continue;
            end
            
            % Add a user identifier column (e.g., user index or user ID)
            userColumn = repmat(userIdx, size(userData, 1), 1); % Use user index (1, 2, 3, ...)
            
            % Append to merged dataset
            mergedData = [mergedData; userData, userColumn]; % Append data with user column
            mergedLabels = [mergedLabels; userLabels];       % Append labels
        else
            fprintf('Warning: Dataset or labels missing for %s.\n', datasetField);
        end
    end

    % Display the size of the merged dataset
    fprintf('Merged dataset size for %s: %d samples, %d features (including user ID).\n', ...
            feature_domain, size(mergedData, 1), size(mergedData, 2));
    fprintf('Merged labels size for %s: %d samples.\n', feature_domain, size(mergedLabels, 1));

    % Save the merged data and labels to a .mat file
    save(['preprocessed_data\mergedData_', feature_domain, '.mat'], 'mergedData','mergedLabels'); % Save to .mat file
end
