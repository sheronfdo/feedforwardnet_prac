% Load dataset
load('preprocessed_data/merge_TimeD_FreqD_Fday_and_Mday.mat');

% Define user IDs and feature domains
user_ids = {'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10'}; % Add more user IDs as needed
feature_domains = {'Acc_FreqD_FDay', 'Acc_TimeD_FDay', 'Acc_TimeD_FreqD_FDay', 'Acc_FreqD_MDay', 'Acc_TimeD_MDay', 'Acc_TimeD_FreqD_MDay'};

% Iterate through each feature domain
for featureIdx = 1:length(feature_domains)
    feature_domain = feature_domains{featureIdx};
    
    % Initialize an empty array for merged data and a cell array for user IDs
    mergedData = [];
    mergedLabels = [];
    mergedUserIDs = {};

    % Iterate through each user
    for userIdx = 1:length(user_ids)
        user_id = user_ids{userIdx};
        datasetField = [user_id, '_', feature_domain];      % e.g., U01_Acc_FreqD_FDay
        labelField = [datasetField, '_Labels'];            % e.g., U01_Acc_FreqD_FDay_Labels
        
        % Check if the dataset exists
        if isfield(all_data, datasetField)
            userData = all_data.(datasetField);
            userLabels = repmat(userIdx, size(userData, 1), 1); % Assign user index as label
            
            % Skip if dataset or labels are empty
            if isempty(userData) || isempty(userLabels)
                fprintf('Warning: Dataset or labels for %s are empty. Skipping.\n', datasetField);
                continue;
            end
            
            % Add the user ID to the user IDs array
            userIDs = repmat({user_id}, size(userData, 1), 1); % Use user ID as cell array
            
            % Append to merged dataset
            mergedData = [mergedData; userData];      % Append data
            mergedLabels = [mergedLabels; userLabels]; % Append labels
            mergedUserIDs = [mergedUserIDs; userIDs]; % Append user IDs
        else
            fprintf('Warning: Dataset or labels missing for %s.\n', datasetField);
        end
    end

    % Convert mergedData to a table and add feature names
    numFeatures = size(mergedData, 2);
    featureNames = arrayfun(@(x) sprintf('Feature%02d', x), 1:numFeatures, 'UniformOutput', false);
    mergedTable = array2table(mergedData, 'VariableNames', featureNames);
    
    % Add user IDs as the last column of the table
    mergedTable.UserID = mergedUserIDs;

    % Display the size of the merged dataset
    fprintf('Merged dataset size for %s: %d samples, %d features (including user ID).\n', ...
            feature_domain, size(mergedTable, 1), size(mergedTable, 2));
    fprintf('Merged labels size for %s: %d samples.\n', feature_domain, size(mergedLabels, 1));

    mergedData = mergedTable;

    % Save the merged data, labels, and user IDs to a .mat file
    save(['preprocessed_data\mergedData_', feature_domain, '.mat'], 'mergedData');
end
