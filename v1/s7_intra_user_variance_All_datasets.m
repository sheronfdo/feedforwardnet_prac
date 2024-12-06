% Define the user IDs
user_ids = {'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10'};

% Define the dataset fields
dataset_fields = {'_Acc_FreqD_FDay', '_Acc_TimeD_FDay', '_Acc_FreqD_MDay', '_Acc_TimeD_MDay', ...
                  '_Acc_TimeD_FreqD_FDay', '_Acc_TimeD_FreqD_MDay'};

% Loop through each user
for userIdx = 1:length(user_ids)
    user_id = user_ids{userIdx};
    
    % Loop through each dataset field
    for datasetIdx = 1:length(dataset_fields)
        dataset_field = [user_id, dataset_fields{datasetIdx}];
        
        % Check if the dataset exists
        if isfield(all_data, dataset_field)
            dataset = all_data.(dataset_field);
            
            % Initialize figure for the current dataset
            figure('Name', sprintf('User %s: %s Analysis', user_id, dataset_fields{datasetIdx}), ...
                   'NumberTitle', 'off', 'Position', [100, 100, 1600, 900]);
            
            % Analyze each column one by one
            for columnIdx = 1:size(dataset, 2)
                data_column = dataset(:, columnIdx);

                % Create a subplot for each column analysis
                subplot(ceil(sqrt(size(dataset, 2))), ceil(sqrt(size(dataset, 2))), columnIdx);
                histogram(data_column);
                title(sprintf('User %s - %s - Column %d', user_id, dataset_fields{datasetIdx}, columnIdx));
                xlabel('Value');
                ylabel('Frequency');
                grid on;
            end
            
            % Add a global title for the figure
            sgtitle(sprintf('Dataset Analysis for User %s: %s', user_id, dataset_fields{datasetIdx}));
            
        else
            warning('Skipping %s: %s is missing in all_data.', user_id, dataset_field);
        end
    end
end
