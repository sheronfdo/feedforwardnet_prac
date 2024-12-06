% Define the user IDs
user_ids = {'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10'};

% Loop through each user
for userIdx = 1:length(user_ids)
    user_id = user_ids{userIdx};
    dataset_field = [user_id, '_Acc_FreqD_FDay']; % Example dataset field
    
    % Check if the dataset exists
    if isfield(all_data, dataset_field)
        dataset = all_data.(dataset_field);
        
        % Initialize figure for the current user
        figure('Name', sprintf('User %s: Dataset Analysis', user_id), ...
               'NumberTitle', 'off', 'Position', [100, 100, 1600, 900]);
        
        % Analyze each column one by one
        for columnIdx = 1:size(dataset, 2)
            data_column = dataset(:, columnIdx);

            % Create a subplot for each column analysis
            subplot(ceil(sqrt(size(dataset, 2))), ceil(sqrt(size(dataset, 2))), columnIdx);
            histogram(data_column);
            title(sprintf('User %s - Column %d', user_id, columnIdx));
            xlabel('Value');
            ylabel('Frequency');
            grid on;
        end
        
        % Add a global title for the figure
        sgtitle(sprintf('Dataset Analysis for User %s', user_id));
        
    else
        warning('Skipping %s: %s is missing in all_data.', user_id, dataset_field);
    end
end
