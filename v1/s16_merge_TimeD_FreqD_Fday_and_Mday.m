% Load the datasets
data1 = load('preprocessed_data/mergedData_Acc_TimeD_FreqD_FDay.mat');
data2 = load('preprocessed_data/mergedData_Acc_TimeD_FreqD_MDay.mat');

% Assuming the datasets are stored in variables 'data1' and 'data2'
mergedData = [data1.mergedData; data2.mergedData];  % Concatenate the datasets

% Shuffle the merged dataset
mergedData = mergedData(randperm(size(mergedData, 1)), :);

% Save the shuffled dataset to a new .mat file
save('preprocessed_data/merge_TimeD_FreqD_Fday_and_Mday.mat', 'mergedData');
