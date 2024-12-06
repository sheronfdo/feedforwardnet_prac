clc;
clear;

% Specify the folder containing the datasets
folder_path = 'D:\machine learning project SELF\matlab ml\Project\CW-Data'; % Replace with your folder's path

% Get a list of all .mat files in the folder
files = dir(fullfile(folder_path, '*.mat'));

% Initialize a structure to store the datasets
all_data = struct();

% Loop through each file
for i = 1:length(files)
    % Load the file
    file_name = files(i).name;
    file_path = fullfile(folder_path, file_name);
    data = load(file_path);
    
    % Determine the variable name (Acc_FD_Feat_Vec, Acc_TD_Feat_Vec, etc.)
    if isfield(data, 'Acc_FD_Feat_Vec')
        var_data = data.Acc_FD_Feat_Vec;
    elseif isfield(data, 'Acc_TD_Feat_Vec')
        var_data = data.Acc_TD_Feat_Vec;
    elseif isfield(data, 'Acc_TDFD_Feat_Vec')
        var_data = data.Acc_TDFD_Feat_Vec;
    else
        error('Unknown variable in file: %s', file_name);
    end
    
    % Store the data in a structure with a meaningful name
    [~, var_name, ~] = fileparts(file_name); % Extract name without extension
    all_data.(var_name) = var_data; % Store data in the structure
end

% List all loaded dataset names
disp(fieldnames(all_data));

