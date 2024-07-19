# Reference : https://github.com/mdabashar/TAnoGAN/blob/master/nab_dataset.py (labeling anomalies)
import json
import pandas as pd
import numpy as np
import os
import csv
import numpy as np
import matplotlib.pylab as plt


# Data loading and compiling with labels
    
def load_data_with_labels(labels_filename, file_name):
    # Load labels
    with open(labels_filename, 'r') as labels_file:
        labels = json.load(labels_file)

    # Read data from the CSV file
    data_frame = pd.read_csv(f'datasets/{file_name}', parse_dates=True)
    # Extract features from timestamp column
    data_frame['timestamp'] = pd.to_datetime(data_frame['timestamp'])
    data_frame['hour_of_day'] = data_frame['timestamp'].dt.hour
    data_frame['day_of_week'] = data_frame['timestamp'].dt.dayofweek
    data_frame['day_of_month'] = data_frame['timestamp'].dt.day
    data_frame['month_of_year'] = data_frame['timestamp'].dt.month
    df_x, df_y = ano_label(labels[file_name], data_frame)
    df_x['value_diff'] = df_x['value'].diff()

    return df_x, df_y

def ano_label(ano_times, df_x):
    df_x['timestamp'] = pd.to_datetime(df_x['timestamp'])
    y = np.zeros(len(df_x))
    for ano_time in ano_times:
        ano_start = pd.to_datetime(ano_time[0])
        ano_end = pd.to_datetime(ano_time[1])
        for idx in df_x.index:
            if df_x.loc[idx, 'timestamp'] >= ano_start and df_x.loc[idx, 'timestamp'] <= ano_end:
                y[idx] = 1.0
    return df_x, pd.DataFrame(y, columns=['anomaly_label'])


def preprocess_data(data_frame):
    values = data_frame['value'].values
    mu = np.mean(values)
    sigma = np.std(values)
    standardized_values = (values - mu) / sigma
    data_frame['value'] = standardized_values
    
    print("\nMean of value is {}".format(np.mean(data_frame['value'])))
    print("Standard Deviation of value is {}".format(np.std(data_frame['value'])))

    return data_frame

    
def combine_data_with_labels(labels_filename, file_name):
    # Load and preprocess data with anomaly labels
    data_frame_train, df_y = load_data_with_labels(labels_filename, file_name)
    # Concatenate df_x and df_y along the column axis (axis=1)
    combined_df = pd.concat([data_frame_train, df_y], axis=1)
    return combined_df


def augment_data(data_frame, num_augmentations, augmentation_func, augmentation_type):
    data_frames_augmented = [data_frame]

    for _ in range(num_augmentations):
        df_augmented = data_frames_augmented[0].copy()
        df_augmented['value'] = augmentation_func(df_augmented['value'])
        data_frames_augmented.append(df_augmented)

    data_frame_augmented = pd.concat(data_frames_augmented)
    return data_frame_augmented



def main(labels_filename, train_file_name, test_file_name, num_augmentations, augmentation_func, augmentation_type):
    # Load and preprocess data with anomaly labels
    data_frame_train, _ = load_data_with_labels(labels_filename, f'datasets/{train_file_name}')
    data_frame_test, _ = load_data_with_labels(labels_filename, f'datasets/{test_file_name}')

    # Preprocess the data
    data_frame_train = preprocess_data(data_frame_train)
    data_frame_test = preprocess_data(data_frame_test)

    # Augment training data
    data_frame_train_augmented = augment_data(data_frame_train, num_augmentations, augmentation_func, augmentation_type)

    # return y_pred, model
    return data_frame_train_augmented

