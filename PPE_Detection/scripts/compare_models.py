import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the base folder where all model subdirectories are located
base_dir = 'full_model_2'

# Function to read all results.csv files in subdirectories and compile into a single DataFrame
def load_all_results(base_folder):
    all_results = []
    for subdir in os.listdir(base_folder):
        sub_path = os.path.join(base_folder, subdir)
        if os.path.isdir(sub_path):
            csv_path = os.path.join(sub_path, 'results.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['model_name'] = subdir
                all_results.append(df)
    return pd.concat(all_results, ignore_index=True)

# Load all results
combined_results = load_all_results(base_dir)

# Function to plot comparison for a specific metric
def plot_metric(metric_name, title):
    plt.figure(figsize=(12, 6))
    for model_name, group in combined_results.groupby('model_name'):
        if metric_name in group.columns:
            plt.plot(group['epoch'], group[metric_name], label=model_name)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


