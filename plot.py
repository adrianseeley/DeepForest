import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import numpy as np

# Increase the maximum number of columns displayed (optional)
pd.set_option('display.max_columns', None)

# Define the path to your CSV file
csv_file = 'results.csv'

# Check if the file exists
if not os.path.isfile(csv_file):
    raise FileNotFoundError(f"The file '{csv_file}' does not exist in the current directory.")

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv(csv_file)

# Create an output directory for the plots
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# 1. Separate Plots for Each Key Metric
print("Generating individual plots for key metrics...")

# List of key metrics to plot individually
key_metrics = ['fails', 'k', 'distanceSum', 'testCorrect', 
               'inputWeightSum', 'distanceWeightSum', 'contributionWeightSum']

for metric in key_metrics:
    if metric not in df.columns:
        print(f"Warning: '{metric}' column not found in the CSV. Skipping this metric.")
        continue
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df[metric], label=metric, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{metric}_vs_epoch.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot for '{metric}' to {plot_path}")

# 2. Weight Charts with Individual Lines

def plot_weight_category(df, weight_prefix, category_name, max_weights=None):
    """
    Plots each weight in a category against epoch.

    Parameters:
    - df: pandas DataFrame containing the data.
    - weight_prefix: Prefix of the weight columns (e.g., 'inputWeight').
    - category_name: Name of the category for labeling purposes.
    - max_weights: Optional integer to limit the number of weights plotted.
    """
    print(f"Generating plot for {category_name} weights...")
    weight_columns = [col for col in df.columns if col.startswith(weight_prefix) and not col.endswith('Sum')]
    
    if not weight_columns:
        print(f"Warning: No columns found with prefix '{weight_prefix}'. Skipping this category.")
        return
    
    if max_weights:
        weight_columns = weight_columns[:max_weights]
        print(f"Plotting first {max_weights} weights out of {len(weight_columns)}.")
    
    num_weights = len(weight_columns)
    cmap = cm.get_cmap('hsv', num_weights)
    
    plt.figure(figsize=(15, 10))
    
    for idx, col in enumerate(weight_columns):
        plt.plot(df['epoch'], df[col], color=cmap(idx), alpha=0.6, linewidth=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.title(f'{category_name.capitalize()} Weights vs Epoch')
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{weight_prefix}_vs_epoch.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot for '{category_name}' weights to {plot_path}")

# Optionally, set a maximum number of weights to plot to manage performance and readability
MAX_WEIGHTS_TO_PLOT = None  # Set to an integer like 100 if needed

# 2.1 Plot: epoch vs each of the input weights
plot_weight_category(df, 'inputWeight', 'input', max_weights=MAX_WEIGHTS_TO_PLOT)

# 2.2 Plot: epoch vs each of the distance weights
plot_weight_category(df, 'distanceWeight', 'distance', max_weights=MAX_WEIGHTS_TO_PLOT)

# 2.3 Plot: epoch vs each of the contribution weights
plot_weight_category(df, 'contributionWeight', 'contribution', max_weights=MAX_WEIGHTS_TO_PLOT)

print("All plots have been generated and saved successfully.")
