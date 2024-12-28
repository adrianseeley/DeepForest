import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file into a DataFrame
data = pd.read_csv('results.csv')

# Create the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(data['forestCount'], data['treeCount'], data['fitness'], c=data['fitness'], cmap='viridis')

# Set labels
ax.set_xlabel('Forest Count')
ax.set_ylabel('Tree Count')
ax.set_zlabel('Fitness')

# Show the plot
plt.show()
