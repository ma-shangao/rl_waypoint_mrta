# Copyright 2023 MA Song at UCL FRL

import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a list of costs for each training instance
training_instance_1 = [10, 8, 6, 4, 3, 2, 1]
training_instance_2 = [12, 9, 7, 5, 4, 3, 2]
training_instance_3 = [8, 6, 5, 4, 3, 2, 1]

# Combine the data into a 2D array
all_instances = np.array([training_instance_1, training_instance_2, training_instance_3])

# Calculate average, upper bound, and lower bound
average = np.mean(all_instances, axis=0)
std_dev = np.std(all_instances, axis=0)
upper_bound = average + std_dev
lower_bound = average - std_dev

# Create x-axis values (e.g., epochs or time steps)
epochs = range(1, len(training_instance_1) + 1)

# Plot the average line with shaded area for bounds
plt.figure(figsize=(10, 6))

plt.plot(epochs, average, label='Average', color='b')
plt.fill_between(epochs, lower_bound, upper_bound, color='skyblue', alpha=0.4, label='Bounds')

# Add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('RL Training Cost Curves')
plt.legend()

# Show the plot
plt.show()
