# Copyright 2023 MA Song at UCL FRL

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store data
num_runs = 3
step = np.arange(10000)
averages = {}

upper_bounds = {}
lower_bounds = {}

for m in [50, 100]:
    values = np.zeros((num_runs, 10000))

    for k in range(num_runs):
        # Load data from the TensorBoard event file
        logdir = str('/Users/masong/codes/rl_waypoint_mrta/trained_sessions/' +
                     'moe_mlp/rand_' +
                     str(m) +
                     '-' +
                     str(k + 3) +
                     '/events.out.tfevents.0')
        summary_iterator = tf.compat.v1.train.summary_iterator(logdir)
        cost = []
        # Extract summary data
        for event in summary_iterator:
            for value in event.summary.value:
                if value.tag == 'cost_d_sum_origin':  # Replace with your specific tag
                    cost.append(value.simple_value)
            if event.step == 10000:
                break
        values[k] = cost

    # Calculate average, upper bound, and lower bound
    average = np.mean(values, axis=0)
    std_dev = np.std(values, axis=0)
    upper_bound = average + std_dev
    lower_bound = average - std_dev
    
    averages[m] = average
    upper_bounds[m] = upper_bound
    lower_bounds[m] = lower_bound

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the average line with shaded area for bounds
# plt.plot(step, values, label='Cost')
plt.plot(step, averages[50], color='cornflowerblue', linestyle='-', label='M = 50 Average')
plt.fill_between(step, lower_bounds[50], upper_bounds[50], color='lightsteelblue', alpha=0.4, label='M = 50 SD')

plt.plot(step, averages[100], color='sandybrown', linestyle='-', label='M = 100 Average')
plt.fill_between(step, lower_bounds[100], upper_bounds[100], color='peachpuff', alpha=0.4, label='M = 100 SD')

# Add labels and legend
plt.xlabel('Steps')
plt.ylabel('Cost: total distance travelled')
# plt.title('RL Training Cost Curves')
plt.legend()

# Show the plot
plt.show()
