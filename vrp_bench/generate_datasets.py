import os
import numpy as np
from rl4co.data.generate_data import generate_vrp_data


vrp_sizes = [10, 20, 50, 100, 200, 500, 1000]
dataset_size = 1000
datasets = []

# 确保输出目录存在
output_dir = "data/cvrp"
os.makedirs(output_dir, exist_ok=True)

for vrp_size in vrp_sizes:
    print(f"Generating data for size {vrp_size}...")
    data = generate_vrp_data(dataset_size, vrp_size)
    data_path = os.path.join(output_dir, "vrp_{}_{}.npz".format(vrp_size, dataset_size))
    np.savez(data_path, **data)

print("Data generation complete.")
