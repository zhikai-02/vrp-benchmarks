
import os
import numpy as np
from tqdm import tqdm
from real_cvrp import generate_cvrp_instance
from real_twcvrp import generate_twcvrp_instance, generate_time_window

# Configuration
SIZES = [10, 20, 50, 100, 200, 500, 1000]
NUM_INSTANCES = 5  # Small number for quick reproduction
OUTPUT_DIR = "data"

def generate_dataset(problem_type, sizes, num_instances):
    output_path = os.path.join(OUTPUT_DIR, problem_type)
    os.makedirs(output_path, exist_ok=True)
    
    for size in sizes:
        print(f"Generating {problem_type} data for size {size}...")
        
        all_locations = []
        all_demands = []
        all_capacities = []
        all_time_windows = []
        all_appear_times = []
        
        for _ in tqdm(range(num_instances)):
            if problem_type == 'cvrp':
                # Force single depot for consistency and to avoid KMeans error on small sizes
                instance = generate_cvrp_instance(num_customers=size, num_depots=1)
                # CVRP has no time windows (use default 0-inf)
                time_windows = np.zeros((len(instance['locations']), 2))
                time_windows[:, 1] = 100000  # Large horizon
            else:
                # Force single depot for TWCVRP as well
                instance = generate_twcvrp_instance(num_customers=size, num_depots=1)
                # Generate time windows
                time_windows = []
                for i, appear in enumerate(instance['appear_time']):
                    if i == 0: # Depot
                        time_windows.append([0, 1440])
                    else:
                        tw = generate_time_window(appear)
                        time_windows.append(list(tw))
                time_windows = np.array(time_windows)

            all_locations.append(instance['locations'])
            all_demands.append(instance['demands'])
            all_capacities.append(instance['vehicle_capacity'])
            all_appear_times.append(instance['appear_time'])
            all_time_windows.append(time_windows)
            
        # Stack into arrays
        # Note: locations might have different lengths if num_depots varies, 
        # but generate_base_instance uses fixed num_depots=1 (or 3 for cvrp default) + num_customers
        # We need to ensure consistency. 
        # generate_cvrp_instance uses num_depots=3 by default.
        # generate_twcvrp_instance uses num_depots=1 by default.
        
        # To make it simple for stacking, we'll trust numpy to handle it if shapes match, 
        # or use object arrays if they don't.
        
        data = {
            'locs': np.array(all_locations),
            'demand': np.array(all_demands),
            'capacity': np.array(all_capacities),
            'time_windows': np.array(all_time_windows),
            'appear_times': np.array(all_appear_times)
        }
        
        # Save
        filename = f"vrp_{size}_1000.npz"
        save_path = os.path.join(output_path, filename)
        np.savez(save_path, **data)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    # Generate CVRP
    generate_dataset('cvrp', SIZES, NUM_INSTANCES)
    
    # Generate TWCVRP
    generate_dataset('twcvrp', SIZES, NUM_INSTANCES)
