import numpy as np
import os

def load_vrp_dataset(problem_type, size, variant, base_path="vrp_bench/data"):
    """
    Load a VRP dataset.
    Example: load_vrp_dataset('cvrp', 50, 'single_depot')
    """
    # Mapping for variants based on README and file structure
    if problem_type == 'cvrp':
        folder = 'real_cvrp'
        prefix = 'cvrp'
        if variant == 'single_depot':
            full_variant = 'single_depot_single_vehicule_sumDemands'
        else:
            full_variant = variant
    elif problem_type == 'twcvrp':
        folder = 'real_twcvrp'
        prefix = 'twvrp'
        if variant == 'equal_city':
            full_variant = 'depots_equal_city'
        else:
            full_variant = variant
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    filename = f"{prefix}_{size}_{full_variant}.npz"
    
    # Check possible paths
    paths = [
        os.path.join(base_path, folder, filename),
        os.path.join("..", base_path, folder, filename), # If running from notebook
        os.path.join(os.path.dirname(os.path.abspath(__file__)), base_path, folder, filename) # Absolute relative to this file
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"Loading dataset from {path}")
            return dict(np.load(path, allow_pickle=True))
            
    # If not found, try to list what is available to help debugging
    print(f"Dataset {filename} not found. Searched in:")
    for p in paths:
        print(f" - {p}")
    
    raise FileNotFoundError(f"Dataset {filename} not found.")

if __name__ == "__main__":
    dataset_path = "data/cvrp/vrp_10_1000.npz"
    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
        # NpzFile 'data/cvrp/vrp_10_1000.npz' with keys: depot, locs, demand, capacity
        instances = []
        for i in range(len(data["locs"])):
            instances.append(
                {
                    "depot": data["depot"][i],
                    "locs": data["locs"][i],
                    "demand": data["demand"][i],
                    "capacity": data["capacity"][i],
                }
            )
        print(instances[0])
    else:
        print(f"Default test file {dataset_path} not found.")
