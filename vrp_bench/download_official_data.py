
import os
import shutil
from huggingface_hub import hf_hub_download
from tqdm import tqdm

OUTPUT_DIR = "data"
SIZES = [10, 20, 50, 100, 200, 500, 1000]

def main():
    print("Downloading official dataset files...")
    
    # Create directories
    os.makedirs(os.path.join(OUTPUT_DIR, "cvrp"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "twcvrp"), exist_ok=True)
    
    for size in tqdm(SIZES, desc="Downloading"):
        # CVRP
        try:
            filename = f"real_cvrp/cvrp_{size}_single_depot.npz"
            local_path = hf_hub_download(repo_id="Yahias21/vrp_benchmark", filename=filename, repo_type="dataset")
            
            # Copy to expected location
            target_path = os.path.join(OUTPUT_DIR, f"cvrp/vrp_{size}_1000.npz")
            shutil.copy(local_path, target_path)
            # Also copy to non-1000 suffix for compatibility
            shutil.copy(local_path, os.path.join(OUTPUT_DIR, f"cvrp/vrp_{size}.npz"))
            
        except Exception as e:
            print(f"Error downloading CVRP size {size}: {e}")
            
        # TWCVRP
        try:
            filename = f"real_twcvrp/twvrp_{size}_single_depot.npz"
            local_path = hf_hub_download(repo_id="Yahias21/vrp_benchmark", filename=filename, repo_type="dataset")
            
            # Copy to expected location
            target_path = os.path.join(OUTPUT_DIR, f"twcvrp/vrp_{size}_1000.npz")
            shutil.copy(local_path, target_path)
            # Also copy to non-1000 suffix for compatibility
            shutil.copy(local_path, os.path.join(OUTPUT_DIR, f"twcvrp/vrp_{size}.npz"))
            
        except Exception as e:
            print(f"Error downloading TWCVRP size {size}: {e}")

    print("Download complete.")

if __name__ == "__main__":
    main()
