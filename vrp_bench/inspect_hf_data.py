
from datasets import load_dataset
import numpy as np
import os

def inspect_dataset():
    print("Loading dataset from Hugging Face...")
    try:
        ds = load_dataset("Yahias21/vrp_benchmark")
        print("Dataset loaded successfully.")
        print("Keys:", ds.keys())
        
        if 'test' in ds:
            print("Sample from 'test' split:")
            sample = ds['test'][0]
            print("Sample keys:", sample.keys())
            for key, value in sample.items():
                print(f"{key}: {type(value)}")
                if isinstance(value, list):
                    print(f"  Length: {len(value)}")
                    if len(value) > 0:
                        print(f"  First element: {value[0]}")
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    inspect_dataset()
