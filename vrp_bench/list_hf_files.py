
from huggingface_hub import list_repo_files

try:
    files = list_repo_files("Yahias21/vrp_benchmark", repo_type="dataset")
    print("Files in repo:")
    for f in files:
        print(f)
except Exception as e:
    print(f"Error: {e}")
