from huggingface_hub import hf_hub_download
import zipfile
import os

REPO_ID = "ShandaAI/FloodDiffusionDownloads"

def download_extract_zip(filename, target_dir="."):
    print(f"Downloading {filename}...")
    path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="model")
    print(f"Extracting {filename} to {target_dir}...")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

# 1. Download and extract Dependencies (creates ./deps/)
download_extract_zip("deps.zip", ".")

# 2. Download and extract Datasets (creates ./raw_data/HumanML3D and ./raw_data/BABEL_streamed)
os.makedirs("raw_data", exist_ok=True)
download_extract_zip("HumanML3D.zip", "raw_data")
download_extract_zip("BABEL_streamed.zip", "raw_data")

# 3. Download Models (creates ./outputs/)
download_extract_zip("outputs.zip", ".")

print("Done! Your project is ready.")

