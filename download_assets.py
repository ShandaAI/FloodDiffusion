import argparse
import os
import zipfile

from huggingface_hub import hf_hub_download

REPO_ID = "ShandaAI/FloodDiffusionDownloads"


def download_extract_zip(filename, target_dir="."):
    print(f"Downloading {filename}...")
    path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="model")
    print(f"Extracting {filename} to {target_dir}...")
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FloodDiffusion assets")
    parser.add_argument(
        "--with-dataset",
        action="store_true",
        help="Download datasets (HumanML3D and BABEL) for training. By default, only deps and outputs are downloaded for inference.",
    )
    args = parser.parse_args()

    # 1. Download and extract Dependencies (creates ./deps/)
    print("Downloading dependencies...")
    download_extract_zip("deps.zip", ".")

    # 2. Download and extract Datasets (optional)
    if args.with_dataset:
        print("Downloading datasets...")
        os.makedirs("raw_data", exist_ok=True)
        download_extract_zip("HumanML3D.zip", "raw_data")
        download_extract_zip("BABEL_streamed.zip", "raw_data")
    else:
        print(
            "Skipping dataset download (add --with-dataset flag to download datasets for training)"
        )

    # 3. Download Models (creates ./outputs/)
    print("Downloading model outputs...")
    download_extract_zip("outputs.zip", ".")
    download_extract_zip("outputs_tiny.zip", ".")

    print("\nDone! Your project is ready.")
    if not args.with_dataset:
        print("Note: Datasets were not downloaded. This setup is for inference only.")
