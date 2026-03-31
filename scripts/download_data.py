import kagglehub
import shutil
import os

# Download Crack Concrete Images for Classification dataset
def download_kaggle_dataset(path, target_dir):
    print("Downloading dataset using KaggleHub...")

    print(f"Downloaded to: {path}")

    if not os.path.exists(target_dir):
        shutil.copytree(path, target_dir)

    print(f"Dataset copied to: {target_dir}")

if __name__ == "__main__":
    # Download Crack Concrete Images for Classification dataset
    download_kaggle_dataset(
        path = kagglehub.dataset_download("arnavr10880/concrete-crack-images-for-classification"),
        target_dir = "../data/raw/concrete_crack"
    )

    # Download Crackforest dataset
    download_kaggle_dataset(
        path = kagglehub.dataset_download("mahendrachouhanml/crackforest"),
        target_dir = "../data/raw/crackforest"
    )

    # Download SDNET 2018 dataset
    download_kaggle_dataset(
        path = kagglehub.dataset_download("aniruddhsharma/structural-defects-network-concrete-crack-images"),
        target_dir = "../data/raw/structural_defects"
    )