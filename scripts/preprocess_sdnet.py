import os
import shutil
from sklearn.model_selection import train_test_split

def collect_images(source_dir):
    all_images = []

    for root, dirs, files in os.walk(source_dir):
        folder_name = os.path.basename(root).lower()

        if folder_name == "cracked":
            label = "crack"
        elif folder_name == "non-cracked":
            label = "no_crack"
        else:
            continue

        for file_name in files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file_name)
                all_images.append((image_path, label))

    return all_images


def split_sdnet(source_dir, dest_dir):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dest_dir, split, "crack"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, "no_crack"), exist_ok=True)

    all_images = collect_images(source_dir)

    print(f"Total images found: {len(all_images)}")

    if len(all_images) == 0:
        print("ERROR: No images found")
        return

    labels = [label for _, label in all_images]

    train_data, temp_data = train_test_split(
        all_images, test_size=0.3, stratify=labels, random_state=42
    )

    temp_labels = [label for _, label in temp_data]

    val_data, test_data = train_test_split(
        temp_data, test_size=0.66, stratify=temp_labels, random_state=42
    )

    for split, data in zip(["train", "val", "test"], [train_data, val_data, test_data]):
        for i, (image_path, label) in enumerate(data):
            if i % 500 == 0:
                print(f"{split}: {i}/{len(data)}")

            out_dir = os.path.join(dest_dir, split, label)

            try:
                shutil.copy2(image_path, out_dir)
            except:
                print(f"Skipping bad file: {image_path}")

    print("Done processing SDNET")


if __name__ == "__main__":
    source_dir = "./data/raw/structural_defects"
    dest_dir = "./data/processed_sdnet"

    split_sdnet(source_dir, dest_dir)