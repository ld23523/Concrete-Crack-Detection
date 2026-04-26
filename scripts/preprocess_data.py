import os
import shutil
from sklearn.model_selection import train_test_split

def split_data_classification(source_dir, dest_dir, test_size=0.2, val_size=0.1):
    """
    Splits data into train, validation, and test sets.

    Parameters:
        source_dir (str): Path to the source data directory.
        dest_dir (str): Path to the destination directory where splits will be saved.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
    """
    # Create necessary directories for train, val, and test splits
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split, 'crack'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, 'no_crack'), exist_ok=True)

    # Define the image categories (crack = Positive, no_crack = Negative)
    categories = ['Positive', 'Negative']
    
    # Iterate over the categories and split the data
    all_images = []
    for label in categories:
        label_dir = os.path.join(source_dir, label)
        for image_name in os.listdir(label_dir):
            # Get image path
            image_path = os.path.join(label_dir, image_name)
            # Label is 0 for 'crack' (Positive) and 1 for 'no_crack' (Negative)
            all_images.append((image_path, 'crack' if label == 'Positive' else 'no_crack'))

    # Split into train, val, test
    train_data, test_data = train_test_split(all_images, test_size=test_size, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=val_size / (val_size + test_size), random_state=42)

    # Move files to respective directories
    for split, data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        for image_path, label in data:
            label_dir = os.path.join(dest_dir, split, label)
            shutil.copy(image_path, label_dir)

def split_segmentation_data(source_images_dir, source_masks_dir, dest_dir, test_size=0.2, val_size=0.1):
    """
    Splits segmentation data into train, validation, and test sets.
    Assumes images and masks have corresponding names.
    """
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    # Get image files
    image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png'))]
    
    # Split
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=val_size / (val_size + test_size), random_state=42)

    for split, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for img_file in files:
            # Copy image
            shutil.copy(os.path.join(source_images_dir, img_file), os.path.join(dest_dir, split, img_file))
            # Copy mask (assume mask has _mask.png)
            mask_file = img_file.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
            if os.path.exists(os.path.join(source_masks_dir, mask_file)):
                shutil.copy(os.path.join(source_masks_dir, mask_file), os.path.join(dest_dir, split, mask_file))

if __name__ == "__main__":
    # Classification
    source_dir = './data/raw/structural_defects/'  # This is the folder with Positive/Negative
    dest_dir = './data/processed/structural_defects_75_10_15/'  # Where we want to save the train/val/test split
    split_data_classification(source_dir, dest_dir, test_size=0.15, val_size=0.1)