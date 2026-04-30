# Concrete Crack Detection

Machine learning project for classification and segmentation of concrete cracks.

## Models
- ResNet-18
- EfficientNet-B3
- MobileNetV2
- U-Net (segmentation)

## Datasets
- Kaggle Concrete Crack Dataset
- CrackForest Dataset (CFD)
- SDNET2018

# METHODOLOGY
Segmentation: Use U-Net with ResNet-18 encoder on CrackForest dataset

Training Experimentation:
- Data size experiment: Train models using 25%, 50%, 75%, and 100% of the training data.
- Hyperparameter tuning: Compare different learning rates (0.001, 0.0001, etc).
- Data Augmentation: Rotate images, flip images, change images brightness/contrast.
- Cross-Dataset Generalization: Train on Kaggle dataset, then test on SDNET 2018, and vice versa.
Evaluation Metrics:
- Classification: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- Segmentation: Intersection over Union (IoU) and Dice coefficient.

## Setup
```bash
pip install -r requirements.txt
```

## Download Datasets
```bash
python scripts/download_data.py
```

## Project Structure
project/

├── data/        (datasets)

├── notebooks/        (Jupyter notebook)

├── outputs/     (results, graphs, models)

├── scripts/     (download data, prepeocess data)

└── src/      (ResNet, EfficientNet, MobileNet, U-Net)

