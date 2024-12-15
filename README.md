# cancer_detection

This project uses a U-Net convolutional neural network (CNN) model to perform binary classification for cancer detection. The model is trained using augmented images and is evaluated on a test set. The U-Net architecture, originally designed for biomedical image segmentation, is adapted for classification in this task.

## Table of Contents
Project Overview
Requirements
About dataset
Data Preparation
Model Training
Model Evaluation
Results
Saving the Model
License

## Project Overview
This project aims to train a U-Net model for binary classification of cancer images, determining whether an image contains cancerous cells (class 1) or not (class 0). The model is trained using data augmentation techniques to improve generalization and robustness. The dataset is split into training, validation, and test sets, and the model's performance is evaluated using accuracy and loss metrics.

Key Steps:
Data Loading & Preprocessing:

Data augmentation is applied to training images to improve model generalization.
Data is split into training, validation, and test sets.
Model Construction:

A U-Net model architecture is implemented and compiled for binary classification.
Model Training:

The model is trained using the augmented data with early stopping and checkpoints to monitor progress.
Model Evaluation:

The trained model is evaluated on the test set to assess performance.
Saving the Model:

The trained model is saved for later use or deployment.

## Requirements
The following packages are required to run this project:

tensorflow (for building and training the U-Net model)
numpy (for numerical operations)
opencv-python (for image manipulation)
albumentations (for image augmentations)
matplotlib (for visualizations)
tqdm (for progress bar during training)
scikit-learn (for data splitting and evaluation)

## About dataset
277,000 images taken from breast cancer patients.

https://www.kaggle.com/paultimothymooney/breast-histopathology-images
## Data Preparation
The dataset used in this project is the IDC (Invasive Ductal Carcinoma) dataset. The dataset contains images of breast cancer tissue, where each image is labeled as either "cancerous" (1) or "non-cancerous" (0).

Data Preprocessing Steps:
Images are loaded from the dataset directory.
Data augmentation (e.g., flipping, rotating, brightness adjustments) is applied to the training images to increase the diversity of the dataset.
Data is split into training (80%), validation (16%), and test (20%) sets.
The image size is adjusted to (256, 256) pixels to fit the model input requirements.
Example of Augmentation:
Horizontal flip
Random rotation
Brightness/contrast adjustment
Shift, scale, and rotate operations

## Model Training
The U-Net model is trained using the Adam optimizer with a learning rate of 1e-4 and binary_crossentropy loss. The training process is monitored using the accuracy metric.

## Model Evaluation
After training, the model's performance is evaluated on the test set. The key metrics used for evaluation are:
Test Accuracy: The proportion of correctly classified images in the test set.
Test Loss: The binary cross-entropy loss computed for the test set.

## Results
After training the model for 10 epochs, the results are displayed as the test accuracy and loss values. The model's performance can be further analyzed by adjusting hyperparameters, exploring more augmentations, or using a different model architecture.

## Dataset
277,000 images taken from breast cancer patients.

https://www.kaggle.com/paultimothymooney/breast-histopathology-images
