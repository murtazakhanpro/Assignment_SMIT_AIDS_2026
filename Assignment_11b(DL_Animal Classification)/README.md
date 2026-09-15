# Assignment 11B: 5-Species Animal Image Classification (CNN)

## Project Overview

This project implements a multi-class image classification system using Convolutional Neural Networks (CNN) in TensorFlow and Keras to classify images across 5 distinct animal species. It features a complete pipeline including dataset structuring, automated train-validation-test splitting, data augmentation, CNN modeling, callback-driven training with early stopping, model checkpointing, and comprehensive evaluation via confusion matrices and classification reports.

## Tech Stack

- Python 3
- TensorFlow
- Keras
- scikit-learn
- Matplotlib
- Seaborn
- NumPy
- Pathlib & Shutil

## Techniques Used

- Image dataset verification and class discovery across directory hierarchies
- Automated dataset splitting into Train (70%), Validation (15%), and Test (15%) sets
- Data augmentation layers (`RandomFlip`, `RandomRotation`, `RandomZoom`, `Rescaling`) to enhance generalization
- Deep Convolutional Neural Network (CNN) architecture:
  - 4 hierarchical `Conv2D` + `MaxPooling2D` feature extraction blocks (32, 64, 128, 256 filters)
  - Dense classification head with `Dropout(0.5)` for regularization
  - Multi-class output layer with `softmax` activation
- Adam optimizer and categorical cross-entropy loss
- Training management with callbacks:
  - `EarlyStopping` with patience to prevent overfitting and restore best weights
  - `ModelCheckpoint` saving `best_animal_classifier.keras` based on validation accuracy
- Comprehensive evaluation:
  - Classification report (precision, recall, F1-score)
  - Confusion matrix heatmap visualized with Seaborn and Matplotlib

## Skills Practiced

- Developing end-to-end multi-class image classification pipelines
- Structuring raw computer vision datasets into standardized train/val/test splits
- Implementing on-the-fly data augmentation within Keras sequential models
- Managing deep learning training with callbacks and model checkpointing
- Evaluating multi-class computer vision models with detailed confusion matrix analysis
