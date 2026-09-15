# Assignment 11A: Gun vs No Gun Image Classification (CNN)

## Project Overview

This project implements a binary image classification Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images into `Gun` and `No Gun` categories. It utilizes the Kaggle Gun Detection dataset, implementing an optimized end-to-end computer vision pipeline covering dataset loading, image preprocessing, train-validation splitting, custom CNN modeling, training, and performance evaluation.

## Tech Stack

- Python 3
- Jupyter Notebook
- TensorFlow
- Keras
- scikit-learn
- NumPy
- Matplotlib
- KaggleHub

## Techniques Used

- Dataset downloading and extraction via KaggleHub
- Image discovery and dynamic label extraction based on bounding annotation files
- Stratified train-validation splitting (80% train / 20% validation)
- High-performance data pipeline using `tf.data.Dataset` (`shuffle`, `batch`, and `prefetch` with `AUTOTUNE`)
- Image decoding (`tf.image.decode_jpeg`), resizing (224x224), and pixel normalization ([0, 1])
- Deep Convolutional Neural Network (CNN) architecture:
  - Multiple `Conv2D` layers with ReLU activation
  - `MaxPooling2D` spatial downsampling
  - `Flatten` and dense hidden layer with `Dropout(0.5)` for regularization
  - Output `Dense` layer with `sigmoid` activation for binary classification
- Binary cross-entropy loss function with Adam optimizer
- Model training monitoring across epochs (accuracy and loss)

## Skills Practiced

- Building end-to-end computer vision workflows in TensorFlow/Keras
- Efficient image loading and batching using the `tf.data` API
- Designing and training custom Convolutional Neural Networks (CNNs)
- Mitigating overfitting using Dropout regularization
- Evaluating binary classification performance for safety-critical computer vision tasks
