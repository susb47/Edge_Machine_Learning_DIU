# Image Classification with CIFAR-10 Using PyTorch

This project demonstrates how to build, train, and evaluate a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10 dataset**. The goal is to classify 32x32 color images into one of 10 classes, such as airplanes, cars, birds, and cats.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Work](#future-work)
8. [License](#license)

---

## Project Overview
This project focuses on building a deep learning model from scratch using **PyTorch**. The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. The project includes:
- Data loading and preprocessing.
- Visualization of sample images from the dataset.
- Building and training a CNN model.
- Evaluating the model using accuracy, ROC curve, and confusion matrix.

---

## Dataset
The **CIFAR-10 dataset** contains 10 classes of objects:
- **Classes**: Airplane, car, bird, cat, deer, dog, frog, horse, ship, and truck.
- **Image Size**: 32x32 pixels with 3 color channels (RGB).
- **Dataset Split**:
  - **Training Set**: 50,000 images.
  - **Test Set**: 10,000 images.

The dataset is preprocessed by normalizing pixel values to the range `[-1, 1]`.

---

## Model Architecture
The CNN model consists of the following layers:
1. **Convolutional Layers**:
   - `Conv1`: 32 filters, 3x3 kernel, ReLU activation.
   - `Conv2`: 64 filters, 3x3 kernel, ReLU activation.
2. **Pooling Layers**: Max pooling with 2x2 kernel and stride 2.
3. **Fully Connected Layers**:
   - `FC1`: 512 neurons, ReLU activation.
   - `FC2`: 10 neurons (output layer), Softmax activation.

The model is trained using **Cross-Entropy Loss** and optimized with **Stochastic Gradient Descent (SGD)**.

---

## Installation
To run this project, you need to install the following dependencies:

```bash
pip install torch torchvision matplotlib numpy scikit-learn
