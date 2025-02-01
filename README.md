Image Classification with CIFAR-10 Using PyTorch
This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The goal is to classify 32x32 color images into one of 10 classes, such as airplanes, cars, birds, and cats.

Table of Contents
Project Overview

Dataset

Model Architecture

Installation

Usage

Results

Future Work

License

Project Overview
This project focuses on building a deep learning model from scratch using PyTorch. The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. The project includes:

Data loading and preprocessing.

Visualization of sample images from the dataset.

Building and training a CNN model.

Evaluating the model using accuracy, ROC curve, and confusion matrix.

Dataset
The CIFAR-10 dataset contains 10 classes of objects:

Classes: Airplane, car, bird, cat, deer, dog, frog, horse, ship, and truck.

Image Size: 32x32 pixels with 3 color channels (RGB).

Dataset Split:

Training Set: 50,000 images.

Test Set: 10,000 images.

The dataset is preprocessed by normalizing pixel values to the range [-1, 1].

Model Architecture
The CNN model consists of the following layers:

Convolutional Layers:

Conv1: 32 filters, 3x3 kernel, ReLU activation.

Conv2: 64 filters, 3x3 kernel, ReLU activation.

Pooling Layers: Max pooling with 2x2 kernel and stride 2.

Fully Connected Layers:

FC1: 512 neurons, ReLU activation.

FC2: 10 neurons (output layer), Softmax activation.

The model is trained using Cross-Entropy Loss and optimized with Stochastic Gradient Descent (SGD).

Installation
To run this project, you need to install the following dependencies:

bash
Copy
pip install torch torchvision matplotlib numpy scikit-learn
Usage
Clone the Repository:

bash
Copy
git clone [https://github.com/your-username/cifar10-classification.git](https://github.com/susb47/Edge_Machine_Learning_DIU.git)
cd cifar10-classification
Run the Script:

bash
Copy
python train.py
Evaluate the Model:
The script will train the model and evaluate it on the test set. It will also generate:

A ROC curve to visualize the model's performance.

A confusion matrix to analyze classification results.

Results
After training for 5 epochs, the model achieves:

Test Accuracy: 78.5%

ROC AUC Score: 0.92

The ROC curve and confusion matrix provide additional insights into the model's performance, highlighting areas for improvement.

Future Work
Model Improvements:

Experiment with deeper architectures (e.g., ResNet, VGG).

Use data augmentation techniques to improve generalization.

Hyperparameter Tuning:

Optimize learning rate, batch size, and number of epochs.

Advanced Evaluation:

Perform per-class precision, recall, and F1-score analysis.
