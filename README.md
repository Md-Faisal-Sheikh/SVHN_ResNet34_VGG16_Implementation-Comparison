# SVHN Digit Classification using ResNet34 and VGG16

This project implements and compares two convolutional neural network architectures, **ResNet34** and **VGG16**, on the **Street View House Numbers (SVHN)** dataset. The goal is to analyze how different deep learning architectures perform on real-world digit recognition tasks.

The project includes the full deep learning pipeline: **data preprocessing, model implementation, training, evaluation, and performance comparison**.

The notebook contains the full implementation of:

* SVHN dataset loading and preprocessing
* Data augmentation
* ResNet34 model implementation
* VGG16 model implementation
* Training pipeline
* Model evaluation
* Performance comparison

---

# Dataset

The **Street View House Numbers (SVHN)** dataset contains images of house numbers collected from Google Street View.

Dataset characteristics:

| Property          | Value        |
| ----------------- | ------------ |
| Image Size        | 32 × 32      |
| Channels          | RGB          |
| Number of Classes | 10           |
| Labels            | Digits (0–9) |
| Training Images   | ~73,000      |
| Test Images       | ~26,000      |

SVHN is more challenging than datasets like MNIST because the digits appear in **natural images with background noise and varying lighting conditions**.
---

# Model Architectures

## ResNet34

ResNet34 is part of the **Residual Network family**, designed to address the **vanishing gradient problem** using residual connections.

Key components:

* Residual blocks
* Identity skip connections
* Batch normalization
* Deep hierarchical feature learning

Architecture overview:

```
Input Image (32x32x3)
        │
Initial Convolution
        │
Residual Block × 3
        │
Residual Block × 4
        │
Residual Block × 6
        │
Residual Block × 3
        │
Global Average Pooling
        │
Fully Connected Layer
        │
Output (10 Classes)
```
---

## VGG16

VGG16 is a deep convolutional neural network known for its **simple and uniform architecture**, using stacked **3×3 convolution filters**.

Key components:

* Sequential convolution blocks
* Small convolution filters (3×3)
* Max pooling layers
* Fully connected classifier

Architecture overview:

```
Input Image (32x32x3)
        │
Conv Block (2 layers)
        │
Max Pool
        │
Conv Block (2 layers)
        │
Max Pool
        │
Conv Block (3 layers)
        │
Max Pool
        │
Conv Block (3 layers)
        │
Max Pool
        │
Fully Connected Layers
        │
Output (10 Classes)
```
---

# Training Pipeline

The training process follows these steps:

1. Load SVHN dataset using Torchvision
2. Apply preprocessing and normalization
3. Create data loaders for training and testing
4. Initialize models (ResNet34 and VGG16)
5. Define loss function and optimizer
6. Train the models for multiple epochs
7. Evaluate performance on the test dataset

---

# Training Configuration

| Parameter     | Value         |
| ------------- | ------------- |
| Framework     | PyTorch       |
| Batch Size    | 64            |
| Optimizer     | Adam          |
| Loss Function | Cross Entropy |
| Epochs        | Adjustable    |
| Device        | GPU / CPU     |

---

# Model Comparison

| Feature            | ResNet34         | VGG16          |
| ------------------ | ---------------- | -------------- |
| Architecture Type  | Residual Network | Sequential CNN |
| Depth              | 34 layers        | 16 layers      |
| Skip Connections   | Yes              | No             |
| Training Stability | Higher           | Moderate       |
| Computational Cost | Moderate         | Higher         |

ResNet34 generally performs better in deeper architectures due to its residual learning mechanism.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
```

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

---

# Running the Notebook

Open the notebook using Jupyter:

```
jupyter notebook
```

or run it in **Google Colab**.

---

# Future Improvements

Possible extensions for this project:

* Using **ResNet50 or WideResNet**
* Hyperparameter tuning
* Advanced data augmentation
* Confusion matrix visualization
* Model compression and optimization

---

# Author

Md. Faisal Sheikh
---
