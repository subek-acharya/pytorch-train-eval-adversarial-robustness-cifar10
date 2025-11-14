# Adversarial Robustness Evaluation on CIFAR-10



A comprehensive implementation for training and evaluating the adversarial robustness of state-of-the-art CNN architectures on the CIFAR-10 dataset using white-box adversarial attacks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architectures](#model-architectures)
- [Adversarial Attacks](#adversarial-attacks)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
  - [Running Adversarial Attacks](#running-adversarial-attacks)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

This project implements a systematic evaluation framework to assess the adversarial robustness of various Convolutional Neural Network (CNN) architectures. The models are trained on the CIFAR-10 dataset and then tested against white-box adversarial attacks to analyze their vulnerability and robustness characteristics.

## ✨ Features

- **Multiple CNN Architectures**: Train and evaluate 4 different state-of-the-art models
- **Comprehensive Metrics**: Accuracy, Precision, Recall, and F1-Score evaluation
- **White-box Adversarial Attacks**: Implementation of FGSM and PGD attacks
- **Comparative Analysis**: Side-by-side comparison of model robustness
- **GPU Acceleration**: Multi-GPU support for faster training
- **Reproducible Results**: Consistent training and evaluation pipeline

## 🏗️ Model Architectures

The following CNN architectures are implemented and evaluated:

1. **ResNet-18** (Pre-Activation)
   - Deep residual learning with skip connections
   - Pre-activation variant for improved gradient flow

2. **VGG-16**
   - Classic deep architecture with small convolutional filters
   - Straightforward layer stacking approach

3. **DenseNet-121**
   - Dense connections between layers
   - Feature reuse for parameter efficiency

4. **GoogLeNet (Inception v1)**
   - Multi-scale feature extraction with inception modules
   - Auxiliary classifiers for improved gradient flow

## ⚔️ Adversarial Attacks

### White-box Attacks Implemented:

1. **FGSM (Fast Gradient Sign Method)**
   - Single-step attack
   - Computationally efficient
   - Baseline adversarial attack

2. **PGD (Projected Gradient Descent)**
   - Multi-step iterative attack
   - Stronger than FGSM
   - Considered one of the strongest first-order adversarial attacks

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- pip package manager

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adversarial-robustness-cifar10.git
cd adversarial-robustness-cifar10
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```bash
adversarial-robustness-cifar10/
│
├── model_architecture/          # Model architecture implementations
│   ├── resnet.py               # ResNet-18 implementation
│   ├── vgg.py                  # VGG-16 implementation
│   ├── densenet.py             # DenseNet-121 implementation
│   └── googlenet.py            # GoogLeNet implementation
│
├── checkpoint/                  # Saved model checkpoints
│   ├── resnet18.pth
│   ├── vgg16.pth
│   ├── densenet121.pth
│   └── googlenet.pth
│
├── data/                        # CIFAR-10 dataset (auto-downloaded)
│
├── train.py                     # Training script
├── main.py                      # Evaluation script
├── evaluation.py                # Evaluation metrics implementation
├── utils.py                     # Utility functions and data loaders
├── adversarial_attack.py        # FGSM and PGD attack implementations
├── requirements.txt             # Python dependencies
└── README.md                    # This file

```

## Usage

### Training Models

To train a model, modify the model selection in train.py and run:
```bash
# Train ResNet-18
python train.py  # Uncomment model = resnet.PreActResNet18()

# Train VGG-16
python train.py  # Uncomment model = vgg.VGG('VGG16')

# Train DenseNet-121
python train.py  # Uncomment model = densenet.DenseNet121()

# Train GoogLeNet
python train.py  # Uncomment model = googlenet.GoogLeNet()
```

### Training Paramters
```bash
Epochs: 200
Batch Size: 64
Learning Rate: 0.01 (with Cosine Annealing)
Optimizer: SGD with momentum (0.9)
Weight Decay: 5e-4
```
### Evaluating Models

Evaluate a trained model on clean CIFAR-10 test data:
```bash
# Evaluate ResNet-18
python main.py  # Set modelDir to "./checkpoint/resnet18.pth"
```

## 📊 Results

### Clean CIFAR-10 Test Accuracy

All models were trained for 200 epochs on the CIFAR-10 dataset using the same training parameters (SGD with momentum, cosine annealing learning rate scheduler, batch size 64).

#### Performance Comparison

| Model | Test Accuracy | Training Time | Total Parameters |
|-------|--------------|---------------|------------------|
| **GoogLeNet** | **95.31%** | 253.27 min | 6.17M |
| **DenseNet-121** | **95.28%** | 254.50 min | 6.96M |
| **ResNet-18** | **95.22%** | 67.14 min | 11.17M |
| **VGG-16** | **93.97%** | 57.82 min | 14.73M |
