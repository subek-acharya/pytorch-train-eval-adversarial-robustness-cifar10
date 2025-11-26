# Adversarial Robustness Evaluation on CIFAR-10

A comprehensive implementation for training and evaluating the adversarial robustness of state-of-the-art CNN architectures on the CIFAR-10 dataset using white-box adversarial attacks.

## 🔍 Overview

This project implements a systematic evaluation framework to assess the adversarial robustness of various Convolutional Neural Network (CNN) architectures. The models are trained on the CIFAR-10 dataset and then tested against white-box adversarial attacks to analyze their vulnerability and robustness characteristics.


## 🏗️ Model Architectures

1. **ResNet-18**
2. **VGG-16**
3. **DenseNet-121**
4. **GoogLeNet**

## ⚔️ Adversarial Attacks

1. **FGSM (Fast Gradient Sign Method)**
2. **PGD (Projected Gradient Descent)**

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

## Usage

### Training Models

To train a model, modify the model selection in train.py and run:
```bash
# In train.py, uncomment the desired model:

# Train ResNet-18
model = resnet.PreActResNet18().to(device)

# Train VGG-16
# model = vgg.VGG('VGG16').to(device)

# Train DenseNet-121
# model = densenet.DenseNet121().to(device)

# Train GoogLeNet
# model = googlenet.GoogLeNet().to(device)

# Then run:
python train.py
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
## In main.py, set the model directory and uncomment the desired model:

modelDir = "./checkpoint/resnet18.pth"
model = resnet.PreActResNet18().to(device)

# For evaluation only, uncomment:
results = evaluation.evaluate_model(device, model, valLoader)
evaluation.print_evaluation_results(results)

# Then run:
python main.py
```

### Adversarial Attacks

Generate and evaluate adversarial examples using white-box attacks.

### Attack Paramters
```bash
epsilonMax = 0.031      # Maximum perturbation
clipMin = 0.0           # Minimum value a pixel can take
clipMax = 1.0           # Maximum value a pixel can take 
numSteps = 20           # Number of PGD steps
epsilonStep = epsilonMax/numSteps  # Step size for PGD
```

### Running Attacks
```bash
# In main.py, set the model directory and uncomment the desired model:

modelDir = "./checkpoint/resnet18.pth"
model = resnet.PreActResNet18().to(device)

# Run the attacks
advLoader_FGSM = AttackWrappersWhiteBox.FGSMNativePytorch(device, correctLoader, model, epsilonMax, clipMin, clipMax)
advLoader_PGD = AttackWrappersWhiteBox.PGDNativePytorch(device, correctLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax)

# Then run:
python main.py
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
├── sample_images/                   # Adversarial example visualizations
│   ├── output1.png                  # Clean vs adversarial image comparison (ε = 1.0)
│   └── output2.png                  # Clean vs adversarial image comparison (ε = 0.031)
│
├── infographics/                    # Performance visualization charts
│   ├── adversarial_accuracy_comparison.png  # Robustness comparison chart
│   └── model_comparison_styled.png          # Model performance metrics
│
├── train.py                     # Training script
├── main.py                      # Evaluation and attack script
├── evaluation.py                # Evaluation metrics implementation
├── utils.py                     # Utility functions and data loaders
├── AttackWrappersWhiteBox.py   # FGSM and PGD attack implementations
├── visualize.py                 # Visualization utilities
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```


## 📊 Results

### Clean CIFAR-10 Test Performance

All models were trained for 200 epochs using identical training parameters to ensure fair comparison.

#### Training Performance Comparison

| Model | Test Accuracy | Training Time | Parameters |
|-------|---------------|---------------|------------|
| DenseNet-121 | 95.33% | 256.62 min | 6.96M |
| GoogLeNet | 95.28% | 266.23 min | 6.17M |
| ResNet-18 | 95.17% | 68.80 min | 11.17M |
| VGG-16 | 93.33% | 47.13 min | 14.73M |

#### Evaluation Metrics (Clean Data)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| DenseNet-121 | 95.33% | 95.33% | 95.33% | 95.33% |
| GoogLeNet | 95.28% | 95.28% | 95.28% | 95.28% |
| ResNet-18 | 95.17% | 95.17% | 95.17% | 95.16% |
| VGG-16 | 93.33% | 93.34% | 93.33% | 93.33% |

*All metrics computed using macro-averaging across all 10 CIFAR-10 classes.*

---

### Adversarial Robustness Results

Adversarial evaluation conducted on 1,000 correctly classified, class-balanced samples (100 per class) with ε = 0.031.

| Model | Clean Accuracy | FGSM Accuracy | PGD Accuracy |
|-------|------------|-----------|----------|
| VGG-16 | 93.33% | 55.9% | 8.7% |
| DenseNet-121 | 95.33% | 52.4% | 11.1% |
| ResNet-18 | 95.17% | 42.9% | 0.5% |
| GoogLeNet | 95.28% | 33.9% | 0.0% |
