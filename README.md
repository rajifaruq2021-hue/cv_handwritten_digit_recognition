# Handwritten Digit Recognition with Convolutional Neural Networks
Course: DLBAIPCV01 - Project: Computer Vision
Degree Programme: BSc Computer Science / Artificial Intelligence
IU International University of Applied Sciences

Author: Faruq Ishola Raji
Matriculation Number: 4253139

---

## 1. Project Overview

This project implements, evaluates, and compares two landmark Convolutional Neural Network (CNN) architectures for handwritten digit recognition on the MNIST benchmark dataset:
1. **The Historical LeCun et al. (1989) CNN:** An approximation of the pioneering 4-layer backpropagation network featuring strided convolutions (5x5 kernels), symmetric hyperbolic tangent (Tanh) activations, and dropout regularization emulating historical constrained receptive field connectivity between hidden layers H2 and H3.
2. **The Modern Chollet (2021) ConvNet:** A modern deep baseline utilizing two convolutional stages with Rectified Linear Unit (ReLU) activations, max pooling, dropout regularization (0.5), and dense classification.

The study compares parameter efficiency, convergence behavior across 15 training epochs, test accuracy, cross-entropy loss, confusion matrices, and training times.

---

## 2. Repository Structure

```
├── data/
│   └── mnist_data.npz              # Cached MNIST dataset (60,000 train, 10,000 test)
├── outputs/
│   ├── figures/                    # High-resolution benchmark figures
│   │   ├── figure1_graphical_abstract.png
│   │   ├── figure2_learning_curves.png
│   │   └── figure3_confusion_matrices.png
│   ├── comparative_metrics.csv     # Tabulated performance and parameter metrics
│   ├── chollet_2021_model.pt       # Trained weights for modern ConvNet
│   └── lecun_1989_model.pt         # Trained weights for historical CNN
├── src/
│   ├── models.py                   # PyTorch neural network architectures
│   ├── train_evaluate.py           # Benchmark training, evaluation, and plotting script
│   └── inference_demo.py           # Interactive test inference CLI script
├── requirements.txt                # Python package dependencies
└── README.md                       # Technical documentation
```

---

## 3. Installation and Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed
- Virtual environment recommended

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/rajifaruq2021-hue/cv_handwritten_digit_recognition.git
cd cv_handwritten_digit_recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. How to Run

### A. Run Benchmark Experiments and Re-train Models
```bash
python src/train_evaluate.py
```
This script will:
- Load the 70,000-image MNIST dataset.
- Train both networks for 15 epochs.
- Log comparative accuracy, loss, and training times.
- Save summary figures in `outputs/figures/`.

### B. Run Interactive Digit Inference
```bash
python src/inference_demo.py
```
Demonstrates real-time digit classification with probability confidence scores on unseen test samples.

---

## 5. Comparative Experimental Results

| Model | Architecture Type | Parameters | Epochs | Training Time (s) | Train Accuracy | Test Accuracy | Test Loss |
|---|---|---|---|---|---|---|---|
| **Chollet (2021)** | Modern Deep CNN (ReLU + MaxPool + Dropout 0.5) | 34,826 | 15 | 929.69 s | 98.92% | **99.11%** | 0.0260 |
| **LeCun et al. (1989)** | Historical 4-Layer (Tanh + Strided Conv + Dropout 0.25) | 10,024 | 15 | **171.38 s** | 97.63% | **98.30%** | 0.0559 |

### Key Findings
- The historical LeCun et al. (1989) network achieves **98.30% test accuracy**, reaching 99.18% of the modern architecture's performance while requiring **71.2% fewer trainable parameters** (10,024 vs. 34,826).
- Due to its lightweight parameter structure and strided convolutions, the historical model trains **5.4 times faster** per epoch than the modern network on CPU hardware.
- Both models demonstrate excellent generalization with minimal overfitting, confirming the enduring efficacy of weight sharing and local receptive fields originally introduced in 1989.
