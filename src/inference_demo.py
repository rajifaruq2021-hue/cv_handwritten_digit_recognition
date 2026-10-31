"""
Interactive Digit Inference Demo CLI Tool
Course: DLBAIPCV01 - Project: Computer Vision
Author: Faruq Ishola Raji (4253139)

Loads the trained LeCun et al. (1989) Historical CNN and Chollet (2021) Modern ConvNet
and performs real-time digit classification on sample test images from MNIST.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F

from models import CholletModernConvNet, LeCun1989HistoricalCNN


def run_inference_demo():
    base_dir = r"D:\IU INTERNATIONAL UNIVERSITY\3 semester\Project - Computer Vision (DLBAIPCV)\Project Report - Handwritten Digit Recognition"
    data_path = os.path.join(base_dir, "data", "mnist_data.npz")
    output_dir = os.path.join(base_dir, "outputs")

    # Load test data
    data = np.load(data_path)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Load models
    chollet_model = CholletModernConvNet()
    chollet_model.load_state_dict(torch.load(os.path.join(output_dir, "chollet_2021_model.pt")))
    chollet_model.eval()

    lecun_model = LeCun1989HistoricalCNN()
    lecun_model.load_state_dict(torch.load(os.path.join(output_dir, "lecun_1989_model.pt")))
    lecun_model.eval()

    # Pick 5 diverse sample indices from test set
    sample_indices = [0, 1, 2, 3, 4]

    print("\n================ INTERACTIVE DIGIT INFERENCE DEMONSTRATION ================")
    for idx in sample_indices:
        img = X_test[idx:idx+1]
        true_label = int(y_test[idx].item())

        with torch.no_grad():
            out_c = chollet_model(img)
            prob_c = F.softmax(out_c, dim=1)
            pred_c = int(torch.argmax(prob_c, dim=1).item())
            conf_c = float(prob_c[0, pred_c].item())

            out_l = lecun_model(img)
            prob_l = F.softmax(out_l, dim=1)
            pred_l = int(torch.argmax(prob_l, dim=1).item())
            conf_l = float(prob_l[0, pred_l].item())

        print(f"\n[Test Sample #{idx+1}] True Label: {true_label}")
        print(f"  -> Chollet (2021) Modern Prediction:  Digit {pred_c} (Confidence: {conf_c:.2%})")
        print(f"  -> LeCun (1989) Historical Prediction: Digit {pred_l} (Confidence: {conf_l:.2%})")
    print("\n===========================================================================\n")


if __name__ == '__main__':
    run_inference_demo()
