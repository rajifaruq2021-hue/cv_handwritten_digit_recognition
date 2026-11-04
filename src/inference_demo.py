"""
inference_demo.py

I am the interactive inference script for my Handwritten Digit Recognition project.
Course: DLBAIPCV01 - Project: Computer Vision
Author: Faruq Ishola Raji
Matriculation Number: 4253139

What I do:
1. I load the trained weights for both the Chollet (2021) model and the LeCun (1989) model.
2. I load test images from the MNIST dataset.
3. I run each test image through both models in real time.
4. I calculate and display the predicted digit alongside the model's confidence percentage.
"""

# I import the libraries I need to handle files, arrays, and PyTorch tensors
import os
import torch
import numpy as np
import torch.nn.functional as F

# I import the two neural network classes that I defined in models.py
from models import CholletModernConvNet, LeCun1989HistoricalCNN


def run_inference_demo():
    """
    I load the saved models and test their predictions on sample digits from the test set.
    """
    # I define the paths to my data and output folders
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "mnist_data.npz")
    output_dir = os.path.join(base_dir, "outputs")

    # I check that my trained model files exist before loading them
    chollet_weights_path = os.path.join(output_dir, "chollet_2021_model.pt")
    lecun_weights_path = os.path.join(output_dir, "lecun_1989_model.pt")

    if not os.path.exists(chollet_weights_path) or not os.path.exists(lecun_weights_path):
        print(f"Error: Could not find trained weights in {output_dir}")
        print("Please run train_evaluate.py first to train the models and save weights.")
        return

    # I load the MNIST test data from my local npz archive
    print("I am loading the test dataset.")
    data = np.load(data_path)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # I initialize the Chollet 2021 model and load its trained weights
    print("I am loading the Chollet (2021) modern model weights.")
    chollet_model = CholletModernConvNet()
    chollet_model.load_state_dict(torch.load(chollet_weights_path, map_location='cpu'))
    chollet_model.eval()  # I put the model into evaluation mode so dropout is turned off

    # I initialize the LeCun 1989 model and load its trained weights
    print("I am loading the LeCun (1989) historical model weights.")
    lecun_model = LeCun1989HistoricalCNN()
    lecun_model.load_state_dict(torch.load(lecun_weights_path, map_location='cpu'))
    lecun_model.eval()  # I put the model into evaluation mode

    # I select 5 diverse sample digits from the test set to demonstrate how both models perform
    sample_indices = [0, 1, 2, 3, 4]

    print("\n================ INTERACTIVE DIGIT INFERENCE DEMONSTRATION ================")
    print("Author: Faruq Ishola Raji (4253139) | Course: DLBAIPCV01")
    print("Comparing: Chollet (2021) Modern ConvNet vs LeCun et al. (1989) Historical CNN")
    print("===========================================================================")

    for idx in sample_indices:
        # I pick the single test image and its real handwritten digit label
        img = X_test[idx:idx+1]
        true_label = int(y_test[idx].item())

        # I turn off gradient calculations because I am only evaluating (inference mode)
        with torch.no_grad():
            # 1. I get the modern model's prediction
            out_c = chollet_model(img)
            prob_c = F.softmax(out_c, dim=1)
            pred_c = int(torch.argmax(prob_c, dim=1).item())
            conf_c = float(prob_c[0, pred_c].item())

            # 2. I get the historical model's prediction
            out_l = lecun_model(img)
            prob_l = F.softmax(out_l, dim=1)
            pred_l = int(torch.argmax(prob_l, dim=1).item())
            conf_l = float(prob_l[0, pred_l].item())

        # I print out the results clearly for comparison
        print(f"\n[Test Sample #{idx+1}] Real Label: {true_label}")
        print(f"  -> Chollet (2021) Prediction:  Digit {pred_c} (Confidence: {conf_c:.2%})")
        print(f"  -> LeCun (1989)   Prediction:  Digit {pred_l} (Confidence: {conf_l:.2%})")

    print("\n===========================================================================")
    print("I have successfully verified both models on the test images.\n")


# I run the demo when this file is executed directly
if __name__ == '__main__':
    run_inference_demo()
