from sklearn.metrics import f1_score
import numpy as np
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert to numpy
    logits = np.array(logits)
    labels = np.array(labels)

    logits, labels = eval_pred
    # using the sigmoid function to make classifications
    probabilites = torch.sigmoid(torch.tensor(logits)).numpy()
    # Add a threshold to choose whether the output prediction is 0 or 1
    predictions = (probabilites > 0.5).astype(int) 


    # Handle edge cases

    # Compute metrics
    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)

    return {
        "eval_f1_micro": f1_micro,
        "eval_f1_macro": f1_macro,
        "eval_f1": f1_micro  # or macro depending on your config
    }
