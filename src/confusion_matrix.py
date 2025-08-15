# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
# from data_prep import get_datasets

# def plot_confusion_matrices(labels, predictions, label_names):
#     cm_list = multilabel_confusion_matrix(labels, predictions)
    
#     for i, name in enumerate(label_names):
#         tn, fp, fn, tp = cm_list[i].ravel()
#         cm_array = np.array([[tn, fp],
#                              [fn, tp]])
#         plt.figure() 
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm_array,
#                                       display_labels=["Pred 0", "Pred 1"])
#         disp.plot(cmap="Blues")
#         plt.title(f"Confusion Matrix - {name}")
#         plt.show(block=True)

    

import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_and_plot(model, dataloader, label_names, device="cpu"):
    """
    Evaluates a trained model and plots confusion matrices for each label.

    Args:
        model: Trained PyTorch model (with multi-label output).
        dataloader: PyTorch DataLoader that yields dicts with keys:
                    'input_ids', 'attention_mask', 'labels'.
        label_names: List of label names, e.g. ["anger", "fear", ...]
        device: 'cpu' or 'cuda'
    """
    model.eval()
    model.to(device)
    
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs and labels to the device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Combine all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Apply sigmoid to get probabilities, then threshold at 0.5
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs > 0.5).astype(int)
    labels = all_labels.numpy()

    # Compute multilabel confusion matrices
    cm_list = multilabel_confusion_matrix(labels, preds)
    
    # Plot each label's confusion matrix
    for i, name in enumerate(label_names):
        tn, fp, fn, tp = cm_list[i].ravel()
        cm_array = np.array([[tn, fp],
                             [fn, tp]])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_array,
                                      display_labels=["Pred 0", "Pred 1"])
        plt.figure(figsize=(4,4))
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.show()
