import torch
import numpy as np
from transformers import AutoModel
from sklearn.metrics import multilabel_confusion_matrix, f1_score, ConfusionMatrixDisplay
from custom_head import FrozenBertClassifier
import matplotlib.pyplot as plt

# Training mode logic
def load_base_model(config):
    base_model = AutoModel.from_pretrained(config["model"]["name"])
    train_type = config["model"]["train_mode"]

    if train_type == "full":
        for param in base_model.parameters():
            param.requires_grad = False

    elif train_type == "partial":
        for param in base_model.parameters():
            param.requires_grad = False
        for layer in base_model.encoder.layer[-config["model"]["partial_unfreeze_layers"]:]:
            for param in layer.parameters():
                param.requires_grad = True

    elif train_type == "Finetune":
        for param in base_model.parameters():
            param.requires_grad = True

    return base_model

# Metrics calculation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # using the sigmoid function to make classifications
    probabilites = torch.sigmoid(torch.tensor(logits)).numpy()
    # Add a threshold to choose whether the output prediction is 0 or 1
    predictions = (probabilites > 0.5).astype(int) 

    # Create a multi label confusion matrix
    confusion_matrix = multilabel_confusion_matrix(labels, predictions)
    label_names = ["anger", "fear", "joy", "sadness", "surprise"]

    # Create a metrics dictionary to store all the data
    metrics = {}
    # for loop to run through all the different labels
    for i, name in enumerate(label_names):
        # place the values within their designated variables
        tn, fp, fn, tp = confusion_matrix[i].ravel()
        # 
        metrics[f"{name}_TP"] = int(tp)
        metrics[f"{name}_FP"] = int(fp)
        metrics[f"{name}_FN"] = int(fn)
        metrics[f"{name}_TN"] = int(tn)


    metrics["f1_micro"] = f1_score(labels, predictions, average="micro", zero_division=0)
    metrics["f1_macro"] = f1_score(labels, predictions, average="macro", zero_division=0)

    return metrics
