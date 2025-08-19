import torch
import numpy as np
from transformers import AutoModelForCausalLM
from sklearn.metrics import multilabel_confusion_matrix, f1_score, ConfusionMatrixDisplay
from custom_head import FrozenBertClassifier
import matplotlib.pyplot as plt

# Training mode logic
def load_base_model(config):
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        device_map=None,       # no manual device assignment
        torch_dtype=torch.bfloat16  # load in bf16 for memory efficiency
    )

    train_type = config["model"]["train_mode"]

    if train_type == "full":
        for param in base_model.parameters():
            param.requires_grad = False

    elif train_type == "partial":
        for param in base_model.parameters():
            param.requires_grad = False
        for layer in base_model.gpt_neox.layers[-config["model"]["partial_unfreeze_layers"] :]:
            for param in layer.parameters():
                param.requires_grad = True

    elif train_type == "Finetune":
        for param in base_model.parameters():
            param.requires_grad = True

    return base_model


# Metrics calculation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Ensure float32 for numeric stability
    probabilities = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
    predictions = (probabilities > 0.5).astype(int) 

    # confusion_matrix = multilabel_confusion_matrix(labels, predictions)
    # label_names = ["anger", "fear", "joy", "sadness", "surprise"]
    metrics = {}

    # for i, name in enumerate(label_names):
    #     tn, fp, fn, tp = confusion_matrix[i].ravel()
    #     metrics[f"{name}_TP"] = int(tp)
    #     metrics[f"{name}_FP"] = int(fp)
    #     metrics[f"{name}_FN"] = int(fn)
    #     metrics[f"{name}_TN"] = int(tn)

    metrics["f1_micro"] = f1_score(labels, predictions, average="micro", zero_division=0)
    metrics["f1_macro"] = f1_score(labels, predictions, average="macro", zero_division=0)

    return metrics
