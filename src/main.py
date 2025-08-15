import os
import wandb
import torch
from transformers import Trainer, DataCollatorWithPadding, EarlyStoppingCallback, AutoTokenizer
from data_prep import get_datasets, get_pos_weight
from model_selection import (
    load_base_model,  
    compute_metrics
)
import yaml
import pandas as pd
from custom_head import FrozenBertClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# from confusion_matrix import plot_confusion_matrices

# Load yaml file
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
config = load_config("../config.yaml")

# Prepare datasets and class weights
train_dataset, eval_dataset, eval_texts = get_datasets(config)
pos_weight = get_pos_weight(train_dataset)

# Load model with chosen training mode
base_model = load_base_model(config)
model = FrozenBertClassifier(base_model=base_model, num_labels=5, pos_weight=pos_weight)

# Weights & Biases setup
if config["logging"]["use_wandb"]:
    wandb.init(
        project=config["logging"]["project"],
        name=config["logging"]["run_name"],
    )

# Training arguments for HuggingFace Trainer
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir=config["training"]["output_dir"],
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=config["training"]["batch_size"],
    per_device_eval_batch_size=config["training"]["batch_size"],
    num_train_epochs=config["training"]["epochs"],
    learning_rate=float(config["training"]["lr"]),
    weight_decay=config["training"]["weight_decay"],
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    load_best_model_at_end=True,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    report_to="wandb" if config["logging"]["use_wandb"] else "no"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=AutoTokenizer.from_pretrained(config["model"]["name"]),
    data_collator=DataCollatorWithPadding(AutoTokenizer.from_pretrained(config["model"]["name"])),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    compute_metrics=compute_metrics,
)

# Train model
# trainer.train()

# Save model
os.makedirs("output", exist_ok=True)
torch.save({
    'base_model_state_dict': model.base_model.state_dict(),
    'classifier_state_dict': model.classifier.state_dict(),
}, "output/frozen_bert.pt")

# Evaluate
# trainer.evaluate()


label_names = ["anger", "fear", "joy", "sadness", "surprise"]

predictions_output = trainer.predict(eval_dataset)
logits, labels = predictions_output.predictions, predictions_output.label_ids
probs = torch.sigmoid(torch.tensor(logits)).numpy()
preds = (probs > 0.7).astype(int)

full_cm = np.zeros((5, 5), dtype=int)

for true_vec, pred_vec in zip(labels, preds):
    true_indices = np.where(true_vec == 1)[0]
    pred_indices = np.where(pred_vec == 1)[0]
    for i in true_indices:
        for j in pred_indices:
            full_cm[i, j] += 1


disp = ConfusionMatrixDisplay(confusion_matrix=full_cm, display_labels=label_names)
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format='d')
plt.title("Multilabel Unified Confusion Matrix")
plt.show()