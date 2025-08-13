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
trainer.train()

# Save model
os.makedirs("output", exist_ok=True)
torch.save({
    'base_model_state_dict': model.base_model.state_dict(),
    'classifier_state_dict': model.classifier.state_dict(),
}, "output/frozen_bert.pt")

# Evaluate
trainer.evaluate()
