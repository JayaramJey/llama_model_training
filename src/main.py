import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline,EarlyStoppingCallback, DataCollatorWithPadding, AutoModel
import numpy as np
import evaluate
from collections import Counter
import wandb
import torch
from torch.nn.functional import softmax
from custom_head import FrozenBertClassifier
import yaml
from sklearn.metrics import f1_score
import pandas as pd
from download import download_file
from sklearn.metrics import multilabel_confusion_matrix

# Load CSV files
datasets = load_dataset("csv", data_files={'train': ['../data/train1.csv','../data/train2.csv'], 'test': ['../data/test1.csv','../data/test2.csv'] })

# List of label names for multi-label classification
label_names = ["anger", "fear", "joy", "sadness", "surprise"]

# Adding a label field in each dataset which includes all emotion labels
def encode_labels(example):
    result = []
    for label in label_names:
        if int(example[label]) > 1:
            example[label] = 1
        else:
            example[label] = int(example[label])
        result.append(example[label])
    example["labels"] = result
    return example

# Applying the label encoding to both the training and test datasets
datasets["test"] = datasets["test"].map(encode_labels)
datasets["train"] = datasets["train"].map(encode_labels)

# Keep this before you tokenize
eval_texts = datasets["test"]["text"]

# Load yaml file
with open ("config.yaml", 'r') as config:
    config = yaml.safe_load(config)

# Tokenizer function 
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
def tokenize_fun(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# tokenizing all data
tokenized_datasets = datasets.map(tokenize_fun, batched=True)

# Assigning the evaluation and training datasets to variables
eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
train_dataset = tokenized_datasets["train"].shuffle(seed=42)

# # Convert datasets to pytorch so they can be worked with
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])



# train_dataset = train_dataset.map(convert_labels)
# eval_dataset = eval_dataset.map(convert_labels)
# Calculate class imbalance weights so they can be used loss function
weigh = np.stack(train_dataset["labels"]) # This line turns the column array into a numpy array to perform calculations on
weights = weigh.sum(axis=0) /weigh.sum() # determines how rare each label is 
weights = np.max(weights) / weights # gives a higher weighting to labels that are rare
pos_weight = torch.tensor(weights, dtype=torch.float) # converts these weights to tensors so they can be used with the model

#training arguments for the huggingface trainer 
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

# Loading the basemodel that was chosen in the yaml file
base_model = AutoModel.from_pretrained(config["model"]["name"])

# Determine which type of training the user wants
train_type = config["model"]["train_mode"]

# Training is split into full, partial, and Finetune
if train_type == "full":
    # Freeze all layers of the model
    for param in base_model.parameters():
        param.requires_grad = False
    
elif train_type == "partial":
    # Freeze all layers first
    for param in base_model.parameters():
        param.requires_grad = False

    # Unfreeze the speccified amount of layers
    for layer in base_model.encoder.layer[-config["model"]["partial_unfreeze_layers"]:]:
        for param in layer.parameters():
            param.requires_grad = True

elif train_type == "Finetune":
    # Don't freeze any layers
    for param in base_model.parameters():
        param.requires_grad = True

# Initializing the model with the custom classifier head, using the base model
model = FrozenBertClassifier(base_model=base_model, num_labels=5, pos_weight=pos_weight)

# Weights and bias setup
if config["logging"]["use_wandb"]:
    wandb.init(
        project=config["logging"]["project"],
        name=config["logging"]["run_name"],
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # using the sigmoid function to make classifications
    probabilites = torch.sigmoid(torch.tensor(logits)).numpy()
    # Add a threshold to choose whether the output prediction is 0 or 1
    predictions = (probabilites > 0.5).astype(int) 

    return {
        # return the macro and micro values
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }



# 
optimizer_alternate_parameters = [
    {"params": [p for n, p in model.named_parameters() if "encoder.layer.11" in n], "lr": 5e-4},
    {"params": [p for n, p in model.named_parameters() if "encoder.layer.10" in n], "lr": 3e-4},
    {"params": [p for n, p in model.named_parameters() if "encoder.layer.9" in n], "lr": 2e-4},
    {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": 5e-4},
]

# alternate learning rate trainer
class CustomTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer =  torch.optim.AdamW(optimizer_alternate_parameters)
        
        return self.optimizer

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    compute_metrics=compute_metrics,
)
import os

# Calling the trainer
trainer.train()
# CHoosing where to send the trained model
# if train_type == "full":
#     torch.save(model.state_dict(), "custom_head.pt")
# else:
os.makedirs("output", exist_ok=True)
torch.save({
    'base_model_state_dict': model.base_model.state_dict(),
    'classifier_state_dict': model.classifier.state_dict(),
}, "output/frozen_bert.pt")
    
# Evaluate the model
trainer.evaluate()
