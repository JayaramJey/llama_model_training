import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# the label names for each emotion
label_names = ["anger", "fear", "joy", "sadness", "surprise"]

# Adding a label field in each dataset which includes all emotion labels
def encode_labels(example):
    result = []
    for label in label_names:
        example[label] = 1 if int(example[label]) > 1 else int(example[label])
        result.append(example[label])
    example["labels"] = result
    return example

def get_datasets(config):
    datasets = load_dataset(
        "csv",
        data_files={
            'train': ['../data/train1.csv', '../data/train2.csv'],
            'test': ['../data/test1.csv', '../data/test2.csv']
        }
    )
    # Encode labels
    datasets["train"] = datasets["train"].map(encode_labels)
    datasets["test"] = datasets["test"].map(encode_labels)

     # Keep eval texts
    eval_texts = datasets["test"]["text"]

    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        use_fast=False
    )
    # Use the end-of-sequence token as a substitute for the pad token if none is defined.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Function to tokeize all datasets
    def tokenize_fun(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = datasets.map(tokenize_fun, batched=True)

    # Assign datasets
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    # Convert to PyTorch tensors
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, eval_dataset, eval_texts

# Calculate pos_weight for loss function
def get_pos_weight(train_dataset):
    labels_array = np.stack(train_dataset["labels"])
    weights = labels_array.sum(axis=0) / labels_array.sum()
    weights = np.max(weights) / weights
    return torch.tensor(weights, dtype=torch.float)