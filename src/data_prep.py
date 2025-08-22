import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

label_names = ["anger", "fear", "joy", "sadness", "surprise"]

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

    datasets["train"] = datasets["train"].map(encode_labels)
    datasets["test"] = datasets["test"].map(encode_labels)
    eval_texts = datasets["test"]["text"]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fun(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = datasets.map(tokenize_fun, batched=True)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, eval_dataset, eval_texts

def get_pos_weight(train_dataset):
    labels_array = np.stack(train_dataset["labels"])
    weights = labels_array.sum(axis=0) / labels_array.sum()
    weights = np.max(weights) / weights
    return torch.tensor(weights, dtype=torch.float)