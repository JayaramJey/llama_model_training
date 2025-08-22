import os
import gc
import yaml
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from transformers import TrainerCallback

# Set single GPU usage
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU

from model_selection import load_base_model
from data_prep import get_datasets, get_pos_weight
from metrics import compute_metrics

class MemoryCleanupCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# def debug_data_and_model(train_dataset, eval_dataset, model, tokenizer):

def main():
    # Load config
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    train_dataset, eval_dataset, eval_texts = get_datasets(config)

    # Load model with pos_weight
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = get_pos_weight(train_dataset)
    model = load_base_model(config, pos_weight=pos_weight).to(device)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=512
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch" if eval_dataset else "no",
        eval_steps=100,
        save_strategy="no",
        per_device_train_batch_size=config["training"].get("batch_size", 8),
        per_device_eval_batch_size=config["training"].get("batch_size", 8),
        gradient_accumulation_steps=1,
        num_train_epochs=config["training"]["epochs"],
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        learning_rate=float(config["training"]["lr"]),
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_f1_micro" if eval_dataset else None,
        greater_is_better=True if eval_dataset else None,
        report_to="none",
        logging_strategy="steps",
        logging_steps=50,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None,
        callbacks=[MemoryCleanupCallback(), EarlyStoppingCallback(early_stopping_patience=2)]
       
    )

    # cleanup_memory()
    trainer.train()

    # Save model weights
    torch.save(model.state_dict(), "custom_llama_classifier_weights.pth")

    # evaluation
    trainer.evaluate()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    # cleanup_memory()

if __name__ == "__main__":
    main()
