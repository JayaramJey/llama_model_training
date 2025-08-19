import os
import wandb
import torch
from transformers import (
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    AutoTokenizer,
    TrainingArguments,
)
from accelerate import Accelerator
from data_prep import get_datasets, get_pos_weight
from model_selection import load_base_model, compute_metrics
from custom_head import FrozenBertClassifier
import yaml

# Set memory allocation strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="wandb" if os.getenv("USE_WANDB", "false").lower() == "true" else None,
    )
    
    config = load_config("../config.yaml")  # Removed ../

    # print(config["model"]["name"])

    # Print setup info on main process
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPUs")
        print(f"Device: {accelerator.device}")
        print(f"Mixed precision: {accelerator.mixed_precision}")

    # Prepare datasets and class weights
    train_dataset, eval_dataset, eval_texts = get_datasets(config)
    pos_weight = get_pos_weight(train_dataset)

    # Load base model - let accelerate handle device placement
    with accelerator.main_process_first():
        base_model = load_base_model(config)
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

    # Wrap with custom classifier
    model = FrozenBertClassifier(base_model=base_model, num_labels=5, pos_weight=pos_weight)

    # Initialize W&B only on main process
    if accelerator.is_main_process and config["logging"]["use_wandb"]:
        wandb.init(
            project=config["logging"]["project"], 
            name=config["logging"]["run_name"],
            config=config
        )

    # Calculate effective batch size
    effective_batch_size = (
        config["training"]["batch_size"] * 
        accelerator.num_processes * 
        4  # gradient_accumulation_steps
    )
    
    if accelerator.is_main_process:
        print(f"Effective batch size: {effective_batch_size}")

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],  # now points to /home/jovyan/output/models/final_model
        eval_strategy="no",
        save_strategy="no",
        # save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=float(config["training"]["lr"]),
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        report_to="wandb" if config["logging"]["use_wandb"] and accelerator.is_main_process else "none",
        gradient_accumulation_steps=4,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        bf16=True,
        remove_unused_columns=False,
        logging_steps=10,
    )


    # Setup tokenizer
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], force_download=True)
        # Add padding token if missing (common for GPT models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Create trainer
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

    # Train the model
    if accelerator.is_main_process:
        print("Starting training...")
    
    trainer.train()
    trainer.save_model("output/models/final_model")
    trainer.evaluate()
    # Save final model only on main process
    if accelerator.is_main_process:
        print("Training completed. Saving model...")
        trainer.save_model()
        if config["logging"]["use_wandb"]:
            wandb.finish()

if __name__ == "__main__":
    main()