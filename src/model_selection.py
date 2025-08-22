import torch
from transformers import AutoModelForCausalLM
from custom_head import FrozenLlamaClassifier

def load_base_model(config, pos_weight=None):
    # obtain required variables from config file
    model_name = config["model"]["name"]
    train_type = config["model"]["train_mode"]
    num_labels = config["model"].get("num_labels", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True
    ).to(device)

    # training mode logic
    if train_type == "partial":
        # Freeze entire base model
        for param in base_model.parameters():
            param.requires_grad = False

        #  Unfreeze a specified amount of layers
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            layers = base_model.model.layers
            unfreeze_count = config["model"].get("partial_unfreeze_layers", 2)
            for layer in layers[-unfreeze_count:]:
                for param in layer.parameters():
                    param.requires_grad = True

        classifier = FrozenLlamaClassifier(
            base_model, 
            pos_weight=pos_weight,
        )

    elif train_type == "full":
        classifier = FrozenLlamaClassifier(
            base_model, 
            pos_weight=pos_weight,
        )

    else:  
        for param in base_model.parameters():
                param.requires_grad = False
        classifier = FrozenLlamaClassifier(
            base_model, 
            num_labels=num_labels, 
            pos_weight=pos_weight,
        )
    # Set model on gpu
    classifier.to(device)
    return classifier
