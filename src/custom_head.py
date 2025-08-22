# Import PyTorch core functionality
import torch
import torch.nn as nn

# Import a standardized output format for sequence classification models
from transformers.modeling_outputs import SequenceClassifierOutput

# Define a custom classifier that wraps a frozen LLaMA model
class FrozenLlamaClassifier(nn.Module):
    def __init__(self, base_model, pos_weight=None):
        super().__init__()  # Initialize the parent nn.Module class

        # store the base model
        self.base_model = base_model  
        
        # Get the hidden size from the base model's config
        hidden_size = base_model.config.hidden_size

        # Define a sequental classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),              # Helps prevent overfitting
            nn.Linear(hidden_size, 5)       # Maps hidden states to 5 output logits
        )

        # Register pos_weight as a buffer to save to the model and also on the correct device
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Pass inputs through the base model to get hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,              # Return outputs as a dictionary
            output_hidden_states=True      # Include all hidden layers
        )

        # Get the last hidden state
        last_hidden = getattr(outputs, "last_hidden_state", None) or (
            outputs.hidden_states[-1] if outputs.hidden_states else None
        )

        # If attention mask is provided, determine a weighted average of hidden states
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())  # Match hidden size
            pooled_output = (last_hidden * expanded_mask).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            # Otherwise, just average across all tokens
            pooled_output = last_hidden.mean(dim=1)

        # Pass pooled output through classifier and clamp logits to avoid extreme values
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # binary cross-entropy loss with optional positive class weighting
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
            loss = loss_fn(logits, labels.float())  # Convert labels to float for BCE loss

        # Return standardized output containing loss and logits
        return SequenceClassifierOutput(loss=loss, logits=logits)