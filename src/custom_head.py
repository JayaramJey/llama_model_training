# import torch
# import torch.nn as nn

# class FrozenBertClassifier(nn.Module):
#     def __init__(self, base_model, num_labels=5, pos_weight=None):
#         super().__init__()
#         self.base_model = base_model

#         hidden_size = getattr(base_model.config, "hidden_size", None)
#         if hidden_size is None:
#             raise ValueError("Could not find hidden_size from base_model.config")

#         self.dropout = nn.Dropout(0.2)
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.ReLU(),
#             nn.LayerNorm(128, eps=1e-5),  # LayerNorm in float32 for stability
#             nn.Dropout(0.3),
#             nn.Linear(128, num_labels)
#         )

#         # Store pos_weight as buffer if provided
#         if pos_weight is not None:
#             pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
#             self.register_buffer("pos_weight_buf", pos_weight)
#         else:
#             self.pos_weight_buf = None

#         self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf)

#     def forward(self, input_ids, attention_mask=None, labels=None):
#         # Forward through the base model
#         outputs = self.base_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True,
#             output_hidden_states=True
#         )

#         # Use last token of last hidden state
#         pooled_output = outputs.hidden_states[-1][:, -1, :]

#         # Cast pooled_output to float32 for stable classifier computations
#         pooled_output = pooled_output.to(torch.float32)

#         # Apply dropout and classifier
#         x = self.dropout(pooled_output)
#         logits = self.classifier(x)

#         # Compute loss if labels are provided
#         if labels is not None:
#             labels = labels.float()
#             loss = self.loss_fn(logits, labels)
#             return {"loss": loss, "logits": logits}

#         return logits


import torch
import torch.nn as nn

class FrozenBertClassifier(nn.Module):
    def __init__(self, base_model, num_labels=5, pos_weight=None):
        super().__init__()
        self.base_model = base_model

        # Freeze base model parameters to save memory
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Determine hidden size dynamically
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not determine hidden_size from base_model")

        # Smaller classification head for memory efficiency
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),  # Reduced from 128
            nn.ReLU(),
            nn.LayerNorm(64, eps=1e-5),  # Reduced from 128
            nn.Dropout(0.3),
            nn.Linear(64, num_labels)    # Reduced from 128
        )

        # Pos weight buffer
        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight_buf", pos_weight)
        else:
            self.pos_weight_buf = None

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf)

        print(f"Initialized classifier with hidden_size={hidden_size}, num_labels={num_labels}")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Use no_grad context for frozen base model to save memory
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # GPT models → last token hidden state
        if hasattr(outputs, "hidden_states"):
            # last layer hidden states, last token
            pooled_output = outputs.hidden_states[-1][:, -1, :].clone().detach()
        elif hasattr(outputs, "logits"):
            pooled_output = outputs.logits[:, -1, :].clone().detach()
        else:
            raise ValueError("Unexpected model output type")

        # Re-enable gradients for classification head
        pooled_output.requires_grad_(True)
        pooled_output = pooled_output.to(torch.float32)
        
        x = self.dropout(pooled_output)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            labels = labels.float()
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}