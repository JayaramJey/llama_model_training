import torch
import torch.nn as nn

# Custom model which adds a classification head
class FrozenBertClassifier(nn.Module):
    def __init__(self, base_model, hidden_size=768, num_labels=5, pos_weight = None):
        super().__init__()
        # give the model the pretrained model with no head
        self.base_model = base_model

        # Add dropout to reduce overfitting
        self.dropout = nn.Dropout(0.2)
        # Custom classification head which provides final outputs
        self.classifier = nn.Sequential(
            # reduce the size of the input to the head
            nn.Linear(hidden_size, 128),
            # Help model learn patterns and not just straight lines
            nn.ReLU(),  
            # normalization
            nn.LayerNorm(128),
            # Another layer of dropout to reduce overfitting
            nn.Dropout(0.3),
            # This outputs the final logits for each label
            nn.Linear(128, num_labels)
        )
        # pos_weight = pos_weight.to(next(self.parameters()).device)
        # Loss function for multi label classification
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Provide the inputs to the base model
        outputs = self.base_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        
        # get the output from the base model
        cls_output = getattr(outputs, "pooler_output", outputs.last_hidden_state[:, 0, :])  
        
        # use the dropout layer to prevent overfitting
        x = self.dropout(cls_output)

        # feed the output to the head and get the output from the head
        logits = self.classifier(x)

        # If labels are provided, calculate loss
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            # return the loss and the output logits
            return {'loss': loss, 'logits': logits}
        
        # Otherwise just return logits (for inference)
        return logits
