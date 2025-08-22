import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class FrozenLlamaClassifier(nn.Module):
    def __init__(self, base_model, pos_weight=None):
        super().__init__()
        self.base_model = base_model

        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, 5)
        )

        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(5))

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        last_hidden = getattr(outputs, "last_hidden_state", None) or (
            outputs.hidden_states[-1] if outputs.hidden_states else None
        )
        if last_hidden is None:
            raise ValueError("Missing hidden states from base model output")

        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())
            pooled_output = (last_hidden * expanded_mask).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            pooled_output = last_hidden.mean(dim=1)

        logits = torch.clamp(self.classifier(pooled_output), min=-30, max=30)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
            loss = loss_fn(logits, labels.float())

        return SequenceClassifierOutput(loss=loss, logits=logits)