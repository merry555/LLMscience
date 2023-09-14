import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModel

class LLMScienceForMultipleChoice(AutoModel):
    def __init__(self, config):
        super().__init__(config)

        self.deberta = AutoModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*5, 1)
        self.device = config['model']['device']

    def forward(
        self,
        input_ids,
        attention_masks,
        labels
    ):
        first_pooler = self.deberta(input_ids=input_ids[0], attention_masks=attention_masks[0]).pooler_output
        second_pooler = self.deberta(input_ids=input_ids[1], attention_masks=attention_masks[1]).pooler_output
        thrid_pooler = self.deberta(input_ids=input_ids[2], attention_masks=attention_masks[2]).pooler_output
        forth_pooler = self.deberta(input_ids=input_ids[3], attention_masks=attention_masks[3]).pooler_output
        fifth_pooler = self.deberta(input_ids=input_ids[4], attention_masks=attention_masks[4]).pooler_output

        concatenate_pooler = torch.cat((first_pooler, second_pooler, thrid_pooler, forth_pooler, fifth_pooler), dim=1)

        logits = self.classifier(concatenate_pooler)
        reshaped_logits = logits.view(-1, 5)

        labels = labels.to(self.device)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)

        return {
            'preds': reshaped_logits,
            'loss': loss
        }


