import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModel

class LLMScienceForMultipleChoice(nn.Module):
    def __init__(self, config):
        super(LLMScienceForMultipleChoice, self).__init__()
        self.deberta = AutoModel.from_pretrained(config['model']['model_name'])
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(1024*5, 1024),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.Linear(512, 5)
        )
        self.device = torch.device(config['model']['device'])

    def forward(
        self,
        input_ids,
        attention_mask,
        labels
    ):
        # print(input_ids.shape) # torch.Size([16, 5, 512])
        first_pooler = self.deberta(input_ids=input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :]).last_hidden_state[:, 0, :]
        second_pooler = self.deberta(input_ids=input_ids[:, 1, :], attention_mask=attention_mask[:, 1, :]).last_hidden_state[:, 0, :]
        thrid_pooler = self.deberta(input_ids=input_ids[:, 2, :], attention_mask=attention_mask[:, 2, :]).last_hidden_state[:, 0, :]
        forth_pooler = self.deberta(input_ids=input_ids[:, 3, :], attention_mask=attention_mask[:, 3, :]).last_hidden_state[:, 0, :]
        fifth_pooler = self.deberta(input_ids=input_ids[:, 4, :], attention_mask=attention_mask[:, 4, :]).last_hidden_state[:, 0, :]

        concatenate_pooler = torch.cat((first_pooler, second_pooler, thrid_pooler, forth_pooler, fifth_pooler), dim=1)

        logits = self.classifier(concatenate_pooler)
        labels = labels.to(self.device)
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return {
            'preds': logits,
            'loss': loss
        }
