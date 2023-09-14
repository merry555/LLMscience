from typing import Optional, Union
import pandas as pd
import numpy as np
# from colorama import Fore, Back, Style
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import gc
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel, EarlyStoppingCallback
import sklearn
import argparse
import yaml
from model import LLMScienceForMultipleChoice
from dataset import CustomDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml",help="config dir")
    args = parser.parse_args()
    return args


def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)


def train(model, train_loader, config):
    device = torch.device(config['model']['device'])
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(5 * len(train_loader) / config['model']['accumulation_steps'])
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-6,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1,5+1):
        print(f'{epoch} epochs train start!!')
        model.train() # 학습시킴.

        loss_list = []

        for idx, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                input_ids = batch['input_ids'].to(device)
                attention_masks = batch['attention_masks'].to(device)
                labels = batch['labels'].long.to(device)
                output = model(input_ids, attention_masks, labels)

            loss = output['loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            avg_loss = np.round(np.mean(loss_list), 4)


            print("Loss: ", avg_loss)

        torch.save(model.state_dict(), '/home/jisukim/LLMscience/plm/output' + f"{epoch}_epochs.bin")
        print('model save complete!')

        # with torch.no_grad():
        #     labels = []
        #     preds = []

        #     for idx, batch in enumerate(valid_loader):
        #         with torch.cuda.amp.autocast():
        #             pred, loss = model(batch['input_ids'], batch['attention_masks'], batch['labels'])
        #             label = batch['label']
        #             preds.append(pred.detach().cpu().numpy().tolist())
        #             labels.append(label.tolist())

        #     print('MAPE@: ', map_at_3(preds, labels))



def main(config):
    device = torch.device(config['model']['device'])
    train_dataset = CustomDataset(config, dtype="train")
    # valid_dataset = CustomDataset(config, dtype="valid")
    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=1,
                          pin_memory=False, drop_last=True)

    model = LLMScienceForMultipleChoice(config).to(device)
    train(model, train_dataloader, config)

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)
