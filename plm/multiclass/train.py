from typing import Optional, Union
import pandas as pd
import numpy as np
# from colorama import Fore, Back, Style
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import gc
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import *
import sklearn
import argparse
import yaml
from dataset import CustomDataset
import wandb

import os
import warnings
warnings.filterwarnings(action='ignore')

import torch, gc
gc.collect()
torch.cuda.empty_cache()

wandb.login()
wandb.init()

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

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



def train(model, train_loader, valid_loader, config):
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
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(16,30+1):
        print(f'{epoch} epochs train start!!')
        model.train() # 학습시킴.

        loss_list = []

        for idx, batch in tqdm(enumerate(train_loader)):
            with torch.cuda.amp.autocast():
                input_ids = batch['input_ids'].to(device)
                attention_masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                loss = output.loss

                # output2 = model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)

                # # 각각의 모델 별 loss mean & kl_loss 생성
                # ce_loss = 0.5 * ( output['loss'] + output2['loss'])
                # kl_loss = compute_kl_loss(output['logits'], output2['logits'])

                # loss = ce_loss + 0.5 * kl_loss

            scaler.scale(loss).backward()
            if idx % config['model']['accumulation_steps'] == 0 or idx == len(train_loader) - 1: #파라미터 찾으려고 스케줄러 실행해주는데
                                        # 매스텝마다 실행시키면 시간이 오래걸려서, 일정 조건을 주고 하도록 ex) 4번에 1번
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()


            train1_map3_rst = map_at_3(output.logits.detach().cpu().tolist(), labels.detach().cpu().tolist())
            # train2_map3_rst = map_at_3(output2['logits'].detach().cpu().tolist(), labels.detach().cpu().tolist())


            wandb.log({"train loss": loss})
            wandb.log({"train1 map3@": train1_map3_rst})
            # wandb.log({"train2 map3@": train2_map3_rst})

        torch.save(model.state_dict(), f'/home/jisukim/LLMscience/plm/multiclass/output/' + f"{epoch}_epochs.bin")
        try:
            model.save_pretrained(f'./output/{epoch}')
        except:
            print('none!')
        print('model save complete!')

        with torch.no_grad():
            avg_map = []

            for idx, batch in enumerate(valid_loader):
                with torch.cuda.amp.autocast():
                    input_ids = batch['input_ids'].to(device)
                    attention_masks = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    output = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)

                map3_rst = map_at_3(output.logits.detach().cpu().tolist(), labels.detach().cpu().tolist())    
                avg_map.append(map3_rst)        
                wandb.log({"valid map3@": map3_rst})

            avg = np.mean(avg_map)
            wandb.log({"avg valid map3@": avg})



def main(config):
    device = torch.device(config['model']['device'])

    tokenized_train_dataset = CustomDataset(config, 'train')
    train_dataloader = DataLoader(tokenized_train_dataset, batch_size=16, shuffle=True, drop_last=True)

    tokenized_valid_dataset = CustomDataset(config, 'valid')
    valid_dataloader = DataLoader(tokenized_valid_dataset, batch_size=16, shuffle=False, drop_last=False)

    model = AutoModelForSequenceClassification.from_pretrained("/home/jisukim/LLMscience/plm/multiclass/output/5", num_labels=5).to(device)

    train(model, train_dataloader, valid_dataloader, config)

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)
    gc.collect()