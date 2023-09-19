from typing import Optional, Union
import pandas as pd
import numpy as np
# from colorama import Fore, Back, Style
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import gc
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import *
import sklearn
import argparse
import yaml
from model import DebertaV2ForMultipleChoice
from dataset import DataCollatorForMultipleChoice
import wandb
from utils import compute_kl_loss

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

# def preprocess(example):
#     options = 'ABCDE'
#     indices = list(range(5))

#     option_to_index = {option: index for option, index in zip(options, indices)}
#     index_to_option = {index: option for option, index in zip(options, indices)}

#     first_sentence = [ "[CLS] " + example['context'] ] * 5 
#     second_sentence = [" #### " + example['prompt'] + " [SEP] " + str(example[option]) + " [SEP]" for option in 'ABCDE']

#     tokenized_example = tokenizer(first_sentence, second_sentence, truncation='only_first')
#     tokenized_example['label'] = option_to_index[example['answer']]
#     return tokenized_example

def preprocess(example):
    options = 'ABCDE'
    indices = list(range(5))
    option_to_index = {option: index for option, index in zip(options, indices)}

    first_sentence = [example['prompt']] * 5
    second_sentences = [example[option] for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=True)
    tokenized_example['label'] = option_to_index[example['answer']]

    return tokenized_example


def evaluate(model, valid_loader, config):
    device = torch.device(config['model']['device'])

    model.eval() # 학습시킴.


    with torch.no_grad():
        avg_map = []

        for idx, batch in enumerate(valid_loader):
            with torch.cuda.amp.autocast():
                input_ids = batch['input_ids'].to(device)
                attention_masks = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)

            map3_rst = map_at_3(output['logits'].detach().cpu().tolist(), labels.detach().cpu().tolist())         
            avg_map.append(map3_rst)
            print("valid map3@: ", map3_rst)
        avg = np.mean(avg_map)
        print("avg valid map3@: ",avg)



def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
    valid_df = pd.read_csv(config['data']['valid']).fillna('None')

    tokenized_valid_dataset = Dataset.from_pandas(valid_df[[ 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer']]).map(preprocess, remove_columns=[ 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    valid_dataloader = DataLoader(tokenized_valid_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)

    model = DebertaV2ForMultipleChoice.from_pretrained("/home/jisukim/LLMscience/plm/output/epoch2").cuda()

    evaluate(model, valid_dataloader, config)

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)
    gc.collect()
