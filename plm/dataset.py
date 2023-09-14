import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

# 사용자 정의 Dataset 클래스 생성
class CustomDataset(Dataset):
    def __init__(self, config):
        self.dataset = pd.read_csv(config['data']['file_path']) ## csv 파일 형식
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.total_max_len = config['data']['max_length']

    def __getitem__(self, index):
        row = self.dataset.iloc[index]

        first_sentence = [ "[CLS] " + row['context'] ] * 5    
        second_sentences = [" #### " + row['prompt'] + " [SEP] " + row[option] + " [SEP]" for option in 'ABCDE']    
        
        ## 5개에 대한 prompt + context 정보
        encode_output = self.tokenizer(
            first_sentence, second_sentences, truncation='only_first', 
            max_length=self.total_max_len, add_special_tokens=False
        )

        option_to_index = {option: idx for idx, option in enumerate('ABCDE')}

        labels = option_to_index[row['answer']]

        return {
            "input_ids": encode_output['input_ids'].squeeze(),
            "attention_masks": encode_output['attention_masks'].squeeze(),
            "labels": torch.FloatTensor([labels]).squeeze()
        }
        
        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]