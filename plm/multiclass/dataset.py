import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd

# 데이터셋 생성
class CustomDataset(Dataset):
    def __init__(self, config, dtype):
        if dtype == "train":
            self.dataset = pd.read_csv(config['data']['train']) ## csv 파일 형식
        else:
            self.dataset = pd.read_csv(config['data']['valid']) ## csv 파일 형식
            
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
        self.max_length = config['data']['max_length']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]

        first_sentence = "[CLS]" + row['prompt']
        second_sentences = [str(row[option]) for option in 'ABCDE']
        second_sentences = "[SEP]" + '[SEP]'.join(second_sentences)

        option_to_index = {option: idx for idx, option in enumerate('ABCDE')}

        labels = option_to_index[row['answer']]

        encode_output = self.tokenizer(first_sentence, second_sentences, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encode_output['input_ids'].squeeze(),
            'attention_mask': encode_output['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }