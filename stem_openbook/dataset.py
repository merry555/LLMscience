import os
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class STEMDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.texts = []

        for i in tqdm(range(len(self.df))):
            tot_text = "[CLS]" + str(self.df.iloc[i]['title']) + "[SEP]" + str(self.df.iloc[i]['section']) + "[SEP]" + str(self.df.iloc[i]['text'])
            self.texts.append(tot_text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    

class PromptDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.texts = []

        for i in tqdm(range(len(self.df))):
            tot_text = "[CLS]" + str(self.df.iloc[i]['prompt']) + "[SEP]" + str(self.df.iloc[i]['A']) + \
                        "[SEP]" + str(self.df.iloc[i]['C']) +"[SEP]" + str(self.df.iloc[i]['D']) + \
                            "[SEP]" + str(self.df.iloc[i]['E'])
            self.texts.append(tot_text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]