import os
import pandas as pd
from torch.utils.data import Dataset

# 사용자 정의 Dataset 클래스 생성
class ParquetDataset(Dataset):
    def __init__(self, folder_path):
        file_list = os.listdir(folder_path)
        self.texts = []
        self.parquet_files = sorted(file_list, key=lambda x: int(x.split('-')[0]))
        print(self.parquet_files)

        for file_name in self.parquet_files:
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_parquet(file_path)
            self.texts.extend(df['text'].tolist())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
