import os
import pandas as pd
from torch.utils.data import Dataset

# 사용자 정의 Dataset 클래스 생성
class ParquetDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
        self.texts = []

        for file_name in self.parquet_files:
            file_path = os.path.join(self.folder_path, file_name)
            df = pd.read_parquet(file_path)
            self.texts.extend(df['text'].tolist())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]