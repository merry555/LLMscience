import pandas as pd
import os
import argparse
import yaml
from tqdm import tqdm
from datachunk import DataChunk
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer
import numpy as np
from dataset import ParquetDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml",help="config dir")
    args = parser.parse_args()
    return args


def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def preprocess(config):
    folder_path = config['data']['folder_path']
    file_list = os.listdir(folder_path)
    # 각 파일에 대해 작업
    for file_name in file_list:
        file = file_name.split('.')[0]
        if file_name.endswith('.parquet'):
            file_path = os.path.join(folder_path, file_name)
            
            # CSV 파일을 데이터 프레임으로 읽어오기
            df = pd.read_parquet(file_path)
            
            # "text" 컬럼에서 "==References=="로 split하고 0번째 텍스트를 가져와서 다시 "text" 컬럼에 저장
            df['text'] = df['text'].str.split('==References==').str[0]
            
            # 수정된 데이터 프레임을 리스트에 추가
            df.to_parquet(f'{folder_path}/{file}.parquet', index=False)
            del df

def chunk_data(config):
    folder_path = config['data']['folder_path']
    file_list = os.listdir(folder_path)
    output_folder_path = config['data']['output_folder_path']

    data_chunk = DataChunk(config)

    idx = 0

    # title + text: 512로 자른 데이터 다시 저장 & index 재 부여
    for file_name in file_list:
        if file_name.endswith('.parquet'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_parquet(file_path)
            
            chunk_text = data_chunk.chunk(df)
            to_save = {idx + i: chunk_text[i] for i in range(len(chunk_text))}
            
            to_save = {idx + i: chunk_text[i] for i in range(len(chunk_text))}
            to_save_parquet = pd.DataFrame(list(to_save.values()), columns=['text'])
            
            to_save_parquet.to_parquet(f'{output_folder_path}/{idx}-{idx+len(chunk_text)-1}.parquet')
            
            idx += len(chunk_text)

            del to_save_parquet

def faiss_index(config):
    folder_path = config['data']['output_folder_path']
    device = torch.device(config['model']['device'])
    index = faiss.IndexFlatIP(config['model']['word_embedding_dimension'])
    index = faiss.IndexIDMap2(index)

    st_encoder = SentenceTransformer(config['model']['model_name']).to(device)    

    dataset = ParquetDataset(folder_path)
    data_loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False)

    with torch.no_grad():
        cnt = 0
        for i, batch in tqdm(enumerate(data_loader)):
            st_output = st_encoder.encode(batch)

            index.add_with_ids(st_output.cpu().numpy().astype('float32'), np.array(range(cnt, cnt + st_output.shape[0])))
            cnt+=st_output.shape[0]

    faiss.write_index(index, config['data']['faiss_output_path'])
    print(index.ntotal)
    print("n total ===============")

def main(config):
    ## 1. 데이터 전처리
    # preprocess(config)

    ## 2. Sentece Transformer 들고와서 tokenizer한 다음 512로 잘라서 다시 저장
    # chunk_data(config)

    ## 3. Faiss index 만들기
    faiss_index(config)

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)