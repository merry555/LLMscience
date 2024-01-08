import pandas as pd
import os
import argparse
import yaml
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer
import numpy as np
from dataset import STEMDataset
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

def faiss_index(config):
    folder_path = config['data']['wiki_path']
    device = torch.device(config['model']['device'])

    index = faiss.IndexFlatIP(config['model']['word_embedding_dimension'])
    index = faiss.IndexIDMap2(index)

    st_encoder = SentenceTransformer(config['model']['model_name']).to(device)

    dataset = STEMDataset(folder_path)
    data_loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, drop_last=False)

    with torch.no_grad():
        cnt = 0

        for i, batch in tqdm(enumerate(data_loader)):
            st_output = st_encoder.encode(batch)

            index.add_with_ids(st_output, np.array(range(cnt, cnt + st_output.shape[0])))
            cnt+=st_output.shape[0]

    faiss.write_index(index, config['data']['faiss_output_path'])

    print(index.ntotal)
    print("n total ===============")

def main(config):
    faiss_index(config)

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)