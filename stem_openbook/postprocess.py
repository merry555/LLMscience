import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import argparse
import yaml
from tqdm import tqdm
import warnings
import json
import gc
import torch
import numpy as np
from dataset import PromptDataset
from torch.utils.data import DataLoader

gc.collect()

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess dataset")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml",help="config dir")
    args = parser.parse_args()
    return args


def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def get_ctx(data_loader, config, dtype):
    device = torch.device(config['model']['device'])
    context = pd.read_parquet(config["data"]["wiki_path"])

    dataset = pd.read_csv(config["data"][f"{dtype}_path"])
    index = faiss.read_index(config['data']['faiss_output_path'])

    ## Encoding
    st_encoder = SentenceTransformer(config['model']['model_name']).to(device)

    ## Dataset 저장
    l12_v2_dataset = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader)):
            prompt_encode = st_encoder.encode(batch)

            ## Top 40개 wiki 추출
            _, indices = index.search(prompt_encode, 40)

            for i, value in enumerate(indices):
                l12_v2_dataset.append(
                    '!?!?'.join(context.iloc[value]['ctx'].tolist())
                )


    dataset['ctx'] = l12_v2_dataset

    dataset.to_csv(config['data'][f'{dtype}_ctx_path'])


def main(config):
    dtype = config['data']['dtype']
    dataset = PromptDataset(config['data'][f'{dtype}_path'])
    data_loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, drop_last=False)

    get_ctx(data_loader, config, dtype)



if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)