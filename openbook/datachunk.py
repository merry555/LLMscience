
from transformers import AutoTokenizer
from tqdm import tqdm

class DataChunk:
    def __init__(self, config):
        self.chunk_size = config['data']['chunk_size']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
        self.config = config

    def chunk(self, dataframe):
        orig_text = []
        dataframe = dataframe.reset_index(drop=True)
        for i in tqdm(range(len(dataframe))):
            encoded_title = self.tokenizer.encode(dataframe['title'][i])
            encoded_txt = self.tokenizer.encode(dataframe['text'][i])
            if len(encoded_txt) < 5:  # 본문 길이가 subword 5개 미만인 경우 패스
                continue

            # article마다 chunk_size 길이의 chunk를 만들어 list에 append. 각 chunk에는 title을 prepend합니다.
            for start_idx in range(0, len(encoded_txt), self.chunk_size):
                end_idx = min(len(encoded_txt), start_idx + self.chunk_size)
                chunk = encoded_title + encoded_txt[start_idx:end_idx]
                orig_text.append(self.tokenizer.decode(chunk))
        
        return orig_text