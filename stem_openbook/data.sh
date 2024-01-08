#!/bin/bash
## preprocess가 완료되어 faiss index가 만들어진 후, postprocess를 진행해야 합니다. postprocess를 수행한 결과 추출된 ctx에 대한 csv파일이 만들어집니다.
# nohup python3 preprocess.py --config_path /home/jisukim/LLMscience/openbook/stem_faiss/configs/bge-base.yaml > bge.log &
nohup python3 postprocess.py --config_path /home/jisukim/LLMscience/openbook/stem_faiss/configs/gen-bge-dataset.yaml > bge-dataset.log &
