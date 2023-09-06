#!/bin/bash
nohup CUDA_VISIBLE_DEVICES=0 python3 /home/jisukim/LLMscience/llm/finetune.py --config /home/jisukim/LLMscience/llm/configs/config_orca_mini_v3_7b.yaml > output.log &