#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python3 /home/jisukim/LLMscience/llm/finetune.py --config /home/jisukim/LLMscience/llm/configs/config_orca_mini_v3_7b.yaml > output.log &