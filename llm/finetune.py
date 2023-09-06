import os

import numpy as np
import pandas as pd

from datasets import load_dataset, Dataset

import torch

import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments

from peft import LoraConfig

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from langchain.prompts import PromptTemplate
import argparse
import yaml
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

wandb.login()
wandb.init()

def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml",help="config dir")
    args = parser.parse_args()
    return args


def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def format_text(example):
    template = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]

                Question: {prompt}\n
                A) {a}\n
                B) {b}\n
                C) {c}\n
                D) {d}\n
                E) {e}\n

                Answer: {answer}"""

    prompt = PromptTemplate(template=template, input_variables=['prompt', 'a', 'b', 'c', 'd', 'e', 'answer'])

    text = prompt.format(prompt=example['prompt'], 
                         a=example['A'], 
                         b=example['B'], 
                         c=example['C'], 
                         d=example['D'], 
                         e=example['E'], 
                         answer=example['answer'])
    
    return {"text": text}

def train(config):
    train = pd.read_csv(config['data']['train']).sample(frac=1).reset_index(drop=True)
    valid = pd.read_csv(config['data']['valid']).sample(frac=1).reset_index(drop=True)
    
    train_ds = Dataset.from_pandas(train)
    valid_ds = Dataset.from_pandas(valid)

    del train, valid

    train = train_ds.map(format_text)
    valid = valid_ds.map(format_text)

    model_id = config['model']['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    qlora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["up_proj", "down_proj"], # , "dense", "dense_h_to_4h", "dense_4h_to_h"
        task_type="CAUSAL_LM"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="auto"
            )
    
    training_args = TrainingArguments(
                        output_dir="./Orca-mini-7b", 
                        overwrite_output_dir=True,
                        save_total_limit=2,
                        evaluation_strategy="steps",
                        warmup_ratio=0.8,
                        eval_steps=500,
                        logging_steps=500,
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        num_train_epochs=5,
                        learning_rate=2e-4,
                        save_strategy='steps',
                        optim="paged_adamw_8bit",
                        fp16=True,
                        run_name="baseline-orca-sft",
                        report_to="wandb"
                    )
    

    supervised_finetuning_trainer = SFTTrainer(
                                        base_model,
                                        train_dataset=train,
                                        eval_dataset=valid,
                                        args=training_args,
                                        tokenizer=tokenizer,
                                        peft_config=qlora_config,
                                        dataset_text_field="text",
                                        max_seq_length=2048,
                                        data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, 
                                                                                    response_template="Answer:"),
                                    )
    
    supervised_finetuning_trainer.train()

    
    
if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    train(config)