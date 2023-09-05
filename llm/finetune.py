import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

import torch

import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments

from peft import LoraConfig

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from langchain.prompts import PromptTemplate
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml",help="config dir")
    args = parser.parse_args()
    return args


def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

# Define your custom evaluation function
def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}

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

    train = train.map(format_text)
    valid = valid.map(format_text)

    model_id = config['model']['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    qlora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query_key_value"], # , "dense", "dense_h_to_4h", "dense_4h_to_h"
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
            device_map="cuda:0"
            )
    
    training_args = TrainingArguments(
                        output_dir="./Orca-mini-7b", 
                        per_device_train_batch_size=4,
                        per_device_eval_batch_size=8,
                        gradient_accumulation_steps=2,
                        learning_rate=2e-4,
                        logging_steps=20,
                        logging_strategy="steps",
                        max_steps=100,
                        optim="paged_adamw_8bit",
                        fp16=True,
                        run_name="baseline-orca-sft"
                    )
    
    supervised_finetuning_trainer = SFTTrainer(
                                        base_model,
                                        train_dataset=train["train"],
                                        eval_dataset=valid["train"],
                                        args=training_args,
                                        tokenizer=tokenizer,
                                        peft_config=qlora_config,
                                        dataset_text_field="text",
                                        max_seq_length=2048,
                                        data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, 
                                                                                    response_template="Answer:"),
                                        # compute_metrics=compute_metrics
                                    )
    
    supervised_finetuning_trainer.train()

    
    
if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    train(config)