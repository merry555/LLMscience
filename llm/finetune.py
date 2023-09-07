import os

import numpy as np
import pandas as pd

from datasets import load_dataset, Dataset

import torch

import transformers
from transformers import LlamaForCausalLM, BitsAndBytesConfig, LlamaTokenizerFast, AutoTokenizer
from transformers import TrainingArguments
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
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

def find_all_linear_names(model):
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, bnb.nn.Linear4bit):
      names = name.split(".")
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])

  if "lm_head" in lora_module_names:  # needed for 16-bit
    lora_module_names.remove("lm_head")
  return list(lora_module_names)


def format_text(example):
    template = """
    ### System:
    You are an AI assistant that follows instruction extremely well. Help as much as you can.
    
    ### User: Choose the following multiple choice answer by giving the most appropriate response. Answer should be one among [A, B, C, D, E]
    Question: {prompt}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nE) {e}
    
    ### Assistant: {answer}"""

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

    model = LlamaForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=False
            )
    
    model = prepare_model_for_int8_training(model)    
    
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["up_proj", "gate_proj", "down_proj"], # , ['k_proj', 'down_proj', 'up_proj', 'o_proj', 'v_proj', 'gate_proj', 'q_proj']
        task_type="CAUSAL_LM"
    )

    # reference: https://platypus-llm.github.io/

    model = get_peft_model(model, config)

    def tokenize(prompt, add_eos_token=True):
       result = tokenizer(
          prompt['text'],
          truncation=True,
          max_length=2048,
          padding=False,
          return_tensors=None
       )

       if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < 2048
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result
       
    
    train_data = train.map(tokenize)
    valid_data = valid.map(tokenize)
    
    
    training_args = TrainingArguments(
                        output_dir="./Orca-mini-7b", 
                        overwrite_output_dir=True,
                        save_total_limit=2,
                        evaluation_strategy="steps",
                        warmup_ratio=0.8,
                        eval_steps=500,
                        logging_steps=500,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=8,
                        num_train_epochs=5,
                        lr_scheduler_type="cosine",
                        learning_rate=2e-4,
                        save_strategy='steps',
                        optim="adamw_torch",
                        fp16=True,
                        run_name="baseline-orca-sft",
                        report_to="wandb",
                        
                    )
    

    supervised_finetuning_trainer = transformers.Trainer(
                                        model,
                                        train_dataset=train_data.remove_columns(['A','prompt', 'B', 'C', 'D', 'E', 'answer', 'text']),
                                        eval_dataset=valid_data.remove_columns(['A','prompt', 'B', 'C', 'D', 'E', 'answer', 'text']),
                                        args=training_args,
                                        tokenizer=tokenizer,
                                        data_collator=transformers.DataCollatorForSeq2Seq(
                                            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                        ),
                                    )
    
    supervised_finetuning_trainer.train()

    
    
if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    train(config)