from typing import Optional, Union
import pandas as pd
import numpy as np
# from colorama import Fore, Back, Style
from tqdm.notebook import tqdm
import torch
from datasets import Dataset
import gc
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel, EarlyStoppingCallback
import wandb
import os
from transformers import BatchEncoding

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# from torchnlp.nn import Attention #pip imstall pytorch-nlp


@dataclass
class DataCollatorForMultipleChoice:
  tokenizer: PreTrainedTokenizerBase
  padding: Union[bool, str, PaddingStrategy] = True
  max_length: Optional[int] = None
  pad_to_multiple_of: Optional[int] = None

  def __call__(self, features):
      label_name = 'label' if 'label' in features[0].keys() else 'labels'
      labels = [feature.pop(label_name) for feature in features]
      batch_size = len(features)
      num_choices = len(features[0]['input_ids'])
      flattened_features = [
          [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
      ]
      flattened_features = sum(flattened_features, [])

      batch = self.tokenizer.pad(
          flattened_features,
          padding=self.padding,
          max_length=self.max_length,
          pad_to_multiple_of=self.pad_to_multiple_of,
          return_tensors='pt',
      )
      batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
      batch['labels'] = torch.tensor(labels, dtype=torch.int64)
      return batch


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



def main():
    train = pd.read_csv("/home/jisukim/LLMscience/dataset/train_ctx.csv")
    valid = pd.read_csv("/home/jisukim/LLMscience/dataset/valid_ctx.csv")

    train = train.fillna('None')
    valid = valid.fillna('None')

    option_to_index = {
        option: idx for idx, option in enumerate('ABCDE')
    }
    index_to_option = {v:k for k,v in option_to_index.items()}

    model_path = "potsawee/longformer-large-4096-answering-race"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess(example):
      ctx_concat = example['ctx'].replace('!?!?',' ').replace('[CLS]',' ').replace('[SEP]',' ').replace('  ',' ')[:2100]
    
      ## context중 10개만 활용
      first_sentence = [ ctx_concat ] * 5
      second_sentences = [" #### " + example['prompt']  + str(example[option]) for option in 'ABCDE']
      tokenized_example = tokenizer(first_sentence, second_sentences, truncation="only_first")
      tokenized_example['label'] = option_to_index[example['answer']]
    
      return tokenized_example


    train_dataset = Dataset.from_pandas(train)
    valid_dataset = Dataset.from_pandas(valid)


    tokenized_train_dataset = train_dataset.map(preprocess, remove_columns=['prompt','ctx', 'A', 'B', 'C', 'D', 'E', 'answer'])
    tokenized_valid_dataset = valid_dataset.map(preprocess, remove_columns=['prompt', 'ctx', 'A', 'B', 'C', 'D', 'E', 'answer'])

    device = torch.device('cuda:0')

    model = AutoModelForMultipleChoice.from_pretrained(model_path).to(device)


    training_args = TrainingArguments(
        output_dir='./output_bge_deberta',
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        evaluation_strategy="steps",
        warmup_ratio=0.8,
        learning_rate=2e-6,
        eval_steps=500,
        logging_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        report_to=['wandb'],
        seed=42,
        metric_for_best_model='map@3',
        save_strategy='steps'
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        compute_metrics = compute_metrics,
    )


    trainer.train()

if __name__ == "__main__":
    main()
