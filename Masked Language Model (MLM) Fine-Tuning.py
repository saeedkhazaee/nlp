# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:39:04 2023

@author: Saeed
"""
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length')
        # We are going to predict masked tokens
        encoding["labels"] = encoding["input_ids"].clone()
        # Create mask
        rand = torch.rand(encoding["input_ids"].shape)
        mask_arr = (rand < 0.15) * (encoding["input_ids"] != 101) * (encoding["input_ids"] != 102) * (encoding["input_ids"] != 0)
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        encoding["input_ids"][selection] = 103  # 103 is the token id for [MASK]
        return encoding





from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments

# Assuming documents is your list of document strings
documents = ["Document 1...", "Document 2...", "Document 3..."]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = TextDataset(documents, tokenizer)

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    gradient_accumulation_steps=2,   # number of updates steps to accumulate before performing a backward/update pass
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset,               # training dataset
)

trainer.train()
