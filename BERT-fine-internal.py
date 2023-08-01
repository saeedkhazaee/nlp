# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:18:26 2023

@author: Saeed
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}

# Sample data (replace this with your internal documents)
texts = [
    "This is a sample document.",
    "Another sample document.",
    "Yet another document."
]

# Create labels for the null class
labels = [0] * len(texts)  # Assign a label of 0 to all documents (null class)

# Create custom dataset and data loader
dataset = CustomDataset(texts, labels)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Fine-tuning loop
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in data_loader:
        inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    average_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1} - Loss: {average_loss}")

# After fine-tuning, you can use the model for various tasks
# For example, you can use it to predict the label for new documents:
model.eval()

new_texts = [
    "A new document to classify.",
    "Another new document."
]

inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
    predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

print("Predicted labels:", predicted_labels)
