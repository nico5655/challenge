import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def read_record(file, config):
    with open(file, "r", encoding="utf-8") as f:
        data=json.load(f)
    X={}
    X['description']=str(data['prob_desc_description'])[:config.max_len_description]
    X['in_out_description']=f"Input: {data['prob_desc_input_spec']}\nOutput: {data['prob_desc_output_spec']}"[:config.max_len_in_out_description]
    X['source_code']=str(data['source_code'])[:config.max_len_code]
    X['difficulty']=float(data['difficulty'] if data['difficulty'] is not None else 0.0)
    if np.isnan(X['difficulty']):
        X['difficulty']=0.0
    Y={}
    for label in config.data_labels:
        Y[label]=int(label in data['tags'])

    return X,Y

def read_all_records(path, config):
    if type(path) is str:
        path=Path(path)
    big={}
    for p in sorted(path.rglob("*.json")):
        X,Y=read_record(p, config)
        for key in X:
            if key not in big:
                big[key]=[]
            big[key].append(X[key])
        for key in Y:
            if key not in big:
                big[key]=[]
            big[key].append(Y[key])
    return big


class CodeRecordDataset(Dataset):
    def __init__(self, config):
        folder_path=config.data_path
        self.folder_path = folder_path
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.config=config
        self.records = []
        self.labels = []
        if type(folder_path) is str:
            folder_path=Path(folder_path)
        for p in sorted(folder_path.rglob("*.json")):
            X, Y = read_record(p, config)
            self.records.append(X)
            self.labels.append(Y)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        label = self.labels[idx]
        
        # Encode the text fields using the tokenizer

        encoded_description = self.tokenizer(
            record['description'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        encoded_in_out_desc = self.tokenizer(
            record['in_out_description'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        encoded_code = self.tokenizer(
            record['source_code'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        label_tensor = torch.tensor([label[l] for l in self.config.data_labels], dtype=torch.float)
    
        item = {
            'description': {'input_ids': encoded_description['input_ids'].squeeze(0),
                'attention_mask': encoded_description['attention_mask'].squeeze(0),
                'raw_text': record['description']},
            'in_out_description': {'input_ids': encoded_in_out_desc['input_ids'].squeeze(0),
                'attention_mask': encoded_in_out_desc['attention_mask'].squeeze(0),
                'raw_text': record['in_out_description']},
            'code': {'input_ids': encoded_code['input_ids'].squeeze(0),
            'attention_mask': encoded_code['attention_mask'].squeeze(0),
            'raw_text': record['source_code']},
            'difficulty': torch.tensor(record['difficulty'], dtype=torch.float32),
            'labels': label_tensor
        }
        
        return item