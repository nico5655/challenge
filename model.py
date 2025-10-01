import torch.nn.functional as F
from torch import nn
import torch
from transformers import AutoModel
from config import ExperimentConfig


class CodeClassifierModel(nn.Module):
    def __init__(self,config,device):
        super().__init__()
        num_labels=len(config.data_labels)
        encoder=AutoModel.from_pretrained(config.model_name)
        self.encoder=encoder
        hidden=encoder.config.dim
        self.config=config
        num_texts=3 if config.use_source_code else 2
        if config.use_difficulty:
            self.difficulty_embedding=nn.Linear(1,10)
            self.projection=nn.Linear(num_texts*hidden+10,hidden)
        else:
            self.projection=nn.Linear(num_texts*hidden,hidden)

        self.proj_dropout=nn.Dropout(config.dropout_rate)
        self.classifier=nn.Linear(hidden,num_labels)
        self.thresholds=nn.Parameter(torch.ones(num_labels)*0.5,requires_grad=False)
        self.device=device
        self.to(device)

    def forward(self,batch):
        device=self.device
        desc={k:v.to(device) for k,v in batch['description'].items() if not 'raw' in k}
        inout_desc={k:v.to(device) for k,v in batch['in_out_description'].items() if not 'raw' in k}
        source_code={k:v.to(device) for k,v in batch['code'].items() if not 'raw' in k}
        x=[]
        fields=[desc,inout_desc]
        if self.config.use_source_code:
            fields.append(source_code)
        for enc_input in fields:
            if enc_input is None:
                continue
            output=self.encoder(**enc_input)
            mask=enc_input['attention_mask'].unsqueeze(-1).float()
            pooled_output=(output.last_hidden_state*mask).sum(dim=1)/(mask.sum(dim=1))
            x.append(pooled_output)
        if self.config.use_difficulty:
            diff=batch['difficulty'].unsqueeze(-1).to(device)
            diff_emb=self.difficulty_embedding(diff)
            x.append(diff_emb)
        x=torch.cat(x,dim=-1)
        x=F.relu(self.projection(x))
        x=self.proj_dropout(x)
        x=self.classifier(x)
        return x
    
    def inference(self,batch):
        self.eval()
        with torch.no_grad():
            logits=self.forward(batch)
            probs=F.sigmoid(logits)
        return (probs>self.thresholds.unsqueeze(0)).float()