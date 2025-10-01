
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import argparse
import os
import copy

from config import ExperimentConfig
from model import CodeClassifierModel
from data import CodeRecordDataset


def calculate_metrics(preds, true_labels):
    tp=(preds*true_labels).sum(dim=0)
    prec=tp/(preds.sum(dim=0)+1e-8)
    rec=tp/(true_labels.sum(dim=0)+1e-8)
    f1=2*prec*rec/(prec+rec+1e-8)
    accuracy=(preds==true_labels).float().mean(dim=0)
    return f1, prec, rec, accuracy

def run_epoch(model, dataloader, loss_function, optimizer=None):
    losses=[]
    all_preds=[]
    all_labels=[]
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    for batch in tqdm(dataloader):
        labels = batch.pop("labels")
        out = model(batch)
        loss = loss_function(out, labels.to(model.device))
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        losses.append(float(loss.detach().cpu().item()))
        all_preds.append(F.sigmoid(out).cpu())
        all_labels.append(labels.detach())
    avg_loss=np.array(losses).mean()
    all_preds=torch.cat(all_preds,dim=0)
    all_labels=torch.cat(all_labels,dim=0)
    return avg_loss, all_preds, all_labels

@torch.no_grad()
def run_inference(model, dataloader):
    all_preds=[]
    all_labels=[]
    model.eval()
    for batch in tqdm(dataloader):
        labels = batch.pop("labels")
        out = model.inference(batch)
        all_preds.append(out.cpu())
        all_labels.append(labels)
    all_preds=torch.cat(all_preds,dim=0)
    all_labels=torch.cat(all_labels,dim=0)
    return all_preds, all_labels

def train(config, train_dataset, val_dataset):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    num_labels=len(config.data_labels)
    model=CodeClassifierModel(config, device)
    num_items=torch.cat([item['labels'].unsqueeze(0) for item in train_dataset],dim=0).sum(dim=0)
    pos_weight = ((len(train_dataset) - num_items) / num_items.clamp(min=1.0)).to(device)
    print(f'Using pos_weight: {pos_weight}')
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    num_epochs=config.num_epochs
    train_dl=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    val_dl=DataLoader(val_dataset,batch_size=config.batch_size,shuffle=False)
    opt = Adam(model.parameters(), lr=config.learning_rate)

    train_losses=[]
    val_losses=[]
    best_val_loss=float('inf')
    best_epoch=-1
    for epoch in range(num_epochs):
        avg_train_loss, _, _ = run_epoch(model, train_dl, loss_function, opt)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - loss {avg_train_loss:.4f}")
        
        with torch.no_grad():
            avg_loss_val, probs, gts = run_epoch(model, val_dl, loss_function)
        val_losses.append(avg_loss_val)
        if avg_loss_val<best_val_loss:
            best_val_loss=avg_loss_val
            best_epoch=epoch
            torch.save(model.state_dict(), 'best_model.pth')
        print(f'Validation loss: {avg_loss_val:.4f}')

        ## Tune thresholds on validation set to get best F1 per tag
        f1s=torch.zeros(num_labels)
        for i,label in enumerate(config.data_labels):
            best_f1=0.0
            best_thresh=0.5
            for thresh in np.linspace(0.1,0.9,9):
                preds=(probs[:,i]>thresh).float()
                f1, _, _, _ = calculate_metrics(preds, gts[:,i])
                if f1>best_f1:
                    best_f1=f1
                    best_thresh=thresh
            f1s[i]=best_f1
            print(f'Tag {label}: best F1 {best_f1:.4f} at threshold {best_thresh:.2f}')
            model.thresholds[i]=best_thresh
        weights=gts.sum(dim=0).float()/gts.sum()
        print(f'Overall weighted F1 in validation: {(weights*f1s).sum():.4f}')
    return best_epoch, best_val_loss, train_losses, val_losses

@torch.no_grad()
def test(model, config, test_dataset):
    test_dl=DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False)
    loss_function = nn.BCEWithLogitsLoss().to(model.device)
    out, labels = run_inference(model, test_dl)
    f1, prec, rec, accuracy = calculate_metrics(out, labels)
    weights=labels.sum(dim=0).float()/labels.sum()
    print(f'Overall accuracy: {(weights*accuracy).sum():.4f}, overall F1: {(f1*weights).sum():.4f}')
    print('Per-tag results:')
    for i,label in enumerate(config.data_labels):
        print(f'Tag {label}: average test F1 {f1[i]:.4f} (average precision {prec[i]:.4f}, average recall {rec[i]:.4f}) average accuracy: {accuracy[i]:.4f}')
    return f1, prec, rec, accuracy, weights

def main(args=None):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    config=ExperimentConfig()
    if args is not None:
        config.use_source_code=not args['no_source_code']
        config.use_difficulty=args['use_difficulty']
        if 'test_only' in args.keys():
            config.test_only=args['test_only']
        if 'test_dataset' in args.keys():
            config.test_dataset=args['test_dataset']
        if 'checkpoint_path' in args.keys():
            config.checkpoint_path=args['checkpoint_path']
        
    full_dataset=CodeRecordDataset(config)
    if config.test_dataset is not None:
        test_config=copy.deepcopy(config)
        test_config.data_path=config.test_dataset
        test_dataset=CodeRecordDataset(test_config)
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.85, 0.15])
    else:
        if config.test_only:
            print('Warning: running test only without specifying test dataset, using new split of original dataset. Potential overlap between train and test')
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.7, 0.15, 0.15])
    
    if not config.test_only:
        print('Training model...')
        best_epoch, best_val_loss, train_losses, val_losses = train(config, train_dataset, val_dataset)
        model=CodeClassifierModel(config, device)
        model.load_state_dict(torch.load('best_model.pth'))
        print(f'Loading best model from epoch {best_epoch+1} with validation loss {best_val_loss:.4f}')
    else:
        print('Loading model from checkpoint for testing...')
        model=CodeClassifierModel(config, device)
        model.load_state_dict(torch.load(config.checkpoint_path))
        print(f'Loaded model from {config.checkpoint_path}')

    ##testing
    test(model, config, test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_source_code', action='store_true', help='Whether to not use source code as input')
    parser.add_argument('--use_difficulty', action='store_true', help='Whether to use difficulty as input')
    parser.add_argument('--test_only', action='store_true', help='Whether to only run testing, works better if test dataset is specified and requires a checkpoint (either default from previous training or a custom one)')
    parser.add_argument('--test_dataset', type=str, required=False, default=None, help='test dataset path, recommended if --test_only is set')
    parser.add_argument('--checkpoint_path', type=str, required=False, default='best_model.pth', help='Path to model checkpoint, used if --test_only is set. Defaults to checkpoint from latest training')
    args=parser.parse_args()
    args=vars(args)
    main(args)