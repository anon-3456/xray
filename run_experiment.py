from load_dataset import data_summary, make_dataset, set_seeds
from nbsession import NotebookSession

import numpy as np
import os

from timm.models import create_model, list_models

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_curve, auc

from lion_pytorch import Lion
from torch.optim.lr_scheduler import (ReduceLROnPlateau)

BASE_DIR = "./data"
IMAGE_DIR = "./data/images"

def calc_accuracy(logits, labels, thresh=0.5):
    return torch.sum((torch.sigmoid(logits) > thresh) == labels).item()


def run_experiment(
    sess: NotebookSession, 
    device: str, 
    short_name: str, 
    model_name: str, 
    resolution: int, 
    batch_size: int, 
    epochs: int, 
    weight_decay: int=0, 
    grid_idx: int=0, 
    seed: int=9999
):
    dataset_settings = {
        "image_size": (resolution, resolution),
        "label_map": {
            'quality 1 (perfect)': 0,
            'quality 2': None,
            'quality 3': None,
            'quality 4': 1,
            'quality 5 (unreadable)': 1,
        }
    }

    model_details = {
        "model_name": model_name,
        "short_name": short_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "seed": seed
    }

    sess.print("Making Dataset")
    set_seeds(model_details["seed"])
    dataset = make_dataset(BASE_DIR, IMAGE_DIR, 
                       image_size=dataset_settings["image_size"],
                       label_map=dataset_settings["label_map"])

    sess.print("READY!")
    
    x_idx, y_values = dataset.idx_representation()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=model_details["seed"])
    
    # Define the PyTorch Model
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    sess.print(f"Using device: {device}")
    sess.print(f"Model details: {model_details}")
    
    sess.save_json(model_details, f"{grid_idx}-model_details.json")
    sess.save_json(dataset_settings, f"{grid_idx}-dataset_settings.json")

    fold_evals = []
    fold_roc_auc = []
    
    for fold_i, (train_indices, test_indices) in enumerate(kfold.split(x_idx, y_values)):
        sess.print(f"===== [ Fold #{fold_i} ] =====")
    
        sess.save(train_indices, f"indices/{grid_idx}-fold_{fold_i}_train_indices.arr")
        sess.save(test_indices, f"indices/{grid_idx}_fold_{fold_i}_test_indices.arr")
        
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
    
        # DataLoader
        # Drop last so that batch norm won't break
        train_loader = DataLoader(train_dataset, 
                                  batch_size = model_details["batch_size"], 
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, 
                                 batch_size = model_details["batch_size"], 
                                 shuffle=False)
    
        sess.print(f"Started training session:")
    
        model = create_model(
            model_details["model_name"], 
            pretrained=True, num_classes=1
        ).to(device)
    
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=model_details["weight_decay"], lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.3, min_lr=1e-8)
    
        sess.print("Model loaded. Starting training:")
    
        # [1] TRAIN STEP
        for epoch in range(model_details["epochs"]):
            model.train()
            
            running_loss = 0.0
            train_acc = 0.0
    
            set_seeds(model_details["seed"])
            for images, labels in train_loader:
                images, labels = images.type(torch.float).to(device), labels.type(torch.float).to(device)
    
                logits = model(images).reshape(-1)
                loss = criterion(logits, labels)
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_acc += calc_accuracy(logits, labels)
                running_loss += loss.item()
        
            # [2] POST-EPOCH VALIDATION STEP
            model.eval()
        
            val_loss = 0.0
            acc = 0.0
        
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.type(torch.float).to(device), labels.type(torch.float).to(device)
    
                    logits = model(images).reshape(-1)
                    loss = criterion(logits, labels)
    
                    acc += calc_accuracy(logits, labels)
                    val_loss += loss.item()
        
            # Average losses
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            avg_accuracy = acc / len(test_dataset)
            avg_train_accuracy = train_acc / len(train_dataset)
        
            sess.print(f"Epoch {epoch + 1}/{model_details["epochs"]}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}, "
                    f"Train Acc: {avg_train_accuracy:.4f}, "
                    f"Val Acc: {avg_accuracy:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.4}"
                 )
                    
            scheduler.step(running_loss)
            # [X] END POST-EPOCH VALIDATION STEP
            
        # [X] END TRAIN LOOP
        
        # [3] POST-FOLD EVAL, SAVING LOGITS, CALC AUC
        model.eval()
    
        with torch.no_grad():
            y_true = []
            y_scores = []
            
            for inputs, labels in test_loader:
                inputs = inputs.type(torch.float).to(device)
                labels = labels.type(torch.float).to(device)
        
                output = model(inputs).reshape(-1)
        
                y_scores.extend(output.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
            y_true = np.array(y_true).flatten()
            y_scores = np.array(y_scores).flatten()
    
        fold_evals.append((y_true, y_scores))

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fold_roc_auc.append({
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc
        })
        
        # Saving model checkpoints
        sess.save(model.state_dict(), f"checkpoints/{grid_idx}-fold-{fold_i}-state.pth")
        
    # [X] END FOLD LOOP

    sess.save(fold_evals, f"logits/{grid_idx}-fold-logits.pth")
    sess.save(fold_roc_auc, f"curve_and_auc/{grid_idx}-fold_roc_auc.pth")
    sess.print("DONE!")