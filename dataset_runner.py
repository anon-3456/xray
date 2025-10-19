import os
import matplotlib.pyplot as plt

import random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from torchvision import transforms
from timm.models import create_model
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data import DataLoader, Dataset, Subset

def grab_results(objects_path=None, dataset=None, split="test", create_model=None):

    if not objects_path or not dataset or not create_model:
        print("Missing some crucial arguments man")
        return []
    
    device = torch.device("cuda:1")
    
    files = filter(lambda s: s.endswith(f"{split}_indices.arr"), os.listdir(objects_path))
    
    # N FOLDs: [(true, actual), (true, actual), ...]
    results = []
    
    for f in files:
        parts = f.split("_")
        name = parts[0] + "_" + parts[1]
        fold_n = int(parts[1])
    
        print(f"Collecting results for fold-{fold_n}")
        
        try:
            indices = torch.load(f"{objects_path}/{name}_{split}_indices.arr", weights_only=False)
            subdataset = Subset(dataset, indices)
            state_dict = torch.load(f"{objects_path}/fold-{fold_n}-state.pth")
        except FileNotFoundError:
            break
    
        model = create_model().to(device)
        model.load_state_dict(state_dict)
        model.eval()
    
        with torch.no_grad():
            loader = DataLoader(subdataset, batch_size = 1, shuffle=False)
        
            y_true = []
            y_scores = []
            
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.unsqueeze(-1).to(device)
        
                output = model(inputs)
        
                y_scores.extend(output.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
            y_true = np.array(y_true)
            y_scores = np.array(y_scores)
    
        results.append((y_true, y_scores))

    return results
