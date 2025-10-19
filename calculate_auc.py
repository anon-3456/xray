import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
import json

def calculate_auc_for_model(file_path):
    data = torch.load(file_path, weights_only=False)
    
    all_fold_aucs = []
    for fold_data in data:
        y_true = fold_data[0].flatten()
        y_scores = fold_data[1].flatten()
        
        # Ensure y_true is binary (0 or 1)
        y_true = (y_true > 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        all_fold_aucs.append(roc_auc)
    
    return all_fold_aucs

if __name__ == "__main__":
    eval_results_dir = "eval_results_zeros"
    results = {}

    for filename in os.listdir(eval_results_dir):
        if filename.endswith(".pth"):
            model_name = filename.replace("_results.pth", "")
            file_path = os.path.join(eval_results_dir, filename)
            
            try:
                fold_aucs = calculate_auc_for_model(file_path)
                results[model_name] = fold_aucs
                print(f"Calculated AUCs for {model_name}: {fold_aucs}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    output_json_path = "auc_results_zeros.json"
    with open(output_json_path, "w") as f:
        json.dump(dict(sorted(results.items())), f, indent=4)
    
    print(f"All AUC results saved to {output_json_path}")
