import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import json
import argparse

def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model results file.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the analysis results.")
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path).replace('_results.pth', '')
    
    all_y_true = []
    all_y_scores = []

    try:
        data = torch.load(args.model_path, weights_only=False)
        for fold_data in data:
            all_y_true.append(fold_data[0].flatten())
            all_y_scores.append(fold_data[1].flatten())
        
        y_true_combined = np.concatenate(all_y_true)
        y_scores_combined = np.concatenate(all_y_scores)

        y_true_combined = (y_true_combined > 0.5).astype(int)

        optimal_threshold = find_optimal_threshold(y_true_combined, y_scores_combined)
        
        y_pred_combined = (y_scores_combined >= optimal_threshold).astype(int)
        
        cm = confusion_matrix(y_true_combined, y_pred_combined)
        
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            "model_name": model_name,
            "optimal_threshold": float(optimal_threshold),
            "confusion_matrix": cm.tolist(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1_score)
        }

        with open(args.save_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Results for {model_name} saved to {args.save_path}")

    except Exception as e:
        print(f"Error processing {args.model_path}: {e}")
