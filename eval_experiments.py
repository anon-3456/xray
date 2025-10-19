import json
import os
import torch
import timm
import numpy as np
from torch import nn
from load_dataset import make_dataset
from dataset_runner import grab_results

def get_model_details(experiment_name, objects_path=None):
    if objects_path is not None:
        try:
            try:
                details_path = os.path.join(objects_path, "model_details.pth")
                details = torch.load(details_path, weights_only=False)
            except FileNotFoundError:
                with open(os.path.join(objects_path, "model_details.json"), "r") as file:
                    details = json.loads(file.read())
                    details["name"] = details["model_name"]
            return details
        except FileNotFoundError:
            pass

    # Fallback for legacy experiments, where the name of the model is not saved
    
    if "vit_large_patch32_384" in experiment_name:
        return {"name": "vit_large_patch32_384", "dropout": 0.1}
    if "convnextv2_nano" in experiment_name:
        return {"name": "convnextv2_nano", "dropout": 0.1}
    if "fastvit_ma36" in experiment_name:
        return {"name": "fastvit_ma36", "dropout": 0.1}
    if "fastvit_t8" in experiment_name:
        return {"name": "fastvit_t8", "dropout": 0.1}
    if "mobilenetv4_conv_aa_large" in experiment_name:
        return {"name": "mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k", "dropout": 0.1}
    if "mobilenetv4_conv_medium" in experiment_name:
        return {"name": "mobilenetv4_conv_medium.e500_r256_in1k", "dropout": 0.1}
    if "resnetv2_18" in experiment_name:
        return {"name": "resnetv2_18", "dropout": 0.1}
    if "tf_efficientnetv2_s" in experiment_name:
        return {"name": "tf_efficientnetv2_s.in21k", "dropout": 0.1}

            
    raise ValueError(f"Unknown model for experiment: {experiment_name}")

def main():
    experiments_dir = "reruns"
    output_dir = "eval_results_zeros"
    os.makedirs(output_dir, exist_ok=True)

    for experiment_name in os.listdir(experiments_dir):
        experiment_path = os.path.join(experiments_dir, experiment_name)
        objects_path = os.path.join(experiment_path, "objects")
        output_filename = os.path.join(output_dir, f"{experiment_name}_results.pth")

        if os.path.exists(output_filename):
            print(f"Skipping {output_filename} because it already exists")
            continue

        if not os.path.isdir(objects_path):
            print("Objects not a directory")
            continue

        print(f"Processing experiment: {experiment_name}")

        try:
            try:
                dataset_settings = torch.load(os.path.join(objects_path, "dataset_settings.pth"))
            except FileNotFoundError:
                with open(os.path.join(objects_path, "dataset_settings.json"), "r") as file:
                    dataset_settings = json.loads(file.read())
            dataset = make_dataset(
                base_dir="./data",
                image_dir="./data/images",
                image_size=dataset_settings['image_size'],
                label_map=dataset_settings['label_map']
            )
        except Exception as e:
            print(f"Error loading dataset for {experiment_name}: {e}")
            continue

        model_details = get_model_details(experiment_name, objects_path)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        def create_model_fn():
            model = timm.create_model(
                model_details["name"], 
                pretrained=True, 
                num_classes=1
            )
            return model

        results = grab_results(
            objects_path=objects_path, 
            dataset=dataset,
            create_model=create_model_fn,
            split="test"
        )

        torch.save(results, output_filename)
        print(f"Saved results to {output_filename}")

if __name__ == "__main__":
    main()
