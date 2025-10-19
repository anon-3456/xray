import torch
import json
import os
from datetime import datetime

class NotebookSession:
    def __init__(self, project="./", unique_id=None):
        if unique_id is not None:
            self.unique_id = unique_id
        else:
            self.unique_id = "run_" + str(int(datetime.now().timestamp()))[-6:]
        
        self.project_dir = project + self.unique_id
        
        if os.path.isdir(self.project_dir):
            raise Exception("existing project")
        
        os.makedirs(self.project_dir + "/logs")
        os.makedirs(self.project_dir + "/objects")

    def _prepare_path(self, path):
        dir_only = path[:path.rindex("/")]
        os.makedirs(dir_only, exist_ok=True)
        return path
    
    def print(self, s: str):
        print(s)

        with open(self.project_dir + "/logs/main.txt", "a") as file:
            file.write(f"{s}\n")

    def save(self, o: any, name: str):
        path = self._prepare_path(f"{self.project_dir}/objects/{name}")
        torch.save(o, path)
        self.print(f"[✓] Saved object to {path}")

    def save_json(self, o: any, name: str):
        path = self._prepare_path(f"{self.project_dir}/objects/{name}")
        with open(path, "w") as file:
            json.dump(o, file, indent=4)
            self.print(f"[✓] Saved object to {path}")
        
        
