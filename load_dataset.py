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
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


def set_seeds(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import json
from collections import Counter


def data_summary(dataset, test_size=0.2, batch_size=2, seed=3000):
    set_seeds(seed)

    x_indices, y_labels = dataset.idx_representation()
    
    train_indices, test_indices, _, _ = train_test_split(
        x_indices, y_labels, stratify=y_labels, test_size=test_size, random_state=seed, shuffle=True
    )
    
    # Create datasets and dataloaders
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    # DATA SUMMARY
    print("X dtype:", next(iter(train_loader))[0].dtype)
    print("Y dtype:", next(iter(train_loader))[1].dtype)
    
    print("Test size:", len(test_loader.dataset))
    print("Train size:", len(train_loader.dataset))
    
    print("Single batch shape:", next(iter(train_loader))[0].shape)
    
    print()
    print("Complete class distribution:")
    print(json.dumps(dataset.label_distribution, indent=2))
    
    print("\nTrainset class distribution:")

    try:
        c = Counter(map(lambda d: d[1].item(), train_dataset))
    except RuntimeError:
        c = Counter(map(lambda d: str(tuple(d[1].tolist())), train_dataset))
    distribution = {k: v / len(train_dataset) for k, v in c.items()}
    print(json.dumps(dict(c.items()), indent=2))
    print(json.dumps(distribution, indent=2))


def make_dataset(base_dir, image_dir, image_size=(600,600), seed=88, no_middle=False, only_middle=False, label_map=None):
    set_seeds(seed)
    
    # Load the CVAT XML file
    tree = ET.parse(os.path.join(base_dir,'annotations.xml'))
    root = tree.getroot()
    qualityTypes = ['quality 1 (perfect)','quality 2','quality 3','quality 4','quality 5 (unreadable)']
    qualities = []

    # PyTorch Dataset Definition (hardcoded for all)
    class CustomDataset(Dataset):
        def __init__(self, xml_root, image_dir, transform=None):
            """
            Args:
                xml_root: Parsed XML root element.
                image_dir: Directory containing images.
                transform: Transformations for images and masks.
            """
            self.xml_root = xml_root
            self.image_dir = image_dir
            self.transform = transform
            
            # all types: ['quality 1 (perfect)','quality 2','quality 3','quality 4','quality 5 (unreadable)']
            if label_map:
                self.label_map = label_map
            else:
                self.label_map = {
                    'quality 1 (perfect)': 0.,
                    'quality 2': 0.,
                    'quality 3': 1.,
                    'quality 4': 1.,
                    'quality 5 (unreadable)': 1.
                }
            self.label_distribution = { k:0 for k in self.label_map }
            self.images = []
            self.labels = []
            self._generate_dataset()
    
        def __len__(self):
            return len(self.images)
    
        def _generate_dataset(self):
            self.label_distribution = { k:0 for k in self.label_map }
            all_images = self.xml_root.findall(".//image")
    
            for idx, image_elem in enumerate(all_images):
                # Check if image can be read and converted
                img_name = image_elem.attrib['name']
                assert Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
                
                # Check if label is valid
                quality_label = None
                quality = None
                scan_type = None
                for annotationView in image_elem.findall("tag"):
                    label = annotationView.attrib['label']
                    if label in self.label_map:
                        quality_label = label
                        quality = None if self.label_map[label] is None else torch.tensor(self.label_map[label], dtype=torch.float)
                    elif "quality" in label:
                        print("Unknown quality label found:", img_name, label)
                    else:
                        scan_type = label
                
                if quality is not None:                    
                    if only_middle and not any(
                        [quality_label == 'quality 2',
                         quality_label == 'quality 3']
                    ):
                        continue
                    
                    # 1v45 FILTER
                    if no_middle and not any(
                        [quality_label == 'quality 1 (perfect)',
                         quality_label == 'quality 4',
                         quality_label == 'quality 5 (unreadable)']):
                        continue
                    
                    self.label_distribution[quality_label] += 1
                    self.labels.append(quality)
                    self.images.append(img_name)
    
        def idx_representation(self, reduce=False):
            return range(len(self.images)), [torch.sum(x).type(torch.float32).item() for x in self.labels]
    
        def __getitem__(self, idx):
            image_path = os.path.join(self.image_dir, self.images[idx])
            image = Image.open(image_path).convert("RGB")
            quality = self.labels[idx]
    
            if self.transform:
                image = self.transform(image)
            
            return image, quality
    
    class GrayAndEdgeTransform:
        def __call__(self, img: Image.Image) -> Image.Image:
            np_img = np.array(img)
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            edges_1 = cv2.Canny(gray, 17, 29)
            edges_2 = cv2.Canny(gray, 60, 60)
            combined = np.stack([gray, edges_1, edges_2], axis=-1)
            # Convert NumPy array back to PIL Image
            return combined

    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize to uniform size
        # GrayAndEdgeTransform(),         # Insert our new transform here
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(root, image_dir, transform=transform)

    return dataset