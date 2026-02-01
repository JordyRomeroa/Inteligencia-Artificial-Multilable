import os
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'
SPLIT_DIRS = {
    'train': DATA_DIR / 'images' / 'train',
    'val': DATA_DIR / 'images' / 'val',
    'test': DATA_DIR / 'images' / 'test',
    'labels_train': DATA_DIR / 'labels' / 'train',
    'labels_val': DATA_DIR / 'labels' / 'val',
    'labels_test': DATA_DIR / 'labels' / 'test',
}

TARGET_CLASSES = ['person', 'car', 'dog']
CLASS_IDS = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}

for dir_path in SPLIT_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

num_train = 400
num_val = 50
num_test = 50

def generate_synthetic_images(split, count):
    split_dir = SPLIT_DIRS[split]
    labels_dir = SPLIT_DIRS[f'labels_{split}']
    
    for i in range(count):
        img_name = f'{split}_{i:04d}.jpg'
        img_path = split_dir / img_name
        
        img_data = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
        
        from PIL import Image
        Image.fromarray(img_data).save(str(img_path))
        
        num_objects = np.random.randint(1, 4)
        label_data = []
        
        for _ in range(num_objects):
            class_id = np.random.randint(0, 3)
            x_center = np.random.uniform(0.1, 0.9)
            y_center = np.random.uniform(0.1, 0.9)
            width = np.random.uniform(0.1, 0.5)
            height = np.random.uniform(0.1, 0.5)
            
            label_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        label_path = labels_dir / img_name.replace('.jpg', '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_data))

generate_synthetic_images('train', num_train)
generate_synthetic_images('val', num_val)
generate_synthetic_images('test', num_test)

print(f"Dataset prepared: {num_train} train, {num_val} val, {num_test} test images")
print(f"Classes: {TARGET_CLASSES}")
print(f"YOLO format: class_id x_center y_center width height (normalized)")
