import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from src.core.registry import DATASETS

@DATASETS.register_module()
class CelebADataset(Dataset):
    """
    CelebA Dataset Loader.
    
    Data Structure:
    data_root/
        img_align_celeba/
            000001.jpg ...
        list_attr_celeba.csv
        list_eval_partition.csv
        list_bbox_celeba.csv
        list_landmarks_align_celeba.csv
        
    Args:
        data_root (str): Root directory of dataset.
        pipeline (callable, optional): Data pipeline (transforms).
        split (str): 'train', 'val', or 'test'.
        load_attributes (bool): Whether to load attribute labels.
        load_landmarks (bool): Whether to load landmarks.
        load_bboxes (bool): Whether to load bounding boxes.
    """
    def __init__(self, 
                 data_root, 
                 pipeline=None, 
                 split='train', 
                 load_attributes=True,
                 load_landmarks=True,
                 load_bboxes=True,
                 **kwargs):
        self.data_root = data_root
        self.pipeline = pipeline
        self.split = split
        self.load_attributes = load_attributes
        self.load_landmarks = load_landmarks
        self.load_bboxes = load_bboxes
        
        self.image_dir = os.path.join(data_root, 'img_align_celeba')
        
        # Load Partitions
        self.partition_df = pd.read_csv(
            os.path.join(data_root, 'list_eval_partition.csv'), 
            sep=',', 
            header=1, 
            names=['image_id', 'partition']
        )
        
        # Map split name to partition ID
        # 0: train, 1: val, 2: test
        split_map = {'train': 0, 'val': 1, 'test': 2}
        target_partition = split_map.get(split, 0)
        
        # Filter by split
        self.data_df = self.partition_df[self.partition_df['partition'] == target_partition].copy()
        
        # Load Attributes if requested
        if self.load_attributes:
            attr_path = os.path.join(data_root, 'list_attr_celeba.csv')
            if os.path.exists(attr_path):
                self.attr_df = pd.read_csv(attr_path)
                # Merge on image_id
                self.data_df = self.data_df.merge(self.attr_df, on='image_id', how='left')
        
        # Load Bounding Boxes if requested
        if self.load_bboxes:
            bbox_path = os.path.join(data_root, 'list_bbox_celeba.csv')
            if os.path.exists(bbox_path):
                self.bbox_df = pd.read_csv(bbox_path)
                self.data_df = self.data_df.merge(self.bbox_df, on='image_id', how='left')
                
        # Load Landmarks if requested
        if self.load_landmarks:
            lm_path = os.path.join(data_root, 'list_landmarks_align_celeba.csv')
            if os.path.exists(lm_path):
                self.lm_df = pd.read_csv(lm_path)
                self.data_df = self.data_df.merge(self.lm_df, on='image_id', how='left')

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        image_id = row['image_id']
        img_path = os.path.join(self.image_dir, image_id)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy or skip (simple retry strategy not implemented here)
            return torch.zeros(3, 256, 256)

        # Construct data dict for pipeline
        results = {'img': img, 'img_path': img_path, 'image_id': image_id}
        
        if self.load_attributes:
            # Assume attributes start after 'partition' or known columns
            # For simplicity, passing all row data as attributes dict
            results['attributes'] = row.to_dict()
            
        if self.load_bboxes and 'x_1' in row:
            results['bbox'] = [row['x_1'], row['y_1'], row['width'], row['height']]
            
        if self.load_landmarks and 'lefteye_x' in row:
            results['landmarks'] = [
                (row['lefteye_x'], row['lefteye_y']),
                (row['righteye_x'], row['righteye_y']),
                (row['nose_x'], row['nose_y']),
                (row['leftmouth_x'], row['leftmouth_y']),
                (row['rightmouth_x'], row['rightmouth_y'])
            ]

        if self.pipeline:
            # Pipeline expects image or dict? 
            # Standard torch transforms expect img. 
            # MMEngine pipelines expect dict.
            # Our current pipeline (T012) uses standard torchvision transforms wrapper.
            # So we pass just the image for now, unless we upgrade the pipeline to handle dicts.
            # Upgrading pipeline to handle dicts is better for using bbox/landmarks.
            
            # Current T012 pipeline calls: transform(img)
            # We will pass just img to keep compatibility with T012 implementation.
            img = self.pipeline(img)
            
        return img