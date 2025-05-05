import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.processing import warp_image_and_keypoints, detect_largest_face


class FaceKeypointDataset(Dataset):
    def __init__(self, data_dir, output_size=(256, 256), transform=None):
        self.data_dir = data_dir
        self.output_size = output_size  # (h, w)
        self.image_files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')
        ])        

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        json_name = img_name.rsplit('.', 1)[0] + '.json'
        img_path = os.path.join(self.data_dir, img_name)
        json_path = os.path.join(self.data_dir, json_name)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load keypoints
        with open(json_path, 'r') as f:
            annot = json.load(f)
        keypoints = np.array([shape['points'][0] for shape in annot['shapes']], dtype=np.float32)

        # Detect face
        bbox = detect_largest_face(image)
        if bbox is None:
            raise ValueError(f"No face found in {img_name}")

        aligned_img, transformed_kps = warp_image_and_keypoints(
            image,
            keypoints,
            bbox,
            output_size=self.output_size,
            margin=0.2
        )

        # Normalize image
        image_tensor = self.transform(aligned_img)
        keypoints_tensor = torch.tensor(transformed_kps, dtype=torch.float32)

        return image_tensor, keypoints_tensor, img_path
