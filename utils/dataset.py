import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.processing import warp_image_and_keypoints, detect_largest_face, calculate_affine_matrix_from_bbox


class FaceKeypointDataset(Dataset):
    def __init__(self, data_dir, output_size=(256, 256), transform=None):
        self.data_dir = data_dir
        self.output_size = output_size  # (w, h)
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

        return {
            "image": image_tensor,
            "keypoints": keypoints_tensor,
            "img_path": img_path
        }


class UDAFaceKeypointDataset(Dataset):
    def __init__(self, data_dir, output_size=(256, 256), margin=0.2, transform=None):
        self.data_dir = data_dir
        self.output_size = output_size
        self.margin = margin

        self.image_files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
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
        img_path = os.path.join(self.data_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect largest face
        bbox = detect_largest_face(image)
        if bbox is None:
            raise ValueError(f"No face found in {img_name}")

        # Get base affine matrix
        M = calculate_affine_matrix_from_bbox(bbox, self.margin, self.output_size)

        # Apply affine warp to get base image
        image_base = cv2.warpAffine(image, M, (self.output_size[1], self.output_size[0]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Apply strong augmentation on base image (e.g., random rotation)
        angle = np.random.uniform(-30, 30)
        scale_aug = np.random.uniform(0.8, 1.2)
        center = (self.output_size[1] / 2, self.output_size[0] / 2)

        aug_matrix = cv2.getRotationMatrix2D(center, angle, scale_aug)
        image_aug = cv2.warpAffine(image_base, aug_matrix,
                                   (self.output_size[1], self.output_size[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Convert to tensor
        image_base_tensor = self.transform(image_base)
        image_aug_tensor = self.transform(image_aug)

        return {
            "image_base": image_base_tensor,
            "image_aug": image_aug_tensor,
            "M_base": torch.tensor(M, dtype=torch.float32),
            "M_aug": torch.tensor(aug_matrix, dtype=torch.float32),
            "img_path": img_path
        }
