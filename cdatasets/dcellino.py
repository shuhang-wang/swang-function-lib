import os
import glob
import numpy as np
import torch
import cv2
from torchvision import transforms
from os.path import join as pj
import random

def find_npy_files(root_folder):
    npy_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files

class Resize(object):
    """Resize a numpy image to a given size using OpenCV."""
    def __init__(self, target_size):
        self.target_size = target_size  # (width, height)

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be resized.
        Returns:
            numpy.ndarray: Resized image.
        """
        img = cv2.resize(img.astype('float32'), self.target_size, interpolation=cv2.INTER_LINEAR)
        return img

class RandomCrop(object):
    def __init__(self, scale_range=[0.7, 1.0]):
        self.scale_range = scale_range

    def __call__(self, img):
        height, width = img.shape[:2]
        scale = random.uniform(*self.scale_range)
        aspect_ratio = random.uniform(3. / 4, 4. / 3)
        
        crop_height = int(height * scale / aspect_ratio)
        crop_width = int(width * scale * aspect_ratio)
        
        if crop_height > height:
            crop_height = height
        if crop_width > width:
            crop_width = width
        
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        
        cropped_image = img[top:top + crop_height, left:left + crop_width]
        
        return cropped_image

class RandomFlip(object):
    '''
        - horizontal flip
        - vertical flip
    '''
    def __call__(self, img):
        # - horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        # - vertical flip
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
        return img

class RandomRotate(object):
    def __init__(self, max_angle=360):
        self.max_angle = max_angle
    def __call__(self, img):
        angle = random.uniform(-self.max_angle, self.max_angle)
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return rotated_image
    
class GaussianBlur(object):
    def __init__(self, kernel_size) -> None:
        self.kernel_size = kernel_size
    def __call__(self, img):
        blurred_image = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
        return blurred_image
    
class RandomGrayscale(object):
    def __call__(self, img, p=0.2):
        if random.random() < p:
            grayscale_image = np.mean(img, axis=2)
            grayscale_image = np.stack([grayscale_image] * img.shape[2], axis=-1)  # Replicate across channels
            return grayscale_image
        return img

class RandomJitter(object):
    def __call__(self, img, brightness=0.2, contrast=0.2):
        
        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            img = img * factor
        
        if contrast > 0:
            factor = 1.0 + random.uniform(-contrast, contrast)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            img = (img - mean) * factor + mean
        
        return img
    
class BrtTileDataset(torch.utils.data.Dataset):
    """
    Dataset for tile dataset.
    For unsupervised learning
    """

    def __init__(
        self,
        image_dir,
        img_channel_num=4,
        transform = None
    ):
        """
        Args:
            source: [str]. Path to the data folder.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            image_size: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        
        self.image_dir = image_dir
        self.img_channel_num = img_channel_num

        img_paths = find_npy_files(image_dir)
        self.img_paths = sorted(img_paths)
        self.transform = transform

    def __getitem__(self, idx):
        
        image_path = self.img_paths[idx]
        image = np.load(image_path)[:,:,:self.img_channel_num]
        image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.img_paths)

def get_simclr_pipeline_transform_cellino(image_size=256):
        data_transforms =  transforms.Compose([
                            RandomCrop(),
                            Resize((image_size, image_size)), 
                            RandomFlip(),
                            RandomRotate(),
                            GaussianBlur(kernel_size = int(0.1 * image_size)),
                            RandomGrayscale(),
                            RandomJitter(),
                            transforms.ToTensor(),])
        return data_transforms

def get_dataloader(image_dir, cfg):
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    image_size = cfg['image_size']
    img_channel_num = cfg['img_channel_num']

    data_transforms =  transforms.Compose([
        RandomCrop(),
        Resize((image_size, image_size)), 
        RandomFlip(),
        RandomRotate(),
        GaussianBlur(kernel_size = int(0.1 * image_size)),
        RandomGrayscale(),
        RandomJitter(),
        transforms.ToTensor(),])

    dataset = BrtTileDataset(
        image_dir=image_dir,
        img_channel_num = img_channel_num,
        data_transforms = data_transforms,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
