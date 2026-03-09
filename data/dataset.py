import torch
from torch.utils.data import Dataset
import numpy as np
from utils.geometry import svd_orthogonalize

class DummyYOLO6DDataset(Dataset):
    """
    產生虛假資料 (Dummy Data) 用來測試 train.py 訓練迴圈是否暢通。
    包含假的圖片 Tensor 以及對應的 4 種 Ground Truth 標註。
    """
    def __init__(self, num_samples=32, img_size=640):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Image (3 channels)
        img = torch.randn(3, self.img_size, self.img_size)
        
        # 2. 為三個尺度的 Head 提供目標 (P3=80x80, P4=40x40, P5=20x20)
        # 實際上 YOLO 標註會把物件依照大小分配到不同的特徵圖尺度上
        # 這裡簡單隨機生成一個 Dummy Target Dictionary 來模擬
        
        # 假設該圖片有一個目標，我們將標標準值塞給對應的維度
        # Box: cx, cy, w, h
        box = torch.rand(4)
        
        # Keypoints: 9 points (x, y)
        kpt = torch.rand(9, 2)
        vis_mask = torch.ones(9) # 全可見
        
        # Rotation: 9D (之後轉 3x3)
        # 將隨機 9D 轉成合法的 3x3 旋轉矩陣
        rot_9d = torch.randn(1, 9)
        rot_3x3 = svd_orthogonalize(rot_9d).squeeze(0)
        
        # Depth: scale between 0, 1
        depth = torch.rand(1)
        
        target = {
            'box': box,
            'kpt': kpt,
            'vis_mask': vis_mask,
            'rot': rot_3x3,
            'depth': depth
        }
        
        return img, target
