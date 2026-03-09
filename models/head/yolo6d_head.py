import torch
import torch.nn as nn
from models.backbone.e_elan import Conv

class BoxHead(nn.Module):
    def __init__(self, in_channels, num_classes=80):
        super().__init__()
        # Standard YOLO Detection Head
        # Outputs: 4 (bbox) + num_classes
        self.nc = num_classes
        self.no = 4 + self.nc  # 4 for x, y, w, h
        
        # Typically uses two convolutions
        self.cv = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            nn.Conv2d(in_channels, self.no, 1, 1) # No activation, raw logits for DFL/CIoU loss
        )

    def forward(self, x):
        return self.cv(x)

class KeypointHead(nn.Module):
    def __init__(self, in_channels, num_points=9):
        super().__init__()
        # 3D BBox projected keypoints (8 corners + 1 center point)
        # Outputs: num_points * 3 (x, y, visibility)
        self.nk = num_points
        self.no = self.nk * 3
        
        self.cv = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            nn.Conv2d(in_channels, self.no, 1, 1) # Raw outputs
        )

    def forward(self, x):
        return self.cv(x)

class RotationHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 3D Rotation Pose (Continuous 9D representation mapped to SO(3) via SVD)
        # Outputs: 9 (reshaped later to 3x3 matrix)
        self.no = 9
        
        self.cv = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            nn.Conv2d(in_channels, self.no, 1, 1) # Raw continuous 9D outputs, no activation
        )

    def forward(self, x):
        # We process to SO(3) via SVD in a post-processing or loss calculation step,
        # but the network itself just outputs these 9 unbounded values.
        return self.cv(x)

class DepthHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Depth/Translation Prediction
        # Outputs: 1 (normalized scale sigma in range (0, 1))
        self.no = 1
        
        self.cv = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            nn.Conv2d(in_channels, self.no, 1, 1),
            nn.Sigmoid() # Force output to be (0, 1) range for depth scaling
        )

    def forward(self, x):
        return self.cv(x)

class YOLO6DHead(nn.Module):
    def __init__(self, in_channels, num_classes=80):
        """
        Unified 6D Pose Prediction Head.
        Takes feature map from FPN/PAN (e.g., P3, P4, P5) and splits into the 4 sub-heads.
        For YOLO, we usually have a Multi-Level head (ModuleList over the P3, P4, P5 inputs),
        but here we apply the heads individually per feature scale.
        """
        super().__init__()
        # For simplicity in this demo, we instantiate the 4 heads for a *single* scale feature map.
        # In full YOLO we'd replicate this for P3, P4, P5 using ModuleList for multi-scale detection.
        self.box_head = BoxHead(in_channels, num_classes)
        self.kpt_head = KeypointHead(in_channels, num_points=9)
        self.rot_head = RotationHead(in_channels)
        self.depth_head = DepthHead(in_channels)
        
    def forward(self, x):
        """
        Input: Feature map from Neck (FPN/PAN)
        Outputs: Tuple of (box, keypoints, rotation, depth)
        """
        box_out = self.box_head(x)     # [B, 4+C, H, W]
        kpt_out = self.kpt_head(x)     # [B, 27, H, W]
        rot_out = self.rot_head(x)     # [B, 9, H, W]
        depth_out = self.depth_head(x) # [B, 1, H, W]
        
        return box_out, kpt_out, rot_out, depth_out

# Test the Heads
if __name__ == "__main__":
    print("Testing YOLO-6D Head Module:")
    
    # 假設 FPN/PAN 輸出的一層 (e.g., P4 with 64 channels, 40x40 spatial)
    B, C, H, W = 1, 64, 40, 40
    dummy_input = torch.randn(B, C, H, W)
    
    # 建立 Head
    head = YOLO6DHead(in_channels=64, num_classes=80)
    
    # 向前傳播
    box, kpt, rot, depth = head(dummy_input)
    
    print(f"Input Features: {dummy_input.shape}")
    print(f"Box Head Output (4 + Nc): {box.shape}")
    print(f"Keypoint Head Output (9x3): {kpt.shape}")
    print(f"Rotation Head Output (9): {rot.shape}")
    print(f"Depth Head Output (1 with Sigmoid): {depth.shape}")
