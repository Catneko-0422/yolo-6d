import torch
import torch.nn as nn
from models.backbone.e_elan import E_ELAN
from models.neck.fpn_pan import FPN_PAN
from models.head.yolo6d_head import YOLO6DHead

class YOLO6D(nn.Module):
    """
    Unified YOLO-6D Architecture
    Combines:
    1. E-ELAN Backbone
    2. FPN + PAN Neck
    3. Multi-task Heads (Box, Keypoints, Rotation, Depth)
    """
    def __init__(self, in_channels=3, base_channels=16, num_classes=80):
        super().__init__()
        # Backbone: returns 5 stages
        self.backbone = E_ELAN(in_channels=in_channels, base_channels=base_channels)
        
        # P3, P4, P5 channels based on E_ELAN channels = [16, 32, 64, 128, 256]
        c3, c4, c5 = base_channels * 4, base_channels * 8, base_channels * 16 # 64, 128, 256
        
        # Neck
        self.neck = FPN_PAN(in_channels=[c3, c4, c5], out_channels=[c3, c4, c5])
        
        # Heads: one for each scale (P3, P4, P5)
        self.heads = nn.ModuleList([
            YOLO6DHead(in_channels=c3, num_classes=num_classes), # Small objects
            YOLO6DHead(in_channels=c4, num_classes=num_classes), # Medium objects
            YOLO6DHead(in_channels=c5, num_classes=num_classes)  # Large objects
        ])

    def forward(self, x):
        # 1. Backbone: Extract multi-scale features
        features = self.backbone(x)
        # Select P3, P4, P5 (outputs from stages 3, 4, 5)
        p3, p4, p5 = features[2], features[3], features[4]
        
        # 2. Neck: Feature fusion
        p3_out, p4_out, p5_out = self.neck(p3, p4, p5)
        
        # 3. Heads: 4 parallel prediction tasks across 3 scales
        outputs = []
        for i, feat in enumerate([p3_out, p4_out, p5_out]):
            # feat represents a single scale feature map
            # Output contains (box_out, kpt_out, rot_out, depth_out)
            outputs.append(self.heads[i](feat))
            
        # Return format: list of tuples (predictions at each scale)
        return outputs

# Quick Test
if __name__ == "__main__":
    print("Testing Complete YOLO-6D Model:")
    dummy_input = torch.randn(1, 3, 640, 640)
    model = YOLO6D(num_classes=80)
    
    outputs = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of output scales: {len(outputs)} (P3, P4, P5)")
    for i, out in enumerate(outputs):
        scale_name = ["P3 (Small)", "P4 (Medium)", "P5 (Large)"][i]
        box, kpt, rot, depth = out
        print(f"--- {scale_name} ---")
        print(f"  Box output: {box.shape}")
        print(f"  Keypoint output: {kpt.shape}")
        print(f"  Rotation output: {rot.shape}")
        print(f"  Depth output: {depth.shape}")
