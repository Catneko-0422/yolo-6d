import torch
import torch.nn as nn
from models.backbone.e_elan import E_ELAN_Block, Conv

class FPN_PAN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], out_channels=[256, 512, 1024]):
        """
        FPN+PAN using E_ELAN modules.
        in_channels: [P3_in, P4_in, P5_in] from backbone
        out_channels: [P3_out, P4_out, P5_out] to head
        """
        super().__init__()
        c3, c4, c5 = in_channels
        oc3, oc4, oc5 = out_channels

        # FPN (Top-down)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P5 -> P4
        # Input to E_ELAN: P4 + Upsample(P5)
        self.elan_p4_fpn = E_ELAN_Block(c4 + c5, oc4 // 2, oc4, n=5)
        
        # P4 -> P3
        # Input to E_ELAN: P3 + Upsample(P4_fpn)
        self.elan_p3_fpn = E_ELAN_Block(c3 + oc4, oc3 // 2, oc3, n=5)

        # PAN (Bottom-up)
        # P3 -> P4
        self.down_p3 = Conv(oc3, oc3, 3, 2) # Stride 2
        # Input to E_ELAN: P4_fpn + Downsample(P3_fpn)
        self.elan_p4_pan = E_ELAN_Block(oc4 + oc3, oc4 // 2, oc4, n=5)
        
        # P4 -> P5
        self.down_p4 = Conv(oc4, oc4, 3, 2) # Stride 2
        # Input to E_ELAN: P5 + Downsample(P4_pan)
        self.elan_p5_pan = E_ELAN_Block(c5 + oc4, oc5 // 2, oc5, n=5)

    def forward(self, p3, p4, p5):
        # FPN
        p5_up = self.up(p5)
        p4_cat = torch.cat([p4, p5_up], dim=1)
        p4_fpn = self.elan_p4_fpn(p4_cat)
        
        p4_up = self.up(p4_fpn)
        p3_cat = torch.cat([p3, p4_up], dim=1)
        p3_out = self.elan_p3_fpn(p3_cat) # Output P3 (Small)
        
        # PAN
        p3_down = self.down_p3(p3_out)
        p4_pan_cat = torch.cat([p4_fpn, p3_down], dim=1)
        p4_out = self.elan_p4_pan(p4_pan_cat) # Output P4 (Medium)
        
        p4_down = self.down_p4(p4_out)
        p5_pan_cat = torch.cat([p5, p4_down], dim=1)
        p5_out = self.elan_p5_pan(p5_pan_cat) # Output P5 (Large)
        
        return p3_out, p4_out, p5_out
