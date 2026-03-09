import torch
import torch.nn as nn

from utils.general import autopad

class Conv(nn.Module):
    # 標準的卷積層組合：Conv2d + BatchNorm + SiLU (Swish)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class E_ELAN_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=5, groups=2):
        """
        c1: 輸入通道數
        c2: 中間處理通道數
        c3: 最終輸出通道數
        n: 堆疊層數
        groups: 分組卷積的數量
        """
        super().__init__()
        self.c2 = c2
        # 第一部分：分支 1
        self.cv1 = Conv(c1, c2, 1, 1)
        # 第二部分：分支 2
        self.cv2 = Conv(c1, c2, 1, 1)
        
        # 中間的特徵提取層（通常為多個 3x3 卷積）
        # 這裡模擬 YOLOv11 的結構，每兩層為一組進行特徵聚合
        self.cv3 = nn.ModuleList([Conv(c2, c2, 3, 1, g=groups) for _ in range(n)])
        
        # 最終的整合卷積
        # 輸入是分支 1 + 分支 2 + 所有的 cv3 輸出拼接
        self.cv4 = Conv(c2 * (2 + n), c3, 1, 1)

    def forward(self, x):
        # 初始化列表來存放待拼接的特徵圖
        outputs = []
        
        # 分支 1 和分支 2 的初始處理
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        outputs.extend([y1, y2])
        
        # 逐層通過卷積層並記錄結果
        y_temp = y2
        for i in range(len(self.cv3)):
            y_temp = self.cv3[i](y_temp)
            outputs.append(y_temp)
            
        # 將所有分支結果在通道維度（dim=1）進行拼接（Concatenation）
        out = torch.cat(outputs, dim=1)
        
        # 通過最後的 1x1 卷積進行特徵融合
        return self.cv4(out)

class E_ELAN(nn.Module):
    """
    完整的 E-ELAN Backbone
    包含 5 個階段 (Stages)，每個階段由一個 stride=2 的降採樣卷積和一個 E_ELAN_Block 組成。
    """
    def __init__(self, in_channels=3, base_channels=16, n=5, groups=2):
        super().__init__()
        self.stages = nn.ModuleList()
        
        # 假設 5 個階段的輸出通道數分別為 base_channels 乘以 2 的冪次
        # 例如: base_channels=16 -> channels = [16, 32, 64, 128, 256]
        channels = [base_channels * (2 ** i) for i in range(5)]
        
        c_in = in_channels
        for c_out in channels:
            # 降採樣卷積，stride=2
            downsample = Conv(c_in, c_out, k=3, s=2)
            # E_ELAN 區塊，中間處理通道 (c2) 通常設為輸出通道的一半
            elan_block = E_ELAN_Block(c1=c_out, c2=c_out // 2, c3=c_out, n=n, groups=groups)
            
            # 使用 nn.Sequential 將降採樣和 ELAN 區塊組合為一個階段
            self.stages.append(nn.Sequential(downsample, elan_block))
            c_in = c_out

    def forward(self, x):
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        # 通常 Backbone 會返回多個尺度的特徵圖，例如最後三個階段的輸出 (P3, P4, P5)
        return outputs

# Test the E-ELAN module
if __name__ == "__main__":
    # 測試單一 E_ELAN_Block
    print("Testing E_ELAN_Block:")
    block_input = torch.randn(1, 64, 32, 32)
    eelan_block = E_ELAN_Block(c1=64, c2=32, c3=128)
    block_output = eelan_block(block_input)
    print(f"Block Input shape: {block_input.shape}")
    print(f"Block Output shape: {block_output.shape}\n")

    # 測試完整 Backbone
    print("Testing E_ELAN Backbone:")
    img_input = torch.randn(1, 3, 640, 640)
    backbone = E_ELAN(in_channels=3, base_channels=16)
    features = backbone(img_input)
    print(f"Backbone Input shape: {img_input.shape}")
    for i, feat in enumerate(features):
        print(f"Stage {i+1} Output shape: {feat.shape}")