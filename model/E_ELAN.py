import torch
import torch.nn as nn

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    # 標準的卷積層組合：Conv2d + BatchNorm + SiLU (Swish)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class E_ELAN(nn.Module):
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
        # 這裡模擬 YOLOv7 的結構，每兩層為一組進行特徵聚合
        self.cv3 = nn.ModuleList([Conv(c2, c2, 3, 1) for _ in range(n)])
        
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

# Test the E-ELAN module
if __name__ == "__main__":
    # 假設輸入影像為 (Batch Size=1, Channels=64, H=32, W=32)
    input_tensor = torch.randn(1, 64, 32, 32)
    
    # 建立 E-ELAN 模組
    # 輸入 64 通道，中間處理 32 通道，輸出 128 通道
    eelan_block = E_ELAN(c1=64, c2=32, c3=128)
    print(eelan_block)
    
    output = eelan_block(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")