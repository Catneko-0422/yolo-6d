import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.geometry import svd_orthogonalize, geodesic_distance

class YOLO6DLoss(nn.Module):
    def __init__(self, lambda_R=1.0, lambda_t=1.0, lambda_kp=1.0, lambda_bb=1.0):
        super().__init__()
        self.lambda_R = lambda_R
        self.lambda_t = lambda_t
        self.lambda_kp = lambda_kp
        self.lambda_bb = lambda_bb
        
        # Translation Loss 使用 Smooth L1 Loss (beta=1.0 為預設)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def rotation_loss(self, rot_pred_9d, rot_gt_3x3):
        """
        L_R: 測地線距離損失
        rot_pred_9d: [B, 9] (或先經 reshape 處理到此維度)
        rot_gt_3x3: [B, 3, 3] 標註的 Ground Truth 旋轉矩陣
        """
        # 1. 將 9D 向量轉換為 3x3 矩陣
        R_pred = svd_orthogonalize(rot_pred_9d)
        
        # 2. 計算測地線距離
        distance = geodesic_distance(R_pred, rot_gt_3x3)
        return distance.mean()

    def translation_loss(self, depth_pred, depth_gt):
        """
        L_t: 深度尺度 (Sigma) 平移損失
        depth_pred: [B, 1] 預測的深度縮放係數 (通常介於 0~1)
        depth_gt: [B, 1] 真實的深度縮放係數
        """
        return self.smooth_l1(depth_pred, depth_gt)

    def keypoint_loss(self, kpt_pred, kpt_gt, vis_mask):
        """
        L_kp: 關鍵點可見度掩蔽 L2 損失
        kpt_pred: [B, NumPoints, 2] (預測的 x, y 座標)
        kpt_gt: [B, NumPoints, 2] (真實的 x, y 座標)
        vis_mask: [B, NumPoints] (可見度掩蔽，0=被遮擋或無標註, 1=可見)
        """
        # 計算 L2 距離 (MSE) (B, NumPoints, 2) -> 對座標軸平方後相加 -> (B, NumPoints)
        sq_dist = torch.sum((kpt_pred - kpt_gt) ** 2, dim=-1)
        
        # 套用 Mask (被遮蔽的點不參與 loss 計算)
        masked_dist = sq_dist * vis_mask
        
        # 只對可見的點求平均 (避免除以 0，加個小 epsilon)
        visible_counts = vis_mask.sum()
        if visible_counts > 0:
            return masked_dist.sum() / visible_counts
        else:
            return torch.tensor(0.0).to(kpt_pred.device)

    def bbox_loss(self, box_pred, box_gt):
        """
        L_bb: YOLO BBox Loss (CIoU + DFL)
        這裡用簡單的 L1 作為示意 Placeholder，在真實 YOLOv11 訓練中，
        此函式通常直接呼叫 Ultralytics v11 原生的 BboxLoss 模組。
        """
        # TODO: Replace with official YOLO CIoU + DFL
        return nn.functional.l1_loss(box_pred, box_gt)

    def forward(self, preds, targets):
        """
        整體 Loss 計算
        preds: Dict 包含四個 head 的預測 Tensor
        targets: Dict 包含對應的 Ground Truth Tensor
        """
        # 1. Rotation Loss
        l_r = self.rotation_loss(preds['rot'], targets['rot'])
        
        # 2. Translation (Depth) Loss
        l_t = self.translation_loss(preds['depth'], targets['depth'])
        
        # 3. Keypoint Masked Loss
        l_kp = self.keypoint_loss(preds['kpt'], targets['kpt'], targets['vis_mask'])
        
        # 4. BBox Loss
        l_bb = self.bbox_loss(preds['box'], targets['box'])
        
        # 加權組合
        total_loss = (self.lambda_R * l_r + 
                      self.lambda_t * l_t + 
                      self.lambda_kp * l_kp + 
                      self.lambda_bb * l_bb)
                      
        loss_items = {
            'total_loss': total_loss,
            'L_R': l_r,
            'L_t': l_t,
            'L_kp': l_kp,
            'L_bb': l_bb
        }
        return total_loss, loss_items

if __name__ == "__main__":
    print("Testing YOLO-6D Loss functions...")
    
    # 建立一個 Batch Size = 2 的 Dummy Data
    B = 2
    N_KPT = 9
    
    preds = {
        'rot': torch.randn(B, 9),                 # 9D 旋轉預測
        'depth': torch.rand(B, 1),                # Depth [0, 1] 之間的縮放比例
        'kpt': torch.randn(B, N_KPT, 2),          # 預測的關鍵點 xy 座標
        'box': torch.randn(B, 4)                  # 預測的 cx, cy, w, h
    }
    
    targets = {
        'rot': svd_orthogonalize(torch.randn(B, 9)), # 真實的 3x3 旋轉矩陣
        'depth': torch.rand(B, 1),                   # 真實的 Depth
        'kpt': torch.randn(B, N_KPT, 2),             # 真實的關鍵點 xy 座標
        # 模擬某些點被遮蔽 (0 或 1)
        'vis_mask': torch.randint(0, 2, (B, N_KPT)).float(), 
        'box': torch.randn(B, 4)
    }
    
    # 初始化 Loss
    # 假設這四個任務我們給予不同的權重，如 $\lambda$ 參數
    criterion = YOLO6DLoss(lambda_R=2.0, lambda_t=1.0, lambda_kp=1.5, lambda_bb=0.5)
    
    total_loss, logs = criterion(preds, targets)
    
    print("\n--- Loss Calculation Results ---")
    for k, v in logs.items():
        print(f"{k:>12}: {v.item():.4f}")
