import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from models.yolo6d import YOLO6D
from utils.loss import YOLO6DLoss
from data.dataset import DummyYOLO6DDataset

def train():
    # 1. 參數設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    epochs = 100
    batch_size = 4
    learning_rate = 1e-3

    # 2. 準備資料與 DataLoader
    # 使用 Dummy Dataset 來測試架構
    dataset = DummyYOLO6DDataset(num_samples=32, img_size=640)
    
    # Collate function (處理 Dict Target 的 Batching)
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch], dim=0)
        # 將 dict array 轉為 batch tensor dict
        targets = {
            'box': torch.stack([b[1]['box'] for b in batch], dim=0),
            'kpt': torch.stack([b[1]['kpt'] for b in batch], dim=0),
            'vis_mask': torch.stack([b[1]['vis_mask'] for b in batch], dim=0),
            'rot': torch.stack([b[1]['rot'] for b in batch], dim=0),
            'depth': torch.stack([b[1]['depth'] for b in batch], dim=0)
        }
        return images, targets

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 3. 初始化模型與損失函數
    model = YOLO6D(in_channels=3, base_channels=16, num_classes=80).to(device)
    criterion = YOLO6DLoss(lambda_R=2.0, lambda_t=1.0, lambda_kp=1.5, lambda_bb=1.0).to(device)
    
    # 4. 初始化優化器 (Optimizer)
    # 使用 AdamW 來處理權重衰減，對大型 CNN/Transformer 有較好效果
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 5. 開始訓練迴圈
    print("Starting Training Loop...")
    model.train() # 切換至訓練模式 (啟用 BN, Dropout 追蹤)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            # 將 targets 移至對應設備
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # --- a. 梯度歸零 ---
            optimizer.zero_grad()
            
            # --- b. 向前傳播 (Forward Pass) ---
            # outputs 包含 P3, P4, P5 三個尺度的預測 Tensor
            outputs = model(images)
            
            # 簡化示範：我們只拿其中一個尺度 (例如 P4: Medium) 的預測結果與 target 算 loss
            # 實際的 YOLO 會需要把真實標註 mapping 到這三個特徵圖網路的 grid cell 上
            p4_out = outputs[1] # p4_out = (box_out, kpt_out, rot_out, depth_out)
            
            # 取得該 Batch 的空間維度中心點預測值作為示範 (H=40, W=40 的中心)
            # 在真實情況下，我們會尋找 "正樣本 (Positive Samples)" 的 Grid Cell 索引來計算 Loss
            H_center, W_center = p4_out[0].shape[-2] // 2, p4_out[0].shape[-1] // 2
            
            # 打包假想遇到的正樣本預測 Tensor
            preds = {
                'box': p4_out[0][:, :4, H_center, W_center], # 取 box_out 的前 4 個為 cx, cy, w, h
                'kpt': p4_out[1][:, :, H_center, W_center].view(-1, 9, 3)[..., :2], # 轉回 9x2
                'rot': p4_out[2][:, :, H_center, W_center],  # 9D vector
                'depth': p4_out[3][:, :, H_center, W_center] # depth scale 
            }
            
            # --- c. 計算損失 (Calculate Loss) ---
            loss, loss_items = criterion(preds, targets)
            
            # --- d. 反向傳播 (Backward Pass) ---
            loss.backward()
            
            # --- e. 更新權重 (Optimizer Step) ---
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # 印出每個 Epoch 的平均 Loss
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average Total Loss: {avg_loss:.4f} | "
              f"L_R: {loss_items['L_R']:.4f}, L_t: {loss_items['L_t']:.4f}, "
              f"L_kp: {loss_items['L_kp']:.4f}, L_bb: {loss_items['L_bb']:.4f}")

    print("Training finished!")
    
    # 儲存權重檔
    torch.save(model.state_dict(), "weights/yolo6d_last.pt")
    print("Model weights saved to weights/yolo6d_last.pt")

if __name__ == "__main__":
    train()
