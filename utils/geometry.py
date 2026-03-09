import torch
import torch.nn.functional as F

def svd_orthogonalize(rot_9d):
    """
    將 9D 旋轉向量轉換為合法的 3x3 SO(3) 旋轉矩陣
    
    Args:
        rot_9d: 預測的連續 9D 旋轉向量 [B, 9] (或 [B, 9, H, W] 等任意形狀，這裡以 batch Flatten 的角度示範)
    
    Returns:
        rot_3x3: 合法的 3x3 旋轉矩陣 [B, 3, 3]
    """
    # 確保最後一個維度為 9，然後重塑為 3x3
    original_shape = rot_9d.shape
    B = rot_9d.shape[0] if len(original_shape) > 1 else 1
    
    # 這裡簡化處理：假設傳入的 shape 是 [B, 9]
    rot_3x3 = rot_9d.view(-1, 3, 3) 
    
    # SVD 奇異值分解 (M = U * S * V^T)
    U, _, V = torch.svd(rot_3x3)
    
    # 重新建構正交矩陣 R = U * V^T
    R = torch.bmm(U, V.transpose(1, 2))
    
    # 確保行列式為 +1 (SO(3) 約束，去除反射鏡像情況)
    det = torch.det(R)
    # 建立一個與 V 形狀相同的修正矩陣
    modifier = torch.ones_like(V)
    
    # 避免 In-place 操作導致 Backward 報錯 (修正 RuntimeError)
    # 不使用 modifier[:, :, 2] = ...
    # 而是用另一個 tensor 將 det 值與其相乘後拼接回去
    modifier_last_col = modifier[:, :, 2] * det.unsqueeze(-1)
    
    # 將前面兩欄和修改後的第三欄拼接
    modifier_new = torch.cat([modifier[:, :, :2], modifier_last_col.unsqueeze(-1)], dim=-1)
    
    V_modified = V * modifier_new
    R_final = torch.bmm(U, V_modified.transpose(1, 2))
    
    return R_final

def geodesic_distance(R_pred, R_gt):
    """
    計算兩個旋轉矩陣之間的測地線距離 (Geodesic distance)
    d(R_pred, R_gt) = arccos((trace(R_pred^T * R_gt) - 1) / 2)
    
    Args:
        R_pred: 預測的旋轉矩陣 [B, 3, 3]
        R_gt: 真實的旋轉矩陣 [B, 3, 3]
        
    Returns:
        測地線距離 (角度誤差) [B]
    """
    # R_pred^T * R_gt
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    
    # Trace(R_diff) 計算每個矩陣對角線元素總和
    trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(dim=-1)
    
    # 計算 arccos 輸入的數值，為了數值穩定性，將其夾在 [-1+eps, 1-eps] 範圍內
    # 否則由於浮點數誤差，trace 可能會稍微大於 3，導致 nan
    val = (trace - 1) / 2.0
    val = torch.clamp(val, -1.0 + 1e-6, 1.0 - 1e-6)
    
    return torch.acos(val)

if __name__ == "__main__":
    # Test Geometry functions
    print("Testing Geometry utilities...")
    dummy_9d = torch.randn(2, 9)
    print(f"Input 9D Rotation: {dummy_9d.shape}")
    
    R_pred = svd_orthogonalize(dummy_9d)
    print(f"SVD Orthogonalized 3x3: {R_pred.shape}")
    print(f"Determinants (should be 1.0): {torch.det(R_pred)}")
    
    # Generate dummy Ground Truth Orthogonal Matrix
    R_gt = svd_orthogonalize(torch.randn(2, 9))
    
    dist = geodesic_distance(R_pred, R_gt)
    print(f"Geodesic distance: {dist}")
