import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from models.backbone.e_elan import E_ELAN_Block
from models.neck.fpn_pan import FPN_PAN
from models.head.yolo6d_head import YOLO6DHead
from models.yolo6d import YOLO6D

def load_image_tensor(path, target_size=None):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    img_pil = Image.open(path).convert("RGB")
    t_list = []
    if target_size is not None:
        t_list.append(transforms.Resize(target_size))
    t_list.append(transforms.ToTensor())
    transform = transforms.Compose(t_list)
    return transform(img_pil).unsqueeze(0)

def visualize_eelan(image_path):
    print("Initializing E_ELAN visualization process...")
    img = load_image_tensor(image_path, target_size=(640, 640))
    if img is None:
        return
    model = E_ELAN_Block(c1=3, c2=16, c3=32, n=5)
    model.eval()
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    model.cv1.register_forward_hook(get_features("cv1 (Branch 1)"))
    model.cv2.register_forward_hook(get_features("cv2 (Branch 2)"))
    for i, layer in enumerate(model.cv3):
        layer.register_forward_hook(get_features(f"cv3_{i} (Stack Layer {i + 1})"))
    model.cv4.register_forward_hook(get_features("cv4 (Final Output)"))

    with torch.no_grad():
        _ = model(img)

    plt.figure(figsize=(15, 12))
    plt.subplot(3, 3, 1)
    plt.imshow(img[0].permute(1, 2, 0))
    plt.title(f"Input RGB\n{list(img.shape)}")
    plt.axis("off")

    plot_idx = 2
    for name, fmap in features.items():
        if plot_idx > 9:
            break
        plt.subplot(3, 3, plot_idx)
        avg_map = fmap[0].mean(dim=0)
        if avg_map.shape != img.shape[2:]:
            avg_map = F.interpolate(avg_map.unsqueeze(0).unsqueeze(0), size=img.shape[2:], mode="bilinear", align_corners=False).squeeze()
        plt.imshow(avg_map, cmap="viridis")
        plt.title(f"{name}\nOrig: {list(fmap.shape)}")
        plt.axis("off")
        plot_idx += 1
    plt.tight_layout()
    print("Displaying chart... (please check the popup window)")
    plt.show(block=True)

def visualize_fpn_pan(image_path):
    print("\nInitializing FPN+PAN visualization process...")
    img = load_image_tensor(image_path, target_size=(640, 640))
    if img is None: return
    _, _, H, W = img.shape
    h5, w5 = max(1, H // 32), max(1, W // 32)
    h4, w4 = h5 * 2, w5 * 2
    h3, w3 = h4 * 2, w4 * 2

    p5_raw = F.interpolate(img, size=(h5, w5), mode="bilinear", align_corners=False)
    p4_raw = F.interpolate(img, size=(h4, w4), mode="bilinear", align_corners=False)
    p3_raw = F.interpolate(img, size=(h3, w3), mode="bilinear", align_corners=False)

    def expand_channels(tensor, target_c):
        return tensor.repeat(1, (target_c // 3) + 1, 1, 1)[:, :target_c, :, :]

    p3 = expand_channels(p3_raw, 32)
    p4 = expand_channels(p4_raw, 64)
    p5 = expand_channels(p5_raw, 128)

    model = FPN_PAN(in_channels=[32, 64, 128], out_channels=[32, 64, 128])
    model.eval()

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    model.elan_p4_fpn.register_forward_hook(get_features("FPN P4 (Top-down)"))
    model.elan_p3_fpn.register_forward_hook(get_features("FPN P3 (Out Small)"))
    model.elan_p4_pan.register_forward_hook(get_features("PAN P4 (Out Medium)"))
    model.elan_p5_pan.register_forward_hook(get_features("PAN P5 (Out Large)"))

    with torch.no_grad():
        _ = model(p3, p4, p5)

    plt.figure(figsize=(15, 8))
    plt.suptitle(f"FPN+PAN Feature Visualization (Resized to {img.shape[2]}x{img.shape[3]})", fontsize=16)

    plot_items = [
        ("Input P3", p3), ("Input P4", p4), ("Input P5", p5), None,
        ("FPN P3 (Out)", features["FPN P3 (Out Small)"]),
        ("FPN P4 (Mid)", features["FPN P4 (Top-down)"]),
        ("PAN P4 (Out)", features["PAN P4 (Out Medium)"]),
        ("PAN P5 (Out)", features["PAN P5 (Out Large)"]),
    ]
    for i, item in enumerate(plot_items):
        if item is None: continue
        name, tensor = item
        plt.subplot(2, 4, i + 1)
        avg_map = tensor[0].mean(0)
        if avg_map.shape != img.shape[2:]:
            avg_map = F.interpolate(avg_map.unsqueeze(0).unsqueeze(0), size=img.shape[2:], mode="bilinear", align_corners=False).squeeze()
        plt.imshow(avg_map, cmap="viridis")
        plt.title(f"{name}\nOrig: {list(tensor.shape)}")
        plt.axis("off")
    plt.tight_layout()
    plt.show(block=True)

def visualize_heads():
    print("\nInitializing 6D Pose Prediction Heads visualization process...")
    B, C, H, W = 1, 64, 40, 40
    p4_feature = torch.randn(B, C, H, W)
    num_classes = 10
    head = YOLO6DHead(in_channels=64, num_classes=num_classes)
    head.eval()
    with torch.no_grad():
        box_out, kpt_out, rot_out, depth_out = head(p4_feature)

    plt.figure(figsize=(15, 6))
    plt.suptitle("6D Pose Prediction Heads Feature Outputs (P4 Scale)", fontsize=16)
    plot_items = [
        ("Input Feature\n(Random Noise)", p4_feature),
        (f"Box Head\n(4+{num_classes} channels)", box_out),
        ("Keypoint Head\n(27 channels)", kpt_out),
        ("Rotation Head\n(9 channels)", rot_out),
        ("Depth Head\n(1 channel, Sigmoid)", depth_out)
    ]
    for i, (title, tensor) in enumerate(plot_items):
        plt.subplot(1, 5, i + 1)
        avg_map = tensor[0].mean(dim=0).detach().numpy()
        plt.imshow(avg_map, cmap="plasma")
        plt.title(f"{title}\nShape: {list(tensor.shape[1:])}")
        plt.axis("off")
    plt.tight_layout()
    plt.show(block=True)

def visualize_yolo6d_full(image_path, weights_path=None):
    print("\nInitializing Full YOLO-6D Architecture visualization process...")
    img = load_image_tensor(image_path, target_size=(640, 640))
    if img is None: return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO6D(in_channels=3, base_channels=16, num_classes=80)
    
    if weights_path is not None and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        print("Using random initialized weights (no weights provided or path invalid).")
        
    model.to(device)
    img = img.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(img)
    scale_idx = 0
    scale_name = "P3 (Small Objects)"
    # outputs[scale_idx] is a tuple of (box_out, kpt_out, rot_out, depth_out)
    # move back to cpu for visualization
    box_out = outputs[scale_idx][0].cpu()
    kpt_out = outputs[scale_idx][1].cpu()
    rot_out = outputs[scale_idx][2].cpu()
    depth_out = outputs[scale_idx][3].cpu()
    img_cpu = img.cpu()
    
    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Full YOLO-6D End-to-End Output ({scale_name})", fontsize=16)
    plot_items = [
        (f"Box Head\n{list(box_out.shape[1:])}", box_out),
        (f"Keypoint Head\n{list(kpt_out.shape[1:])}", kpt_out),
        (f"Rotation Head\n{list(rot_out.shape[1:])}", rot_out),
        (f"Depth Head\n{list(depth_out.shape[1:])}", depth_out)
    ]
    plt.subplot(1, 5, 1)
    plt.imshow(img_cpu[0].permute(1, 2, 0).numpy())
    plt.title(f"Input Image\n{list(img_cpu.shape[2:])}")
    plt.axis("off")
    
    for i, (title, tensor) in enumerate(plot_items):
        plt.subplot(1, 5, i + 2)
        avg_map = tensor[0].mean(dim=0).unsqueeze(0).unsqueeze(0)
        avg_map_resized = F.interpolate(avg_map, size=img_cpu.shape[2:], mode="bilinear", align_corners=False).squeeze()
        plt.imshow(avg_map_resized.numpy(), cmap="plasma")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show(block=True)
