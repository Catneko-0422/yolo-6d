import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from model.E_ELAN import E_ELAN
from model.FPN_PAN import FPN_PAN

# Set image path
IMAGE_PATH = "test.jpg"

def load_image_tensor(path, target_size=None):
    if not os.path.exists(path):
        print(f"Error: {path} not found. Please check if the image exists in the directory.")
        return None
    
    img_pil = Image.open(path).convert('RGB')
    # Resize if target size is specified, otherwise keep original size
    t_list = []
    if target_size is not None:
        t_list.append(transforms.Resize(target_size))
    t_list.append(transforms.ToTensor())
    transform = transforms.Compose(t_list)
    return transform(img_pil).unsqueeze(0) # [1, 3, H, W]

def visualize_eelan():
    print("Initializing E_ELAN visualization process...")
    
    img = load_image_tensor(IMAGE_PATH, target_size=(640, 640))
    if img is None: return
    print(f"E_ELAN input image shape: {img.shape}")

    # 2. Instantiate E_ELAN
    # c1=3 (corresponds to RGB input), c2=16, c3=32, n=5
    model = E_ELAN(c1=3, c2=16, c3=32, n=5)
    model.eval()

    # 3. Register Hook to capture output of each layer
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    # Register Hook to key layers
    model.cv1.register_forward_hook(get_features('cv1 (Branch 1)'))
    model.cv2.register_forward_hook(get_features('cv2 (Branch 2)'))
    
    # Capture each layer of the Stack
    for i, layer in enumerate(model.cv3):
        layer.register_forward_hook(get_features(f'cv3_{i} (Stack Layer {i+1})'))
        
    model.cv4.register_forward_hook(get_features('cv4 (Final Output)'))

    # 4. Forward Pass
    with torch.no_grad():
        _ = model(img)

    # 5. Plot feature maps using Matplotlib
    # We will plot Input + cv1 + cv2 + 5 stack layers + Output, total 9 images
    # Layout: 3x3 grid
    plt.figure(figsize=(15, 12))
    
    # Plot input
    plt.subplot(3, 3, 1)
    plt.imshow(img[0].permute(1, 2, 0)) # [C, H, W] -> [H, W, C] for plotting
    plt.title(f"Input RGB\n{list(img.shape)}")
    plt.axis('off')

    # Plot features of each layer
    # Since feature maps have multiple channels, we take the mean to display as a "heatmap"
    plot_idx = 2
    for name, fmap in features.items():
        if plot_idx > 9: break # Avoid exceeding grid
        plt.subplot(3, 3, plot_idx)
        avg_map = fmap[0].mean(dim=0) # [C, H, W] -> [H, W]
        
        # Resize back to original size for comparison (use bilinear interpolation for smoother heatmap)
        if avg_map.shape != img.shape[2:]:
            avg_map = F.interpolate(avg_map.unsqueeze(0).unsqueeze(0), size=img.shape[2:], mode='bilinear', align_corners=False).squeeze()
            
        plt.imshow(avg_map, cmap='viridis') # Use viridis colormap to show feature intensity
        plt.title(f"{name}\nOrig: {list(fmap.shape)}")
        plt.axis('off')
        plot_idx += 1

    plt.tight_layout()
    print("Displaying chart... (please check the popup window)")
    plt.show()

def visualize_fpn_pan():
    print("\nInitializing FPN+PAN visualization process...")
    
    img = load_image_tensor(IMAGE_PATH, target_size=(640, 640))
    if img is None: return
    
    # 1. Generate simulated P3, P4, P5 based on input image
    # To align FPN/PAN Upsample/Downsample, we need to calculate dimensions matching Stride
    _, _, H, W = img.shape
    
    # Calculate P5 (Stride 32) dimensions
    h5, w5 = max(1, H // 32), max(1, W // 32)
    # Infer P4 (Stride 16) and P3 (Stride 8)
    h4, w4 = h5 * 2, w5 * 2
    h3, w3 = h4 * 2, w4 * 2
    
    # Use interpolation to resize image to simulate Backbone output
    p5_raw = F.interpolate(img, size=(h5, w5), mode='bilinear', align_corners=False)
    p4_raw = F.interpolate(img, size=(h4, w4), mode='bilinear', align_corners=False)
    p3_raw = F.interpolate(img, size=(h3, w3), mode='bilinear', align_corners=False)
    
    # Expand channel count to match model settings [32, 64, 128]
    # Repeat stack RGB (3 channels) to fill channels
    def expand_channels(tensor, target_c):
        return tensor.repeat(1, (target_c // 3) + 1, 1, 1)[:, :target_c, :, :]
        
    p3 = expand_channels(p3_raw, 32)
    p4 = expand_channels(p4_raw, 64)
    p5 = expand_channels(p5_raw, 128)
    
    print(f"Simulated Inputs: P3={list(p3.shape)}, P4={list(p4.shape)}, P5={list(p5.shape)}")
    
    # 2. Instantiate FPN_PAN (use smaller channel counts for visualization)
    model = FPN_PAN(in_channels=[32, 64, 128], out_channels=[32, 64, 128])
    model.eval()
    
    # 3. Register Hook
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    model.elan_p4_fpn.register_forward_hook(get_features('FPN P4 (Top-down)'))
    model.elan_p3_fpn.register_forward_hook(get_features('FPN P3 (Out Small)'))
    model.elan_p4_pan.register_forward_hook(get_features('PAN P4 (Out Medium)'))
    model.elan_p5_pan.register_forward_hook(get_features('PAN P5 (Out Large)'))

    # 4. Forward
    with torch.no_grad():
        _ = model(p3, p4, p5)
        
    # 5. Plotting
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"FPN+PAN Feature Visualization (Resized to {img.shape[2]}x{img.shape[3]})", fontsize=16)
    
    # Display order: First row shows input, second row shows output feature changes
    plot_items = [
        ('Input P3', p3), ('Input P4', p4), ('Input P5', p5), None,
        ('FPN P3 (Out)', features['FPN P3 (Out Small)']), 
        ('FPN P4 (Mid)', features['FPN P4 (Top-down)']), 
        ('PAN P4 (Out)', features['PAN P4 (Out Medium)']), 
        ('PAN P5 (Out)', features['PAN P5 (Out Large)'])
    ]
    
    for i, item in enumerate(plot_items):
        if item is None: continue
        name, tensor = item
        plt.subplot(2, 4, i+1)
        
        # Take mean and Resize back to original size
        avg_map = tensor[0].mean(0)
        if avg_map.shape != img.shape[2:]:
            avg_map = F.interpolate(avg_map.unsqueeze(0).unsqueeze(0), size=img.shape[2:], mode='bilinear', align_corners=False).squeeze()
            
        plt.imshow(avg_map, cmap='viridis')
        plt.title(f"{name}\nOrig: {list(tensor.shape)}")
        plt.axis('off')
            
    plt.tight_layout()
    print("Displaying FPN+PAN chart... (please check the popup window)")
    plt.show()

if __name__ == "__main__":
    visualize_eelan()
    visualize_fpn_pan()