import matplotlib
from utils.visualization import (
    visualize_eelan,
    visualize_fpn_pan,
    visualize_heads,
    visualize_yolo6d_full
)

# Use Qt5Agg for separate window popups
matplotlib.use("Qt5Agg")

if __name__ == "__main__":
    IMAGE_PATH = "test.jpg"
    print("Testing YOLO-6D End-to-End Architecture Inference...\n")
    
    # Run the visualisations via the separated module
    visualize_eelan(IMAGE_PATH)
    visualize_fpn_pan(IMAGE_PATH)
    visualize_heads()
    visualize_yolo6d_full(IMAGE_PATH)
