import matplotlib
import argparse
from utils.visualization import (
    visualize_eelan,
    visualize_fpn_pan,
    visualize_heads,
    visualize_yolo6d_full
)

# Use Qt5Agg for separate window popups
matplotlib.use("Qt5Agg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-6D Inference and Visualization Tool")
    parser.add_argument("--image", type=str, default="test.jpg", help="Path to input test image")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights (.pt)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip visualization of internal modules")
    args = parser.parse_args()
    
    print("Testing YOLO-6D End-to-End Architecture Inference...\n")
    
    # Run the visualisations via the separated module
    if not args.skip_tests:
        visualize_eelan(args.image)
        visualize_fpn_pan(args.image)
        visualize_heads()
        
    visualize_yolo6d_full(args.image, weights_path=args.weights)
