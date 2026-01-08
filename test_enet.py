# Quick guide for running with ENet
# 
# 1. ENet model installed at: enet/ENET.pth
# 2. Run this file: python test_enet.py
# 3. Should see: [ENet] Model loaded successfully

import sys
sys.path.insert(0, 'enet')

from enet_loader import ENetLaneDetector
import torch

def test_enet():
    print("Testing ENet model loading...")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Try to load the model
    enet = ENetLaneDetector("enet/ENET.pth", device=device)
    
    if enet.enabled:
        print("✓ ENet model loaded successfully!")
        print(f"✓ Input size: {enet.input_w} x {enet.input_h}")
        print("✓ Ready to process frames")
        
        # Test with a dummy frame
        import numpy as np
        import cv2
        
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        mask = enet.infer(dummy_frame)
        
        if mask is not None:
            print(f"✓ Inference test passed! Output shape: {mask.shape}")
        else:
            print("✗ Inference returned None")
    else:
        print("✗ ENet model failed to load")
        
if __name__ == "__main__":
    test_enet()
