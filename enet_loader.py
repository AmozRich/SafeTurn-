# ENet Model Loader for SafeTurn+
# This module handles loading the pretrained LaneNet/ENet model

import os
import sys
import torch
import cv2
import numpy as np
import torch.nn.functional as F

# Add enet directory to path
enet_dir = os.path.join(os.path.dirname(__file__), 'enet')
if enet_dir not in sys.path:
    sys.path.insert(0, enet_dir)

from lane_detector import ENet

def load_enet_model(model_path, device="cuda"):
    """
    Load the pretrained LaneNet/ENet model.
    
    Args:
        model_path: Path to the .pth model file
        device: 'cuda' or 'cpu'
        
    Returns:
        model: Loaded PyTorch model in eval mode
    """
    # Initialize the model architecture
    # binary_seg=2 (background + lane), embedding_dim=4 (for instance segmentation)
    model = ENet(binary_seg=2, embedding_dim=4)
    
    # Load the pretrained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # The checkpoint is a state_dict (OrderedDict)
    # Load it directly into the model
    model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    return model


class ENetLaneDetector:
    """
    Simplified ENet wrapper for SafeTurn+ integration.
    Uses the pre-trained LaneNet model for dense lane segmentation.
    """
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.enabled = False
        
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = load_enet_model(model_path, device)
            self.enabled = True
            print(f"[ENet] Model loaded successfully on {device}")
            
        except Exception as e:
            print(f"[ENet] Failed to load model: {e}")
            print(f"[ENet] Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("[ENet] Falling back to traditional detection")
            self.enabled = False
            return
        
        # Standard input dimensions for TuSimple dataset
        self.input_w = 512
        self.input_h = 256
    
    def infer(self, frame_bgr):
        """
        Run inference on a BGR frame.
        
        Args:
            frame_bgr: Input frame in BGR format (H, W, 3)
            
        Returns:
            Binary lane mask (H, W) uint8, values 0-255
        """
        if not self.enabled:
            return None
        
        H, W, _ = frame_bgr.shape
        
        # Preprocess: Resize and convert to grayscale
        # Model architecture expects 1-channel grayscale input
        img = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        
        # Better normalization for grayscale (0-1 range works best for this model)
        # Removed the (img - 0.5) / 0.5 scaling which was too aggressive
        pass # img is already 0-1 from the line above
        
        # Add channel dimension and convert to tensor
        img = img[..., None]  # (H, W, 1)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
        img = img.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(img)
       
        # Post-process: Extract binary segmentation
        mask = self._extract_binary_mask(output)
        
        # Debug: Print statistics every 30 frames
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        
        if self._frame_count % 30 == 0:
            white_px = np.sum(mask == 255)
            total_px = mask.shape[0] * mask.shape[1]
            pct = (white_px / total_px) * 100
            print(f"[ENet Debug] Frame {self._frame_count}: Detected {pct:.2f}% lane pixels ({white_px}/{total_px})")
        
        # Resize back to original frame size
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def _extract_binary_mask(self, model_output):
        """
        Extract binary lane mask from model output.
        LaneNet returns (binary_segmentation, instance_segmentation).
        """
        # LaneNet model returns a tuple
        if isinstance(model_output, tuple):
            binary_logits = model_output[0]  # (1, 2, H, W)
        else:
            binary_logits = model_output
        
        # Apply softmax and extract lane channel (index 1)
        probs = F.softmax(binary_logits, dim=1)
        lane_prob = probs[:, 1, :, :]  # Get lane channel
        lane_prob_np = lane_prob.squeeze().cpu().numpy()
        
        # Debug: Check probability statistics
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        self._debug_count += 1
        
        if self._debug_count % 30 == 0:
            print(f"[ENet Debug] Probability range: min={lane_prob_np.min():.4f}, max={lane_prob_np.max():.4f}, mean={lane_prob_np.mean():.4f}")
            # Count pixels at different thresholds
            th_03 = np.sum(lane_prob_np > 0.3)
            th_05 = np.sum(lane_prob_np > 0.5)
            th_07 = np.sum(lane_prob_np > 0.7)
            print(f"[ENet Debug] Pixels > 0.3: {th_03}, > 0.5: {th_05}, > 0.7: {th_07}")
        
        # Lower threshold to 0.25 for better lane pixel recovery
        mask = (lane_prob_np > 0.25).astype(np.uint8) * 255
        
        # Morphological dilation to thicken lanes
        # This helps with histogram detection and visibility
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
