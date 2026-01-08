# ENet Setup & Installation

## 🔧 Step 1: Install Dependencies

### Option A: CPU-Only (Slower, but works everywhere)
```bash
pip install torch torchvision opencv-python numpy
```

### Option B: GPU-Accelerated (Recommended for laptop with NVIDIA GPU)

1. **Check your CUDA version:**
   ```bash
   nvidia-smi
   ```
   Look for "CUDA Version" in the output (e.g., 11.8, 12.1)

2. **Install PyTorch with CUDA:**
   
   Visit: https://pytorch.org/get-started/locally/
   
   **For CUDA 11.8 (most common):**
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
   
   **For CUDA 12.1:**
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install other dependencies:**
   ```bash
   pip install opencv-python numpy
   ```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
```

---

## 📦 Step 2: Get Pretrained ENet Model

### Where to Find Models

1. **Official ENet TuSimple:**
   - Search GitHub for: "ENet lane detection TuSimple pretrained"
   - Look for `.pth` or `.pt` files
   - Recommended repos:
     - https://github.com/cardwing/Codes-for-Lane-Detection
     - https://github.com/davidtvs/PyTorch-ENet

2. **Model Requirements:**
   - **Input shape:** (batch, 3, 256, 512) or similar
   - **Output shape:** (batch, 2, 256, 512) for binary segmentation
   - **Dataset:** TuSimple or CULane (both work for general roads)
   - **Format:** PyTorch `.pth` file

3. **Download and Place:**
   ```bash
   # Download model to project directory
   # Rename it to:
   d:\MAIN-PROJECT\enet_lane_model.pth
   ```

### Alternative: Test Without ENet First

You can run the code **without** ENet to verify everything else works:
```bash
python main.py
```

Expected output:
```
[ENet] Initializing lane detector...
[ENet] Failed to load model: [Errno 2] No such file or directory: 'enet_lane_model.pth'
[ENet] Falling back to traditional detection
Reading video from: drive.mp4
```

The system will automatically use the traditional `detect_lane_pixels()` method.

---

## 🧪 Step 3: Test ENet Integration

Once you have the model file:

```bash
python main.py
```

**Check console output:**
```
[ENet] Initializing lane detector...
[ENet] Model loaded successfully on cuda    <-- Good!
Reading video from: drive.mp4
```

**Check windows:**
- Main window: "SafeTurn+ Main" (should show Bézier curves)
- Debug window: "ENet Lane Mask (Debug)" (should show white lane pixels)

**Visual Quality Check:**
✅ Continuous lane boundaries (not dots)  
✅ Smooth curves on turns  
✅ Stable detection (no flickering)  
✅ Dense pixels (fills the lane width)

---

## ⚡ Performance Tuning

### If ENet is Too Slow:

1. **Reduce inference frequency:**
   ```python
   ENET_INFERENCE_INTERVAL = 3  # Run every 3rd frame (~10 FPS)
   ```

2. **Use CPU if GPU has issues:**
   ```python
   ENET_DEVICE = "cpu"
   ```

3. **Lower input resolution (requires model retraining):**
   ```python
   self.input_w = 256  # Instead of 512
   self.input_h = 128  # Instead of 256
   ```

### Expected Performance:
- **GPU (CUDA):** 6-12 ms per inference → 30-45 FPS total
- **CPU:** 30-80 ms per inference → 15-20 FPS total (with caching)

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Use CPU mode
```python
ENET_DEVICE = "cpu"
```

### Issue: "RuntimeError: expected scalar type Float but found Half"
**Solution:** Model uses mixed precision. Fix:
```python
self.model = torch.load(model_path, map_location=device)
self.model = self.model.float()  # Add this line
```

### Issue: "Shape mismatch in _logits_to_mask"
**Solution:** Model output doesn't match expected format. Check:
```python
print(f"Logits shape: {logits.shape}")  # Add after inference
```
Adjust `_logits_to_mask()` accordingly.

### Issue: Weak detection warnings every frame
**Possible causes:**
1. Model trained on different dataset (highways vs city roads)
2. Input preprocessing mismatch (normalization)
3. Confidence threshold too high

**Fix:** Lower the threshold
```python
if binary_lane is not None and np.count_nonzero(binary_lane) >= 500:  # Was 1000
```

---

## 📁 Final File Structure

```
d:\MAIN-PROJECT\
│
├── main.py                          # Modified with ENet integration
├── drive.mp4                        # Your test video
├── enet_lane_model.pth              # Pretrained ENet model (download separately)
│
├── ENET_INTEGRATION_GUIDE.md        # Usage guide
├── ENET_SETUP.md                    # This file
├── requirements.txt                 # Dependencies list
```

---

## 🚀 Quick Start Checklist

- [ ] Install PyTorch (with CUDA if available)
- [ ] Install OpenCV and NumPy
- [ ] Verify imports: `python -c "import torch, cv2"`
- [ ] Run without ENet: `python main.py` (should work with fallback)
- [ ] Download pretrained ENet model
- [ ] Place model as `enet_lane_model.pth`
- [ ] Run with ENet: `python main.py`
- [ ] Check debug window for quality
- [ ] Verify smooth curves in main window

---

## 💡 Pro Tips

1. **Start with CPU mode first** to verify the integration works, then switch to GPU
2. **Use the debug window** - if the mask looks bad, curves will be bad
3. **The fallback is your friend** - traditional detection is a safety net
4. **Cache is critical** - running ENet every frame will kill FPS
5. **Model quality matters** - a good pretrained model is 80% of success

---

## 📞 Need Help?

Common errors and solutions are in `ENET_INTEGRATION_GUIDE.md`.

If stuck, check:
1. Console output (any `[ENet]` messages?)
2. Debug window (mask quality?)
3. GPU availability (`nvidia-smi`)
4. Model file exists and is valid PyTorch format

---

Ready to install! Start with **Step 1** above. 🚀
