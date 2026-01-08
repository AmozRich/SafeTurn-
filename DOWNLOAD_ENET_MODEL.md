# 📥 Download Pretrained ENet Model

## ✅ Recommended Option: Kaggle Model (Easiest)

The easiest way to get a working ENet model is from Kaggle.

### **Method 1: Download via Kaggle Website**

1. **Visit the model page:**
   ```
   https://www.kaggle.com/models/rangalamahesh/lane-detection-enet
   ```

2. **Download ENET.pth:**
   - Click on the "Download" button
   - Save the file as `ENET.pth`

3. **Rename and place the file:**
   ```powershell
   # Move to your project directory
   Move-Item -Path "~\Downloads\ENET.pth" -Destination "d:\MAIN-PROJECT\enet_lane_model.pth"
   ```

4. **Verify the file:**
   ```powershell
   # Check if file exists
   Test-Path "d:\MAIN-PROJECT\enet_lane_model.pth"
   ```

---

## 🔄 Alternative Option 2: LaneNet Model Zoo

If Kaggle doesn't work, try this SharePoint link:

### **Method 2: LaneNet Checkpoint**

1. **Visit the model zoo:**
   ```
   https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/qianshh_shanghaitech_edu_cn/ElU__fWwiZ1DtwlhKN0hfKUBqfmqkWep3Ey93pZ7y74TIQ?e=9kR88z
   ```

2. **Download the ENet checkpoint:**
   - Look for files with `enet` in the name
   - Common names: `enet_tusimple.pth`, `lanenet_enet.pth`
   - Download the `.pth` file

3. **Rename and place:**
   ```powershell
   Move-Item -Path "~\Downloads\<downloaded_file>.pth" -Destination "d:\MAIN-PROJECT\enet_lane_model.pth"
   ```

---

## 🛠️ Alternative Option 3: Use Kaggle CLI (For Power Users)

If you have Kaggle API set up:

```bash
# Install Kaggle CLI
pip install kaggle

# Download model (requires Kaggle API credentials)
kaggle models instances versions download rangalamahesh/lane-detection-enet/pyTorch/default

# Move to project
Move-Item ENET.pth d:\MAIN-PROJECT\enet_lane_model.pth
```

---

## 🔍 Verify Model Compatibility

After downloading, verify the model works with your code:

```powershell
cd d:\MAIN-PROJECT
python -c "import torch; model = torch.load('enet_lane_model.pth', map_location='cpu'); print('Model loaded successfully'); print(f'Model type: {type(model)}')"
```

**Expected output:**
```
Model loaded successfully
Model type: <class 'torch.nn.modules.container.Sequential'> 
# or similar PyTorch model type
```

---

## 🎯 Quick Test After Download

Once you have the model file in place:

```powershell
python main.py
```

**Expected console output:**
```
[ENet] Initializing lane detector...
[ENet] Model loaded successfully on cuda    <-- Success!
Reading video from: drive.mp4
```

**Check the debug window:**
- Window: "ENet Lane Mask (Debug)"
- Should show: Continuous white lane lines on black background

---

## ⚠️ Troubleshooting

### Issue 1: Download link doesn't work
**Solution:** Try alternative options (Kaggle → SharePoint → GitHub)

### Issue 2: Model file is corrupted
```powershell
# Check file size (should be 10+ MB)
(Get-Item "d:\MAIN-PROJECT\enet_lane_model.pth").Length / 1MB
```

### Issue 3: Model architecture mismatch
If you get errors like `"RuntimeError: Error(s) in loading state_dict"`:

**Option A:** The model might be wrapped differently. Update the loader:
```python
# In ENetLaneDetector.__init__, try:
self.model = torch.load(model_path, map_location=device)

# If that fails, try:
checkpoint = torch.load(model_path, map_location=device)
self.model = checkpoint['model']  # or checkpoint['state_dict']
```

**Option B:** Try a different pretrained model from the alternatives above.

### Issue 4: Input/output shape mismatch
The model might expect different dimensions. Add debug:
```python
# In ENetLaneDetector.infer(), after inference:
print(f"Input shape: {img.shape}")
print(f"Output shape: {logits.shape}")
```

Common formats:
- Input: `(1, 3, 256, 512)` ✅
- Output: `(1, 2, 256, 512)` for binary segmentation ✅
- Output: `(1, 1, 256, 512)` for single-class ✅

---

## 📊 Model Information

### Expected Model Specs:
- **Framework:** PyTorch
- **Architecture:** ENet (Efficient Neural Network)
- **Task:** Binary lane segmentation
- **Training Dataset:** TuSimple or CULane
- **Input:** RGB image (3, 256, 512)
- **Output:** Lane probability map (1 or 2, 256, 512)
- **File Size:** ~10-50 MB
- **Format:** `.pth` or `.pt`

---

## 🎓 Understanding the Models

### Kaggle Model (Recommended)
- **Pros:** Ready to use, well-documented, tested
- **Cons:** Might be optimized for specific lighting/road types
- **Best for:** General road lanes, highways

### LaneNet Models
- **Pros:** State-of-the-art accuracy, multiple variants
- **Cons:** More complex architecture (instance segmentation)
- **Best for:** Complex scenarios, multiple lanes

---

## 🚀 Next Steps After Download

1. ✅ Run `python main.py`
2. ✅ Check console for `[ENet] Model loaded successfully`
3. ✅ Verify debug window shows clean lane masks
4. ✅ Watch main window - curves should be smooth
5. ✅ If all good - ENet is working! 🎉

---

## 💡 Pro Tips

1. **Keep the original download:** Don't delete the original downloaded file until you confirm it works

2. **Test with short video first:** Use a 10-second clip before processing long videos

3. **Compare outputs:** Run with and without ENet to see the difference:
   - Delete `enet_lane_model.pth` → Traditional detection
   - Add it back → ENet detection

4. **Adjust confidence threshold:** If detection is too sensitive or not sensitive enough, edit `main.py`:
   ```python
   # Line ~64 in ENetLaneDetector._logits_to_mask
   mask = (lane.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
   #                                       ^^^
   # Try 0.3 (more sensitive) or 0.7 (less sensitive)
   ```

---

## 📂 Final Checklist

After downloading, you should have:
```
d:\MAIN-PROJECT\
├── main.py
├── drive.mp4
├── enet_lane_model.pth          <-- This file (10-50 MB)
├── ENET_INTEGRATION_GUIDE.md
├── ENET_SETUP.md
└── DOWNLOAD_ENET_MODEL.md       <-- This guide
```

---

**Ready to download?** Start with **Method 1 (Kaggle)** - it's the easiest! 🚀
