# 🚀 Kaggle Setup Guide for VisiHealth

## 📋 Table of Contents
1. [Uploading Your Datasets](#1-uploading-your-datasets)
2. [Creating the Kaggle Notebook](#2-creating-the-kaggle-notebook)
3. [Running the Training](#3-running-the-training)
4. [Downloading Your Models](#4-downloading-your-models)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Uploading Your Datasets

### Step 1.1: Prepare Your Folders

You need to upload two things to Kaggle:

**A. Your Project Code Folder**
- Location on your PC: `VISIHEALTH CODE`
- Contains: `models/`, `scripts/`, `data/`, `utils/`, `config.yaml`, etc.

**B. Your SLAKE Dataset Folder**
- Location on your PC: `Slake1.0`
- Contains: `train.json`, `test.json`, `imgs/`, `kg.txt`

### Step 1.2: Upload Project Code to Kaggle

1. Go to **https://www.kaggle.com/datasets**
2. Click **"New Dataset"** button
3. Click **"Upload"** or drag your `VISIHEALTH CODE` folder
4. Settings:
   - **Title:** `visihealth-code`
   - **Subtitle:** VisiHealth Medical VQA Project Code
   - **Privacy:** Private (recommended)
5. Click **"Create"**
6. Wait for upload to complete (1-2 minutes)

### Step 1.3: Upload SLAKE Dataset to Kaggle

1. Go to **https://www.kaggle.com/datasets**
2. Click **"New Dataset"** button
3. Click **"Upload"** or drag your `Slake1.0` folder
4. Settings:
   - **Title:** `slake-dataset`
   - **Subtitle:** SLAKE Medical VQA Dataset
   - **Privacy:** Private (recommended)
5. Click **"Create"**
6. Wait for upload to complete (5-10 minutes for ~642 images)

### Step 1.4: Verify Uploads

After uploading, verify your datasets:

**For visihealth-code:**
```
visihealth-code/
├── models/
│   ├── bert_model.py
│   ├── cnn_model.py
│   └── fusion_model.py
├── scripts/
│   ├── train.py
│   └── demo.py
├── data/
├── utils/
├── config.yaml
└── requirements.txt
```

**For slake-dataset:**
```
slake-dataset/
├── train.json
├── test.json
├── imgs/
│   ├── xmlab0/
│   ├── xmlab1/
│   └── ... (642 folders total)
└── kg.txt
```

---

## 2. Creating the Kaggle Notebook

### Step 2.1: Create New Notebook

1. Go to **https://www.kaggle.com/code**
2. Click **"New Notebook"**
3. You'll see a blank notebook

### Step 2.2: Upload Your Notebook File

**Option A: Import from File**
1. Click **"File"** → **"Import Notebook"**
2. Upload: `VisiHealth_Kaggle_Training.ipynb`
3. Click **"Import"**

**Option B: Copy Cells Manually**
1. Open `VisiHealth_Kaggle_Training.ipynb` in VS Code
2. Copy each cell
3. Paste into Kaggle notebook

### Step 2.3: Add Your Datasets

1. Click **"+ Add Data"** button (right sidebar)
2. Go to **"Your Datasets"** tab
3. Search for `visihealth-code`
4. Click **"Add"**
5. Repeat for `slake-dataset`

You should now see both datasets in the "Input" section.

### Step 2.4: Enable GPU

1. Click **"Settings"** (right sidebar)
2. Under **"Accelerator"**:
   - Select **"GPU T4 x2"** (recommended)
   - Or **"GPU P100"** if available
3. Click **"Save"**

### Step 2.5: Configure Session

1. **Internet:** Turn **ON** (needed for downloading BERT model)
2. **Language:** Python
3. **Environment:** Latest

---

## 3. Running the Training

### Step 3.1: Update Dataset Paths (Important!)

In **Step 4** of the notebook, update these lines to match your dataset names:

```python
# ⚠️ UPDATE THESE PATHS
PROJECT_INPUT = '/kaggle/input/visihealth-code'  # Your code folder
SLAKE_INPUT = '/kaggle/input/slake-dataset'      # Your SLAKE dataset
```

**How to find your exact paths:**
1. Run the cell in Step 4
2. Look at "Available input datasets"
3. Update paths accordingly

Example if your datasets have different names:
```python
# If you named them differently:
PROJECT_INPUT = '/kaggle/input/my-visihealth-project'
SLAKE_INPUT = '/kaggle/input/my-slake-data'
```

### Step 3.2: Run All Cells

**Recommended: Run All**
1. Click **"Run All"** button at the top
2. Go grab coffee ☕ (training takes 2-4 hours)

**Alternative: Run Step by Step**
1. Run cells 1-6 (Setup) - should complete in 5-10 minutes
2. Verify everything is ✅ in Step 5
3. Run cells 7-9 (Training) - takes 2-4 hours
4. Run cells 10-15 (Inference) - takes 10-15 minutes

### Step 3.3: Monitor Progress

**During Training (Step 7):**
- You'll see progress bars for each epoch
- Loss values decreasing = good!
- Accuracy increasing = good!
- Checkpoints saved every 5 epochs

**Check Training Logs (Step 8):**
- Click the TensorBoard cell output
- See real-time graphs of:
  - Training loss
  - Validation accuracy
  - Learning rate

### Step 3.4: What's Happening Behind the Scenes

```
Step 1-5:  Setup & Verification (5-10 min)
Step 6:    Load dataset (2-3 min)
Step 7:    TRAINING (2-4 hours) ⏳
           ├── Epoch 1/50
           ├── Epoch 2/50
           ├── ...
           └── Early stopping or completes
Step 8-9:  Check results (1 min)
Step 10-13: Inference & evaluation (10-15 min)
Step 14-15: Export results (1 min)
```

---

## 4. Downloading Your Models

### Step 4.1: Wait for Session to End

**Option A: Let it run completely**
- Training finishes
- All cells complete
- Session auto-saves

**Option B: Manually stop**
- Click **"Stop Session"** (top right)
- Everything is saved automatically

### Step 4.2: Access Output Tab

1. Look for **"Output"** tab (top right, next to "Data")
2. Click it
3. You'll see all generated files

### Step 4.3: Download Files

You'll see these folders/files:

```
📁 checkpoints/
   ├── best_checkpoint.pth (500-600 MB) ⬅️ MOST IMPORTANT
   ├── checkpoint_epoch_5.pth
   ├── checkpoint_epoch_10.pth
   └── ...

📁 results/
   ├── VisiHealth_Results.json
   └── VisiHealth_Model_Info.json

📁 logs/
   └── (TensorBoard log files)
```

**Download Priority:**
1. **MUST DOWNLOAD:** `checkpoints/best_checkpoint.pth`
2. **Recommended:** `results/VisiHealth_Model_Info.json`
3. **Optional:** Other checkpoints, logs

**How to Download:**
- Click download icon next to each file
- Or click **"Download All"** to get everything as ZIP

### Step 4.4: Extract on Your Computer

```bash
# On your PC
cd "C:\Users\Hamad\Desktop\VISIHEALTH CODE"

# Extract downloaded files
# You should have:
checkpoints/
  └── best_checkpoint.pth

results/
  ├── VisiHealth_Results.json
  └── VisiHealth_Model_Info.json
```

---

## 5. Troubleshooting

### Problem: "Dataset not found"

**Solution:**
```python
# In Step 4, check what datasets are actually loaded
print("Available datasets:")
!ls /kaggle/input/

# Update paths to match what you see
PROJECT_INPUT = '/kaggle/input/YOUR-ACTUAL-DATASET-NAME'
```

### Problem: "CUDA out of memory"

**Solution:**
- Reduce batch size in `config.yaml`:
  ```yaml
  training:
    batch_size: 8  # Change from 16 to 8
  ```
- Or use smaller model in config

### Problem: "Package import error"

**Solution:**
- Make sure Step 2 completed successfully
- Re-run Step 2 installation cell
- Check that you're using Python (not R kernel)

### Problem: "No module named 'models'"

**Solution:**
- Check that Step 4 copied files correctly
- Verify all project files with Step 5
- Make sure dataset paths are correct

### Problem: "Training is too slow"

**Check:**
1. GPU is enabled (Settings → Accelerator → GPU)
2. GPU is being used:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```
3. Using Kaggle's faster GPUs (T4 x2 or P100)

### Problem: "Session disconnected"

**Good News:** Your progress is saved!

**Solution:**
1. Refresh the page
2. Your notebook should auto-recover
3. Training will resume from last checkpoint
4. Just re-run Step 7 with `--resume` flag (already in notebook)

### Problem: "Can't download files"

**Solution:**
- Make sure session has stopped
- Wait 1-2 minutes after stopping
- Refresh the Output tab
- Try downloading individual files instead of "All"

---

## 📊 Expected Results

### Training Time:
- **Kaggle GPU T4:** 2-4 hours for 50 epochs
- **With early stopping:** May finish at 25-35 epochs

### Model Size:
- **best_checkpoint.pth:** ~500-600 MB
- **Other checkpoints:** ~500 MB each

### Accuracy:
- **Target:** 60-75% on SLAKE test set
- **Depends on:** Dataset quality, hyperparameters, training epochs

### Storage Usage:
- **During training:** ~5-10 GB
- **Final download:** ~1-2 GB (with all checkpoints)

---

## 💡 Pro Tips

### Tip 1: Save Kaggle Credits
- Kaggle gives 30 GPU hours/week for free
- Turn off session when not using
- Delete old checkpoints to save space

### Tip 2: Use Version Control
- Kaggle auto-saves notebook versions
- Click "Save Version" before major changes
- Can revert to previous versions

### Tip 3: Monitor Training Remotely
- Keep Kaggle tab open
- Enable browser notifications
- Check TensorBoard graphs periodically

### Tip 4: Experiment Efficiently
- Start with small epochs (10-20) to test
- If working well, increase to 50
- Use early stopping to save time

### Tip 5: Backup Your Checkpoints
- Download checkpoints immediately
- Upload important checkpoints as Kaggle datasets
- This way you can resume training later

---

## 🔄 Resuming Training

If you want to continue training later:

1. **Upload your checkpoint as a dataset:**
   - Go to Kaggle Datasets
   - Upload your `checkpoints/` folder
   - Name it `visihealth-checkpoints`

2. **In new notebook:**
   - Add your checkpoint dataset
   - Copy checkpoint to working directory:
     ```python
     !cp -r /kaggle/input/visihealth-checkpoints/* checkpoints/
     ```
   - Run training with `--resume` flag (already in Step 7)

---

## ✅ Checklist

Before you start:
- [ ] Kaggle account created
- [ ] Project code uploaded as dataset
- [ ] SLAKE dataset uploaded as dataset
- [ ] Notebook imported/created
- [ ] Both datasets added to notebook
- [ ] GPU enabled (T4 x2)
- [ ] Internet enabled

During training:
- [ ] Step 5 shows all ✅ green checks
- [ ] Training started successfully
- [ ] Can see progress bars
- [ ] TensorBoard working (optional)

After training:
- [ ] best_checkpoint.pth created
- [ ] Test accuracy calculated
- [ ] Files downloaded from Output tab
- [ ] Checkpoint saved locally

---

## 🆘 Need Help?

**Common Resources:**
- Kaggle Documentation: https://www.kaggle.com/docs
- Kaggle Forums: https://www.kaggle.com/discussions
- Check notebook comments for tips

**Your Files:**
- Main notebook: `VisiHealth_Kaggle_Training.ipynb`
- This guide: `KAGGLE_SETUP_GUIDE.md`
- Project code: `VISIHEALTH CODE/` folder
- Dataset: `Slake1.0/` folder

---

## 🎯 Quick Start Summary

1. **Upload datasets** to Kaggle (2 datasets)
2. **Create notebook** on Kaggle
3. **Add your datasets** to the notebook
4. **Enable GPU** in settings
5. **Update paths** in Step 4 of notebook
6. **Run all cells** and wait
7. **Download** from Output tab when done
8. **Use model** locally or deploy

**Total Time:** 3-5 hours (mostly automated)

---

Good luck! Your model will be trained and ready to use! 🚀
