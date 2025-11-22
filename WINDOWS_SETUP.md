# Windows Setup Guide

This guide will help you set up the Rumor Detection project on a Windows PC.

## Prerequisites

### 1. Python Installation

1. **Download Python 3.8+** (recommended: Python 3.9 or 3.10)
   - Visit: https://www.python.org/downloads/
   - Download the Windows installer (64-bit)
   - **Important**: During installation, check "Add Python to PATH"

2. **Verify Installation**
   ```cmd
   python --version
   pip --version
   ```

### 2. CUDA Setup (Optional but Recommended for GPU)

If you have an NVIDIA GPU and want to use it for faster training:

1. **Check GPU Compatibility**
   - Open Device Manager → Display adapters
   - Verify you have an NVIDIA GPU

2. **Install CUDA Toolkit** (if using GPU)
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Download CUDA 11.8 or 12.1 (check PyTorch compatibility)
   - Follow the installation wizard

3. **Install cuDNN** (if using GPU)
   - Visit: https://developer.nvidia.com/cudnn
   - Download and extract to CUDA installation directory

---

## Step-by-Step Setup

### Step 1: Clone/Download the Project

1. Download or clone the project to your desired location, e.g.:
   ```
   C:\Users\YourName\Projects\RumorDetection_FYP
   ```

2. Open Command Prompt or PowerShell in the project directory:
   ```cmd
   cd C:\Users\YourName\Projects\RumorDetection_FYP
   ```

### Step 2: Create Virtual Environment

**Using Command Prompt:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Using PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error in PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Verify activation:**
You should see `(venv)` at the beginning of your command prompt.

### Step 3: Upgrade pip

```cmd
python -m pip install --upgrade pip
```

### Step 4: Install PyTorch (Choose CPU or GPU)

**For CPU-only (slower, but simpler):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (NVIDIA CUDA 11.8):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (NVIDIA CUDA 12.1):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify PyTorch installation:**
```cmd
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Step 5: Install PyTorch Geometric

**For CPU:**
```cmd
pip install torch-geometric
```

**For GPU (after installing PyTorch with CUDA):**
```cmd
pip install torch-geometric
```

### Step 6: Install Other Dependencies

```cmd
pip install -r requirements.txt
```

### Step 7: Install Additional Dependencies

```cmd
pip install transformers sentence-transformers
```

---

## Configuration

### Step 8: Set Up Environment Variables

1. **Create `.env` file** in the project root directory:
   ```
   RumorDetection_FYP/
   ├── .env          ← Create this file
   ├── config.py
   ├── requirements.txt
   └── ...
   ```

2. **Add the following to `.env`** (edit as needed):

   ```env
   # Azure OpenAI Configuration (if using LLM augmentation)
   AZURE_API_KEY=your_api_key_here
   AZURE_ENDPOINT=https://cuhk-apip.azure-api.net
   AZURE_MODEL=gpt-4o-mini
   API_VERSION=2023-05-15

   # LLM Parameters (optional)
   LLM_MAX_TOKENS=500
   LLM_TEMPERATURE=0.7
   LLM_AUGMENTATION_FACTOR=5
   LLM_BATCH_SIZE=10

   # Cost Control (optional)
   LLM_MAX_SAMPLES=50
   LLM_ENABLE_CACHE=true
   USE_LLM=false

   # Data Directory (optional - only if data is in a different location)
   # RUMOR_DATA_DIR=C:\path\to\your\data
   ```

   **Note**: 
   - If you don't have Azure OpenAI API access, set `USE_LLM=false`
   - The project will work without LLM (uses sentence-transformers instead)
   - You can leave API keys empty if not using LLM features

### Step 9: Verify Data Directory

Ensure your data directory structure exists:
```
data/
├── Twitter/
│   ├── Twitter15/
│   └── Twitter16/
└── Weibo/
```

If your data is in a different location, set the `RUMOR_DATA_DIR` environment variable in `.env`.

---

## Testing the Installation

### Test 1: Verify Imports

```cmd
python -c "import torch; import torch_geometric; import transformers; print('All imports successful!')"
```

### Test 2: Test Individual Components

```cmd
# Test BERT feature extraction
python bert_feature_extractor.py

# Test node selector
python node_selector.py

# Test node augmentor
python node_augmentor.py

# Test feature fusion
python feature_fusion.py

# Test model
python model_seaug.py
```

### Test 3: Run Quick Pipeline Test

```cmd
# Run with small sample to verify everything works
python seaug_pipeline.py --dataset Twitter15 --sample_ratio 0.1
```

---

## Common Issues and Solutions

### Issue 1: "pip is not recognized"
**Solution**: 
- Reinstall Python and make sure to check "Add Python to PATH"
- Or use `python -m pip` instead of `pip`

### Issue 2: "Execution Policy" error in PowerShell
**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 3: PyTorch CUDA not available
**Solutions**:
- Verify CUDA installation: `nvidia-smi` in Command Prompt
- Reinstall PyTorch with correct CUDA version
- If GPU setup is too complex, use CPU version (slower but works)

### Issue 4: "ModuleNotFoundError" after installation
**Solution**:
- Make sure virtual environment is activated (you should see `(venv)`)
- Reinstall the missing package: `pip install package_name`

### Issue 5: Path issues with backslashes
**Solution**:
- Windows uses backslashes (`\`) but Python handles both
- If you see path errors, try using forward slashes (`/`) or raw strings: `r"C:\path\to\file"`

### Issue 6: Long path issues
**Solution**:
- Enable long paths in Windows:
  1. Open Group Policy Editor (`gpedit.msc`)
  2. Navigate to: Computer Configuration → Administrative Templates → System → Filesystem
  3. Enable "Enable Win32 long paths"

### Issue 7: Out of memory errors
**Solution**:
- Reduce batch size in `config.py`: `BATCH_SIZE = 16` or `BATCH_SIZE = 8`
- Use smaller sample ratio: `--sample_ratio 0.1`
- Close other applications

---

## Quick Start Commands

Once everything is set up:

```cmd
# Activate virtual environment
venv\Scripts\activate

# Run experiments
python run_experiments.py --dataset Twitter15 --sample_ratio 0.1

# Run full pipeline
python seaug_pipeline.py --dataset Twitter15 --enable_augmentation
```

---

## Directory Structure After Setup

```
RumorDetection_FYP/
├── venv/                    # Virtual environment (created)
├── .env                     # Environment variables (you create)
├── data/                    # Data directory
│   ├── Twitter/
│   ├── Weibo/
│   └── archive/
├── checkpoints/             # Model checkpoints (auto-created)
├── logs/                    # Training logs (auto-created)
├── *.py                     # Python source files
└── requirements.txt
```

---

## Next Steps

1. ✅ Complete the setup steps above
2. ✅ Run the test commands to verify installation
3. ✅ Review `README.md` for usage instructions
4. ✅ Check `config.py` for configuration options
5. ✅ Start running experiments!

---

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Review the "Common Issues" section above
3. Verify all prerequisites are installed
4. Make sure virtual environment is activated
5. Check that all dependencies are installed: `pip list`

---

**Last Updated**: 2024
**Tested on**: Windows 10, Windows 11

