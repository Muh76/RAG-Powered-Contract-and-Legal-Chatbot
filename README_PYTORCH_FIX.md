# PyTorch Segfault Fix Guide

## Problem
PyTorch segfaults occur when the PyTorch installation is corrupted or incompatible, causing the Python process to crash immediately when trying to load embeddings. This prevents BM25 + Embeddings hybrid search from working.

## Root Cause
- Missing or corrupted `libtorch_cpu.dylib` (macOS library file)
- Incompatible PyTorch version with your Python/system
- Mixed installation from conda + pip causing conflicts
- Corrupted installation files

## Solution

### Option 1: Automatic Fix Script (Recommended)

Run the automated fix script:

```bash
cd "/Users/javadbeni/Desktop/Legal Chatbot"
python scripts/fix_pytorch_installation.py
```

This script will:
1. Uninstall existing PyTorch (both pip and conda)
2. Clean caches
3. Reinstall PyTorch from conda-forge (most reliable)
4. Reinstall sentence-transformers
5. Verify everything works

**Time:** ~5-10 minutes

### Option 2: Manual Fix

```bash
# 1. Uninstall existing installations
pip uninstall -y torch torchvision torchaudio sentence-transformers
conda remove -y pytorch torchvision torchaudio

# 2. Clean caches
pip cache purge

# 3. Reinstall from conda-forge (most stable on macOS)
conda install -y -c conda-forge pytorch cpuonly

# 4. Install sentence-transformers
pip install sentence-transformers

# 5. Verify
python scripts/test_pytorch_fix.py
```

### Option 3: Use Pip (Alternative)

If conda doesn't work:

```bash
pip uninstall -y torch torchvision torchaudio sentence-transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

## Verification

After fixing, test PyTorch:

```bash
python scripts/test_pytorch_fix.py
```

You should see:
- ✅ PyTorch version printed
- ✅ Tensor operations work
- ✅ Model loads successfully
- ✅ Embeddings generated
- ✅ No segfaults

## What Changed in Code

The code now:
1. **Checks PyTorch availability first** - Uses `_check_pytorch_available()` to detect issues before loading
2. **Direct initialization** - If PyTorch check passes, initializes embeddings directly (no ThreadPoolExecutor needed)
3. **Proper error handling** - Clear error messages pointing to fix script
4. **Verification** - Tests embedding generation after initialization

## After Fix

Once PyTorch is fixed:
1. ✅ Embeddings will initialize successfully
2. ✅ TRUE HYBRID SEARCH: BM25 + Embeddings will work
3. ✅ Agentic mode will use semantic embeddings
4. ✅ No more segfaults

## Troubleshooting

If fix script fails:
1. Check Python/conda environment is correct
2. Try manual fix steps one at a time
3. Check system Python dependencies: `brew install cmake` (may be needed)
4. If still failing, consider using OpenAI embeddings (API-based, no PyTorch)

## Status

- ✅ Fix script created: `scripts/fix_pytorch_installation.py`
- ✅ Test script created: `scripts/test_pytorch_fix.py`
- ✅ Code updated to use proper PyTorch checks
- ⏳ **ACTION REQUIRED:** Run fix script to resolve installation


















