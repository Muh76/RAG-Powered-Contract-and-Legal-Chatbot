# ✅ FIX APPLIED - Server Restart Required

## Root Cause Identified
PyTorch is broken (missing `libtorch_cpu.dylib`). When the code tries to import PyTorch, it causes a **segfault** that crashes the entire Python process. Segfaults cannot be caught with try/except.

## Fix Applied
✅ **Subprocess-based PyTorch check** - Now uses a subprocess to check PyTorch. If PyTorch crashes, only the subprocess dies, not the main server.

✅ **Error handling** - Server will now return proper HTTP errors instead of crashing.

## ⚠️ ACTION REQUIRED: Restart Server

The server is still running old code. You **MUST restart it**:

```bash
# 1. Stop the current server
# Press Ctrl+C in the terminal where uvicorn is running

# 2. Restart the server
./scripts/start_server.sh
```

## Expected Behavior After Restart

✅ Server starts without crashing  
✅ Requests return HTTP error messages instead of "Empty reply from server"  
✅ Error message explains PyTorch is broken and needs to be fixed

## Test After Restart

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are employment rights?", "top_k": 5}'
```

**Expected result:** You should get an HTTP response (even if it's an error) instead of "Empty reply from server".

## Optional: Disable PyTorch Completely

If you want to completely disable PyTorch to avoid any issues, edit `scripts/start_server.sh` and uncomment:
```bash
export DISABLE_PYTORCH=1
```

## To Fix PyTorch (Optional)

If you want to use sentence-transformers embeddings:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

Or reinstall sentence-transformers:
```bash
pip uninstall sentence-transformers
pip install sentence-transformers
```










