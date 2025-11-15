# Server Restart Required

## Issue
The server is crashing with "Empty reply from server" due to PyTorch segfault issues.

## Fix Applied
✅ Added PyTorch detection before import
✅ Added error handling to prevent crashes
✅ Added ThreadPoolExecutor for async safety

## Action Required
**You MUST restart the server** for the fixes to take effect:

```bash
# 1. Stop the current server (Ctrl+C in the terminal where it's running)

# 2. Restart the server
./scripts/start_server.sh
```

## Expected Behavior After Restart
- Server should start without crashing
- When you make a request, you should get an error message instead of "Empty reply from server"
- The error will explain that PyTorch is broken and needs to be fixed

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

## Test After Restart
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are employment rights?", "top_k": 5}'
```

You should now get a proper HTTP response (even if it's an error) instead of "Empty reply from server".


