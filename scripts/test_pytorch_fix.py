#!/usr/bin/env python3
"""
Quick test script to verify PyTorch is working correctly
"""

import sys
import subprocess
import time

def test_pytorch():
    """Test PyTorch installation"""
    print("=" * 70)
    print("TESTING PYTORCH INSTALLATION")
    print("=" * 70)
    
    test_code = """
import torch
from sentence_transformers import SentenceTransformer

print("1️⃣  Testing PyTorch import...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   PyTorch file: {torch.__file__}")

print("\\n2️⃣  Testing PyTorch operations...")
x = torch.zeros(3, 3)
y = x + 1
print(f"   ✅ Tensor operations work (shape: {x.shape})")

print("\\n3️⃣  Testing sentence-transformers import...")
print("   Loading model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("   ✅ Model loaded successfully")

print("\\n4️⃣  Testing embedding generation...")
test_text = "This is a test sentence for embedding generation"
embedding = model.encode(test_text, convert_to_tensor=False)
print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
print(f"   First 5 values: {embedding[:5]}")

print("\\n5️⃣  Testing batch encoding...")
test_texts = ["First sentence", "Second sentence", "Third sentence"]
embeddings = model.encode(test_texts, convert_to_tensor=False)
print(f"   ✅ Batch encoding works: {len(embeddings)} embeddings, each {len(embeddings[0])} dims")

print("\\n" + "=" * 70)
print("✅✅✅ ALL TESTS PASSED! PyTorch is working correctly!")
print("=" * 70)
print("\\n✅ No segfaults detected")
print("✅ Embeddings work correctly")
print("✅ Ready for BM25 + Embeddings hybrid search!")
"""
    
    print("\nRunning tests in isolated subprocess...")
    print("(This prevents segfaults from crashing this script)\n")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            timeout=120,  # 2 minute timeout
            text=True
        )
        elapsed = time.time() - start_time
        
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n✅ Tests completed in {elapsed:.2f} seconds")
            print("✅ No errors or segfaults!")
            return True
        else:
            print("\n❌ Tests failed!")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"\nError output:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n❌ Tests timed out (>2 minutes)")
        print("This might indicate a hang or segfault during model loading")
        return False
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pytorch()
    sys.exit(0 if success else 1)





































