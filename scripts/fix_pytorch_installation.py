#!/usr/bin/env python3
"""
Fix PyTorch Installation Script
This script fixes PyTorch segfault issues by reinstalling PyTorch properly.
"""

import subprocess
import sys
import os
import shutil

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS")
            if result.stdout:
                print(f"Output: {result.stdout[:500]}")
            return True
        else:
            print(f"❌ FAILED (return code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT - Command took too long (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("=" * 70)
    print("PYTORCH SEGFAULT FIX SCRIPT")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Uninstall existing PyTorch installation")
    print("2. Clean up any corrupted files")
    print("3. Reinstall PyTorch from conda-forge (most reliable)")
    print("4. Reinstall sentence-transformers")
    print("5. Verify installation works")
    print("\n⚠️  This will take several minutes...")
    
    # Get confirmation
    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    python_executable = sys.executable
    print(f"\nUsing Python: {python_executable}")
    
    steps_completed = 0
    total_steps = 6
    
    # Step 1: Uninstall PyTorch (pip)
    if run_command(
        f"{python_executable} -m pip uninstall -y torch torchvision torchaudio sentence-transformers",
        f"Step {steps_completed + 1}/{total_steps}: Uninstalling PyTorch (pip)"
    ):
        steps_completed += 1
    
    # Step 2: Uninstall PyTorch (conda)
    if run_command(
        f"conda remove -y pytorch torchvision torchaudio",
        f"Step {steps_completed + 1}/{total_steps}: Uninstalling PyTorch (conda)"
    ):
        steps_completed += 1
    
    # Step 3: Clean pip cache
    if run_command(
        f"{python_executable} -m pip cache purge",
        f"Step {steps_completed + 1}/{total_steps}: Cleaning pip cache"
    ):
        steps_completed += 1
    
    # Step 4: Install PyTorch from conda-forge (CPU-only, most stable)
    print(f"\n{'='*70}")
    print(f"Step {steps_completed + 1}/{total_steps}: Installing PyTorch (conda-forge)")
    print(f"{'='*70}")
    print("⚠️  This will take 2-5 minutes...")
    
    # Try conda-forge first (most reliable on macOS)
    success = run_command(
        "conda install -y -c conda-forge pytorch cpuonly",
        "Installing PyTorch CPU from conda-forge"
    )
    
    if not success:
        # Fallback to pip if conda fails
        print("\n⚠️  Conda install failed, trying pip...")
        success = run_command(
            f"{python_executable} -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "Installing PyTorch CPU from PyPI"
        )
    
    if success:
        steps_completed += 1
    
    # Step 5: Install sentence-transformers
    if run_command(
        f"{python_executable} -m pip install sentence-transformers",
        f"Step {steps_completed + 1}/{total_steps}: Installing sentence-transformers"
    ):
        steps_completed += 1
    
    # Step 6: Verify installation
    print(f"\n{'='*70}")
    print(f"Step {steps_completed + 1}/{total_steps}: Verifying Installation")
    print(f"{'='*70}")
    
    test_script = """
import torch
from sentence_transformers import SentenceTransformer

print("Testing PyTorch...")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch location: {torch.__file__}")

# Test basic operation
x = torch.zeros(1)
print("✅ PyTorch basic operations work")

# Test model loading
print("\\nTesting sentence-transformers...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded successfully")

# Test encoding
emb = model.encode("test sentence")
print(f"✅ Embedding generated: {len(emb)} dimensions")

print("\\n✅✅✅ ALL TESTS PASSED! PyTorch is working correctly!")
""".strip()
    
    if run_command(
        f'{python_executable} -c "{test_script}"',
        "Running verification tests"
    ):
        steps_completed += 1
        print("\n" + "=" * 70)
        print("✅✅✅ PYTORCH INSTALLATION FIXED SUCCESSFULLY!")
        print("=" * 70)
        print("\nEmbeddings should now work without segfaults!")
    else:
        print("\n" + "=" * 70)
        print("⚠️  INSTALLATION COMPLETE BUT VERIFICATION FAILED")
        print("=" * 70)
        print("\nPyTorch was installed but tests failed.")
        print("Try running the test manually to see the exact error.")
    
    print(f"\nCompleted {steps_completed}/{total_steps} steps")

if __name__ == "__main__":
    main()


















