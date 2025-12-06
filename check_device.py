#!/usr/bin/env python3
"""check_device.py
Check available compute devices and provide recommendations.

Usage:
    python check_device.py
"""
import sys
import platform

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Run: pip install torch")
    sys.exit(1)


def check_devices():
    """Check and display available compute devices."""
    print("="*60)
    print("Device Availability Check")
    print("="*60)

    # System info
    print(f"\nSystem Information:")
    print(f"  Platform: {platform.system()}")
    print(f"  Machine:  {platform.machine()}")
    print(f"  Python:   {platform.python_version()}")
    print(f"  PyTorch:  {torch.__version__}")

    # CPU
    print(f"\n✓ CPU: Available")
    print(f"  Threads: {torch.get_num_threads()}")

    # CUDA
    print(f"\nCUDA:")
    if torch.cuda.is_available():
        print(f"  ✓ Available")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        recommended = "cuda"
    else:
        print(f"  ✗ Not available")
        if platform.system() == "Darwin":
            print(f"  Note: CUDA not supported on macOS")
        else:
            print(f"  Install CUDA-enabled PyTorch:")
            print(f"  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        recommended = "cpu"

    # MPS (Apple Silicon)
    print(f"\nMPS (Apple Silicon):")
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            print(f"  ✓ Available")
            print(f"  Note: MPS support is experimental")
            if recommended == "cpu" and platform.system() == "Darwin":
                recommended = "mps"
        else:
            print(f"  ✗ Not available")
            if platform.system() == "Darwin":
                print(f"  Your macOS version may not support MPS")
    else:
        print(f"  ✗ Not supported (PyTorch version too old)")

    # Recommendations
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)
    print(f"\n✓ Recommended device: {recommended}")

    print(f"\nUsage in scripts:")
    print(f"  python train_small_gpt.py --device {recommended}")
    print(f"  python evaluate.py --device {recommended}")
    print(f"  python measure_performance.py --device {recommended}")

    if recommended == "cpu":
        print(f"\n⚠️  Training on CPU will be significantly slower!")
        print(f"   Consider using:")
        print(f"   - Smaller model (--n-layer 4 --n-embd 256)")
        print(f"   - Smaller batch size (--batch-size 4)")
        print(f"   - Fewer epochs (--epochs 5)")
        print(f"   - Cloud GPU (Google Colab, AWS, etc.)")

    # Performance test
    print("\n" + "="*60)
    print("Quick Performance Test")
    print("="*60)

    for device_name in [recommended]:
        try:
            device = torch.device(device_name)
            x = torch.randn(1000, 1000).to(device)

            import time
            start = time.time()
            for _ in range(100):
                y = torch.matmul(x, x)
            elapsed = time.time() - start

            print(f"\n{device_name.upper()}:")
            print(f"  100 matrix multiplications (1000x1000)")
            print(f"  Time: {elapsed:.3f} seconds")
            print(f"  Performance: {100/elapsed:.1f} ops/sec")

        except Exception as e:
            print(f"\n{device_name.upper()}: Error - {e}")

    print("\n" + "="*60)


if __name__ == '__main__':
    check_devices()
