#!/bin/bash
# Quick setup script for the project

echo "===================================="
echo "Efficient LLM Inference Setup"
echo "===================================="

# Check CUDA availability
echo -e "\n[1/4] Checking CUDA installation..."
if command -v nvcc &> /dev/null
then
    CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
    echo "✓ CUDA found: version $CUDA_VERSION"
else
    echo "✗ CUDA not found. Please install CUDA Toolkit."
    echo "  Download: https://developer.nvidia.com/cuda-downloads"
fi

# Check Python
echo -e "\n[2/4] Checking Python..."
if command -v python &> /dev/null
then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "✓ Python found: version $PYTHON_VERSION"
else
    echo "✗ Python not found. Please install Python 3.10+"
    exit 1
fi

# Install dependencies
echo -e "\n[3/4] Installing Python dependencies..."
pip install -r requirements.txt

# Compile CUDA kernels
echo -e "\n[4/4] Compiling CUDA kernels..."
python setup.py install

echo -e "\n===================================="
echo "Setup complete! ✓"
echo "===================================="

echo -e "\nQuick start:"
echo "  1. Test kernels:   python cuda_kernels/__init__.py"
echo "  2. Run benchmark:  python profiling/benchmark.py"
echo "  3. Run inference:  python inference/vllm_quantized.py"

echo -e "\nFor more info, see README.md"
