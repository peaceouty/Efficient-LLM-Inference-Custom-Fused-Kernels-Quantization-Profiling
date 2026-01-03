# Quick Setup Script for Windows
# Run in PowerShell

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Efficient LLM Inference Setup" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Check CUDA
Write-Host "`n[1/4] Checking CUDA installation..." -ForegroundColor Yellow
try {
    $cudaVersion = nvcc --version 2>&1 | Select-String "release" | ForEach-Object { $_ -replace '.*release\s+(\d+\.\d+).*','$1' }
    Write-Host "✓ CUDA found: version $cudaVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ CUDA not found. Please install CUDA Toolkit." -ForegroundColor Red
    Write-Host "  Download: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
}

# Check Python
Write-Host "`n[2/4] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`n[3/4] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Compile CUDA kernels
Write-Host "`n[4/4] Compiling CUDA kernels..." -ForegroundColor Yellow
python setup.py install

Write-Host "`n====================================" -ForegroundColor Cyan
Write-Host "Setup complete! ✓" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Cyan

Write-Host "`nQuick start:" -ForegroundColor Yellow
Write-Host "  1. Test kernels:   python cuda_kernels\__init__.py"
Write-Host "  2. Run benchmark:  python profiling\benchmark.py"
Write-Host "  3. Run inference:  python inference\vllm_quantized.py"

Write-Host "`nFor more info, see README.md" -ForegroundColor Cyan
