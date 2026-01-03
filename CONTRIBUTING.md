# Contributing to Efficient LLM Inference

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🎯 Areas for Contribution

We welcome contributions in the following areas:

### 1. New CUDA Kernels
- Flash Attention implementation
- Quantized MatMul kernels
- Fused attention layers
- Custom activation functions

### 2. Quantization Methods
- INT8 activation quantization
- GPTQ integration
- Mixed-precision strategies
- Calibration improvements

### 3. Model Support
- Mistral/Mixtral architecture
- Qwen models
- GPT-style models
- Multi-modal models

### 4. Platform Support
- AMD GPUs (ROCm)
- Apple Silicon (Metal)
- CPU optimizations
- ARM processors

### 5. Testing & Documentation
- Additional unit tests
- Benchmark scripts
- Tutorial notebooks
- API documentation

---

## 🔧 Development Setup

### 1. Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Efficient-LLM-Inference.git
cd Efficient-LLM-Inference

# Add upstream remote
git remote add upstream https://github.com/peaceouty/Efficient-LLM-Inference.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Editable install

# Install development tools
pip install pytest black flake8 mypy
```

### 3. Compile CUDA Kernels

```bash
python setup.py develop
```

---

## 📝 Coding Standards

### Python Style

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use Black for formatting
- Type hints for function signatures

```python
def fused_operation(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Brief description.
    
    Args:
        input_tensor: Description
        weight: Description
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor
    """
    pass
```

### CUDA Style

- Use descriptive kernel names
- Add comments for non-obvious optimizations
- Include performance notes

```cuda
// Fused RMSNorm kernel with shared memory reduction
// Performance: ~1.4x faster than PyTorch native
template <typename scalar_t>
__global__ void fused_rmsnorm_kernel(...) {
    // Implementation
}
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_kernels.py::TestCustomKernels::test_rmsnorm_correctness

# Run with coverage
pytest --cov=cuda_kernels tests/
```

### Writing Tests

Every new kernel must have:
1. **Correctness test**: Compare with PyTorch reference
2. **Performance test**: Ensure no regression
3. **Edge case tests**: Empty tensors, large inputs, etc.

Example:
```python
def test_new_kernel_correctness(self):
    """Test kernel output matches reference implementation."""
    input_tensor = torch.randn(32, 4096, device='cuda')
    
    # Custom implementation
    output = my_custom_kernel(input_tensor)
    
    # Reference
    expected = pytorch_reference(input_tensor)
    
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-5)
```

---

## 📊 Benchmarking

When adding new kernels, include benchmarks:

```python
from profiling.benchmark import PerformanceProfiler

profiler = PerformanceProfiler()
results = profiler.profile_kernel(
    my_kernel,
    input_data,
    name="MyKernel",
    num_iterations=1000
)
```

---

## 🔄 Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/my-new-kernel
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation
- Run linters

```bash
# Format code
black cuda_kernels/ profiling/ inference/

# Check style
flake8 cuda_kernels/ profiling/ inference/

# Type checking
mypy cuda_kernels/
```

### 3. Commit

Use clear commit messages:
```bash
git commit -m "Add fused attention kernel with Flash Attention optimization

- Implement tiling strategy for memory efficiency
- Add unit tests for correctness
- Benchmark shows 2.1x speedup over PyTorch
- Update README with usage example"
```

### 4. Push and Create PR

```bash
git push origin feature/my-new-kernel
```

Then create a Pull Request on GitHub with:
- **Title**: Brief description (e.g., "Add Flash Attention kernel")
- **Description**:
  - What does this PR do?
  - Why is it needed?
  - Benchmark results (if applicable)
  - Related issues

### 5. Review Process

- Maintainers will review your code
- Address feedback
- Once approved, it will be merged

---

## 🐛 Bug Reports

When reporting bugs, include:

### Issue Template

```markdown
**Environment:**
- OS: [e.g., Ubuntu 22.04]
- GPU: [e.g., RTX 4070]
- CUDA: [e.g., 12.1]
- PyTorch: [e.g., 2.1.0]

**Bug Description:**
Clear description of the issue.

**Steps to Reproduce:**
1. Run `python ...`
2. See error

**Expected Behavior:**
What should happen.

**Actual Behavior:**
What actually happens.

**Error Message:**
```
[Paste full error traceback]
```

**Additional Context:**
Any other relevant information.
```

---

## 💡 Feature Requests

For new features:
1. Check existing issues first
2. Open a new issue with:
   - Use case description
   - Proposed implementation
   - Expected benefits
3. Discuss with maintainers before implementing

---

## 📄 Documentation

### Updating README

When adding features, update:
- Feature list
- Usage examples
- Benchmark results table

### Adding Technical Details

For complex implementations, add to `docs/TECHNICAL_DETAILS.md`:
- Algorithm explanation
- Performance analysis
- Trade-offs

---

## 🏅 Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Added to AUTHORS file

---

## 📧 Contact

Questions? Reach out via:
- GitHub Issues
- Email: your.email@example.com

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for making this project better! 🚀**
