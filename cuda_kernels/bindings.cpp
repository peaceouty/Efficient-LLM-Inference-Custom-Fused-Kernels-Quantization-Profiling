#include <torch/extension.h>

// Forward declarations
torch::Tensor fused_rmsnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float eps);

torch::Tensor fused_silu_cuda(torch::Tensor input);

torch::Tensor fused_gelu_cuda(torch::Tensor input);

torch::Tensor fused_add_layernorm_cuda(
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm", &fused_rmsnorm_cuda, 
          "Fused RMSNorm (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("eps") = 1e-6);
    
    m.def("fused_silu", &fused_silu_cuda,
          "Fused SiLU activation (CUDA)",
          py::arg("input"));
    
    m.def("fused_gelu", &fused_gelu_cuda,
          "Fused GeLU activation (CUDA)",
          py::arg("input"));
    
    m.def("fused_add_layernorm", &fused_add_layernorm_cuda,
          "Fused Add + LayerNorm (CUDA)",
          py::arg("input1"),
          py::arg("input2"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps") = 1e-5);
}
