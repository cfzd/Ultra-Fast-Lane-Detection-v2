#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor my_interp_cuda(
    torch::Tensor input, torch::Tensor interp_loc, int direction);
  // direction 0 for horizontal, 1 for vertical


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor run(
    torch::Tensor input, torch::Tensor interp_loc, int direction) {
  CHECK_INPUT(input);
  CHECK_INPUT(interp_loc);
  // direction 0 for horizontal, 1 for vertical
  auto res =  my_interp_cuda(input, interp_loc, direction);
  return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "run my interp");
}
