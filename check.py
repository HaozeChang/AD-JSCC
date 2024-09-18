# test_torch_cuda.py
import torch

if torch.cuda.is_available():
    print("CUDA is available! :)")
    print("CUDA Version:", torch.version.cuda)
else:
    print("CUDA is not available. :(")

if torch.cuda.is_available():
    tensor = torch.randn(3, 3)
    tensor_cuda = tensor.to('cuda')
    print("Tensor on CUDA:", tensor_cuda)
else:
    print("CUDA is not available. Please check your installation.")
