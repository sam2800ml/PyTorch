import torch 
import numpy as np

# Tensor initialization and properties
print("Tensor initialization and properties")
#Create a tensor 
tensor = torch.tensor([2,3])
print(tensor)

#create tensor empty
tensor_empty = torch.empty(3,2)
print(tensor_empty)

#create a tensor with ones ans zeros

tensor_ones = torch.ones(2,3)
print(tensor_ones)

tensor_zeros = torch.zeros(3,2)
print(tensor_zeros)

#creation of a tensor using randomic numbers, the main difference is that one is created with positive numbers and the other use both positive and negative

tensor_rand = torch.rand(3,2)
print(tensor_rand)

tensor_randn = torch.randn(2,3)
print(tensor_randn)

#We can convert a numpy array to torch 

np_array = np.array([[3,4,5],[1,2,3]])
print(np_array)
tensor_from_numpy = torch.from_numpy(np_array)

print(f" tensor from numpy: {tensor_from_numpy}")
print(f"dtype: {tensor_from_numpy.dtype}")
print(f"device: {tensor_from_numpy.device}")
print(f"shape: {tensor_from_numpy.shape}")
print(f"size: {tensor_from_numpy.size()}")
print(f"numel: {tensor_from_numpy.numel()}")
print("-"*100)
print("Device operations and indexing and slicing")
# Device operations and indexing and slicing
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
tensor_device = torch.tensor([2,3]).to(device)
print(tensor_device)


