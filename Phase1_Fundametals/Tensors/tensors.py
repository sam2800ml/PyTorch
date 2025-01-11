import torch 
import numpy as np

# we can inicialize a tensro with specific data

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

print(x_data)
print(x_data.type)
print("-"*50)
# we can also transform numpy arrays to tensor

np_array = np.array(data)
torch_tensor = torch.tensor(data)
print("-"*50)
print(np_array)
print(torch_tensor)
print("-"*50)
# if we want to create a tensor just with ones 
x_ones = torch.ones_like(x_data)
print(f"Ones tensor: \n {x_ones}")
print("-"*50)
# we can also create a tensor with random values 
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random tensor: \n {x_rand}")
print("-"*50)
# We can also create a shape, and in this shape we can create the tensors so they can be created base on that tensor 

shape = (2,3)
rand_tensor = torch.rand(shape)
one_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"rand: {rand_tensor}")
print(f"Ones: {one_tensor}")
print(f"zeros: {zeros_tensor}")
print("-"*50)
# We can also create a tensor, and follow all the information, about where is stored and the type

new_tensor = torch.rand(3,4)

print(f"Shape: {new_tensor.shape}")
print(f"Type {new_tensor.dtype}")
print(f"saved {new_tensor.device}")
print("-"*50)
# Operation of tensors 
"""
We can use the tensor to create multiple operations as arithmetic, linear algebra, matrix manipulation, and we can also control where this tensor is kept
that could be on GPU, CPU or MPS

"""

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu" )
print(f"The device on this computer is: {device}")
print("-"*50)

#indexing with tensors 

torch_index = torch.rand(4,4)
print(f"Tensor: {torch_index}")
print(f"First row: {torch_index[0]}")
print(f"Last row: {torch_index[3]}")
print(f"First Column: {torch_index[:,0]}")
print(f"Last Column: {torch_index[:,3]}")
torch_index[:,2] = 1
print(torch_index)
print("-"*50)

#Join differents tensors, to be able to concatenate they have to be the same size

joined_tensor = torch.concat([torch_index,torch_index], dim=1)
print(joined_tensor)
print("-"*50)
#Arithmetic Operations
Test_tensor = [[0,1,2],[3,4,5],[6,7,8]]
torch_test = torch.tensor(Test_tensor)

print(torch_test)
print(torch_test.T)
y1 = torch_test @ torch_test.T # matrix multiplication
print(y1)
y2 = torch_test * torch_test.T # element wise product
print(y2)
print("-"*50)