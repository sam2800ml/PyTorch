import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# loading a preloaded dataset

training_data = datasets.FashionMNIST(
    root="data", # this is where all the datasets are going to be stored
    train=True, # in case that we are loading a part that we are going to use for training we specify that in there
    download=False, # In this case we dont have to download anything because we already did it, in case you havent done it, you have to put True
    transform=ToTensor() # We are goign to be recieving the dataset in format of a tensor 
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

# Becasue we use the datasets of pytorch we can use indexing to be able to load the information so we van visualize them

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8,8))
cols, row = 3,3

for i in range(1, cols * row + 1):
    sample_index = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_index]
    figure.add_subplot(row,cols,i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
