# Imports necesarily to be able to load all the information for the project 

import torch # torch is the main library used
from torch import nn 
import torch.optim.sgd
from torch.utils.data import DataLoader
from torchvision import datasets # When we use the torchvision means that we are going to load a project that uses images
from torchvision.transforms import ToTensor


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataloader:
    print(f"Shape of [ N,C,H,W]: {x.shape}") # N: Number of samples, C: Channels, H: Height, W: Width
    print(f"Shape of y: {y.shape} {y.dtype}") # Shape of the labels, type of the tensor 
    break


device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using {device} device") 

class NeuralNetwork(nn.Module):  # This is the class that is going to be our model 
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.Linear(512,10)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss() # Loss that we are going to use in this model 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # Optimizer that is going to be use setting the learning rate 
scaler = torch.GradScaler()

def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        with torch.autocast(device_type="mps", dtype=torch.bfloat16):    
            pred = model(X)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        #loss.backward()
        scaler.scale(optimizer)
        scaler.update()
        #optimizer.step()
        #optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"Loss: {loss:>7f} [{current:>5f} {size:>5d}]")


def test(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct = 0 , 0 
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1} \n-----------------")
    train(train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    test(test_dataloader, model=model, loss_fn=loss_fn)
print("Done")


# Saving the model 
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")


# Loading the model 

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))


# Testing the model behavior

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: {predicted}, Actual: {actual}")