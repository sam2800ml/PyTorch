import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Add data augmentation
    transforms.RandomRotation(10),      # Add data augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjusted for grayscale
    transforms.Grayscale()
])

train_dataset = datasets.ImageFolder(root='Phase1_Fundametals/Test_Project/dataset/train', transform=data_transforms)
test_dataset = datasets.ImageFolder(root='Phase1_Fundametals/Test_Project/dataset/test', transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Check class labels
print(train_dataset.classes)  # Output: ['class1', 'class2']
print(train_dataset.class_to_idx)  # Output: {'class1': 0, 'class2': 1}

labels = train_dataset.classes

figure = plt.figure(figsize=(8,8))
cols, row = 3,3

for i in range(1, cols * row + 1):
    sample_index = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_index]
    figure.add_subplot(row,cols,i)
    plt.title(labels[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


for x, y in test_loader:
    print(f"Shape of [ N,C,H,W]: {x.shape}") # N: Number of samples, C: Channels, H: Height, W: Width
    print(f"Shape of y: {y.shape} {y.dtype}") # Shape of the labels, type of the tensor 
    break


device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using {device} device") 


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,3)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.pool(F.relu(self.bn(self.conv1(x)))) # 224 -> 112
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 112 -> 56
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 56 -> 28
        x = x.view(-1, 256 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss() # Loss that we are going to use in this model 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5) # Optimizer that is going to be use setting the learning rate 

def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        #with torch.autocast(device_type="mps", dtype=torch.bfloat16):    
        pred = model(X)
        loss = loss_fn(pred, y)
    
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

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


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1} \n-----------------")
    train(train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    test(test_loader, model=model, loss_fn=loss_fn)
print("Done")


# Saving the model 
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
