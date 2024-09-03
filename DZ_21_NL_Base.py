import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train_and_validate(num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')


        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy after epoch {epoch+1}: {accuracy:.2f}%')

train_and_validate(num_epochs=10)

"""
Epoch 1/10, Loss: 1.1601689029534659
Validation Accuracy after epoch 1: 85.91%
Epoch 2/10, Loss: 0.4199201690951983
Validation Accuracy after epoch 2: 89.32%
Epoch 3/10, Loss: 0.34261023889978726
Validation Accuracy after epoch 3: 90.42%
Epoch 4/10, Loss: 0.30945832379659016
Validation Accuracy after epoch 4: 91.38%
Epoch 5/10, Loss: 0.28561078465978307
Validation Accuracy after epoch 5: 91.66%
Epoch 6/10, Loss: 0.2663673921128114
Validation Accuracy after epoch 6: 92.31%
Epoch 7/10, Loss: 0.24830781929691634
Validation Accuracy after epoch 7: 92.76%
Epoch 8/10, Loss: 0.23203313066562017
Validation Accuracy after epoch 8: 92.88%
Epoch 9/10, Loss: 0.216768373131752
Validation Accuracy after epoch 9: 93.63%
Epoch 10/10, Loss: 0.20251536360879738
Validation Accuracy after epoch 10: 93.70%

Перший етап: Loss 1.16, точність на валідаційних даних  85.91%
Останній етап: Loss 0.20, точність на валідаційних даних 93.70%

"""
