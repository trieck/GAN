import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Discriminator, DiscriminatorClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = Discriminator()
D.load_state_dict(torch.load('checkpoints/D_epoch_latest.pt', map_location=device))
D.to(device)
D.eval()

# Use your loaded Discriminator (D)
model = DiscriminatorClassifier(D).to(device)

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

trainset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

for epoch in range(10):
    loss = torch.zeros(1, device=device)  # reset loss

    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} loss: {loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/discriminator_classifier.pt")
