import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Discriminator, DiscriminatorClassifier

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # scale to [-1, 1]
])


def predict_digit(model, img_tensor):
    """
    img_tensor: shape (1, 28, 28), unbatched, already normalized to [-1, 1]
    returns: predicted label (int), and softmax probabilities (Tensor of shape [10])
    """
    model.eval()
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).to(device)  # (1, 1, 28, 28)
        logits = model(img)  # (1, 10)
        probs = torch.softmax(logits, dim=1)[0]  # (10,)
        pred = torch.argmax(probs).item()
    return pred, probs


device = 'cuda' if torch.cuda.is_available() else 'cpu'

D = Discriminator()
D.load_state_dict(torch.load('checkpoints/D_epoch_latest.pt', map_location=device))
D.to(device)
D.eval()

DC = DiscriminatorClassifier(D).to(device)
DC.load_state_dict(torch.load('checkpoints/discriminator_classifier.pt', map_location=device))
DC.to(device)
DC.eval()

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for img, label in dataloader:
    img, label = img.squeeze().to(device), label.squeeze().to(device)

    pred, probs = predict_digit(DC, img)
    index = torch.argmax(probs, dim=0)
    prob = probs[index].item()

    print(f"True label: {label} -- Predicted: {pred} (prob: {prob:.4f})")
