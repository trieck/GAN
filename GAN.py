import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# ======== HYPERPARAMETERS ========
latent_dim = 100
batch_size = 512
epochs = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== DATASET ========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # scale to [-1, 1]
])

fixed_labels = torch.arange(10).to(device)  # shape: [10]

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class NoisyEmbedding(nn.Embedding):
    def forward(self, input):
        emb = super().forward(input)
        if self.training:
            noise = torch.randn_like(emb) * 0.05
            emb = emb + noise
        return emb


# ======== MODELS ========
class Generator(nn.Module):
    def __init__(self, embedding_dim=latent_dim):
        super().__init__()
        self.embedding = NoisyEmbedding(10, embedding_dim)  # 10 classes, embedding_dim size

        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * embedding_dim, 512, 4, 1, 0),  # 1x1 → 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16 → 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=0),  # 32x32 → 28x28
            nn.Tanh()
        )

    def forward(self, labels):
        embedding_dim = self.embedding.embedding_dim
        embedding = self.embedding(labels)  # [B, embedding_dim]
        z = torch.rand_like(embedding)  # [B, embedding_dim]
        x = torch.cat([embedding, z], dim=1)  # [B, 2*embedding_dim]
        x = x.view(-1, 2 * embedding_dim, 1, 1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 28 * 28)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),  # 28 → 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14 → 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 7 → 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 4 → 2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),  # forces spatial output to 1×1
            nn.Flatten(),
            nn.Linear(512, 1),
        )

    def forward(self, imgs, labels):
        embedding = self.embedding(labels).view(labels.size(0), 1, 28, 28)  # reshape to [B, 1, 28, 28]
        x = torch.cat([imgs, embedding], dim=1)  # concatenate along channel dimension
        x = self.net(x)
        return x


# ======== INIT ========
G = Generator().to(device)
D = Discriminator().to(device)

opt_D = optim.Adam(D.parameters(), lr=0.0001)  # slower D
opt_G = optim.Adam(G.parameters(), lr=0.0002)  # faster G

D_loss = torch.tensor(0.0, device=device)
G_loss = torch.tensor(0.0, device=device)

# ======== TRAINING LOOP ========
for epoch in range(epochs):
    for i, (real_imgs, labels) in enumerate(loader):
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)
        batch_size = real_imgs.size(0)

        # === Train Discriminator ===
        fake_imgs = G(labels).detach()

        noise_strength = max(0.1 * (1 - epoch / 50), 0.02)  # decay over time
        real_imgs += noise_strength * torch.randn_like(real_imgs)
        fake_imgs += noise_strength * torch.randn_like(fake_imgs)

        # Use hinge loss
        real_loss = torch.relu(1.0 - D(real_imgs, labels)).mean()
        fake_loss = torch.relu(1.0 + D(fake_imgs, labels)).mean()
        D_loss = real_loss + fake_loss

        opt_D.zero_grad()
        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
        opt_D.step()

        # === Train Generator ===
        gen_imgs = G(labels)
        real_targets = torch.ones(batch_size, 1, device=device)

        # Use hinge loss
        G_loss = -D(gen_imgs, labels).mean()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

    print(f"[Epoch {epoch + 1}/{epochs}] D_loss: {D_loss.item():.4f} G_loss: {G_loss.item():.4f}")

    # === Show sample ===
    with torch.no_grad():
        G.eval()
        samples = G(fixed_labels).cpu()  # shape: [10, 1, 28, 28]
        G.train()

        # Apply magma colormap to each sample
        colored_samples = []
        for img in samples:
            img = img.squeeze(0)  # [28, 28]
            img = (img + 1) / 2  # normalize from [-1, 1] → [0, 1]
            img_np = img.numpy()

            # Apply magma colormap and ignore alpha
            img_magma = cm.magma(img_np)[..., :3]  # [H, W, 3]
            img_tensor = torch.tensor(img_magma).permute(2, 0, 1)  # [3, H, W]
            colored_samples.append(img_tensor)

        # Stack and grid the batch
        colored_batch = torch.stack(colored_samples)  # [10, 3, 28, 28]
        grid = make_grid(colored_batch, nrow=5)  # 5x2 layout

        # Display
        plt.imshow(grid.permute(1, 2, 0))  # CHW → HWC
        plt.axis('off')
        plt.title(f"Conditional Samples (Digits 0–9) at Epoch {epoch + 1}")
        plt.show()
