import torch
import torch.nn as nn


class DiscriminatorClassifier(nn.Module):
    def __init__(self, trained_discriminator):
        super().__init__()
        self.embedding = trained_discriminator.embedding
        self.features = nn.Sequential(*list(trained_discriminator.net.children())[:-2])  # remove Flatten + final Linear
        self.flatten = nn.Flatten()

        # NEW classifier head (512 → 1), trainable
        self.classifier = self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

        # Freeze original Discriminator
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, imgs):
        batch_size = imgs.size(0)
        outputs = []

        for i in range(batch_size):
            img = imgs[i].unsqueeze(0).expand(10, -1, -1, -1)  # repeat image 10×
            labels = torch.arange(10, device=imgs.device)

            emb = self.embedding(labels).view(10, 1, 28, 28)
            x = torch.cat([img, emb], dim=1)
            feat = self.features(x)  # (10, 512, 1, 1)
            feat = self.flatten(feat)  # (10, 512)
            score = self.classifier(feat)[:, 0]  # shape: (10,)
            outputs.append(score)

        return torch.stack(outputs)  # (batch_size, 10)
