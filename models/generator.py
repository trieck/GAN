import torch
import torch.nn as nn


class NoisyEmbedding(nn.Embedding):
    def forward(self, input):
        emb = super().forward(input)
        if self.training:
            noise = torch.randn_like(emb) * 0.05
            emb = emb + noise
        return emb


class Generator(nn.Module):
    def __init__(self, embedding_dim=100):
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
