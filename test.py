import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Constants
EMBED_DIM = 32  # Reduced embedding dimension
HIDDEN_DIM = 64  # Reduced hidden dimension
NUM_EPOCHS = 10  # Fewer epochs
BATCH_SIZE = 8  # Smaller batch size
VOCAB = "0123456789+-*/=?"  # Simple vocab

class SimpleDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = [self.generate_sample() for _ in range(size)]
        self.vocab = VOCAB
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.max_length = max(len(q) for q, _ in self.data)

    def generate_sample(self):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        result = a + b
        question = f"{a}+{b}=?"
        answer = result
        return question, answer

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        indices = [self.char_to_idx[c] for c in question]
        padded = indices + [0] * (self.max_length - len(indices))
        question_tensor = torch.tensor(padded, dtype=torch.long)
        return question_tensor, torch.tensor(answer, dtype=torch.float32)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4),
            num_layers=1
        )
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pooling
        x = torch.relu(self.fc(x))
        return self.output(x).squeeze(-1)

def train_model():
    dataset = SimpleDataset(size=100)  # Small dataset
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleTransformer(len(dataset.vocab), EMBED_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Mean Squared Error for regression

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for questions, answers in dataloader:
            optimizer.zero_grad()
            outputs = model(questions)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    return model

if __name__ == "__main__":
    train_model()
    