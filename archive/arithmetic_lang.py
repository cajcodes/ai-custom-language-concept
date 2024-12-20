import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Constants
MAX_NUM = 100  # Maximum number for our arithmetic operations
VOCAB = "0123456789↑↓→←=[]?"  # Our custom language vocabulary
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_EPOCHS = 50
BATCH_SIZE = 32

class ArithmeticLanguage:
    def __init__(self):
        self.char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
        self.idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}
        self.vocab_size = len(VOCAB)
    
    def encode_question(self, num1, op, num2, comparison, num3):
        """Convert arithmetic question to our custom language"""
        # Map standard operators to our symbols
        op_map = {
            '*': '↑',
            '/': '↓',
            '>': '→',
            '<': '←'
        }
        expr = f"[{num1}{op_map[op]}{num2}]{op_map[comparison]}{num3}?"
        return expr
    
    def generate_question(self):
        """Generate a random arithmetic question"""
        num1 = random.randint(1, MAX_NUM)
        num2 = random.randint(1, MAX_NUM)
        
        # For division, ensure clean division
        if random.random() < 0.5:
            op = '*'
            result = num1 * num2
        else:
            op = '/'
            result = num1
            num1 = result * num2
        
        # Generate comparison
        num3 = random.randint(1, MAX_NUM)
        comparison = '>' if result > num3 else '<'
        
        question = self.encode_question(num1, op, num2, comparison, num3)
        answer = 1 if result > num3 else 0
        
        return question, answer

class ArithmeticDataset(Dataset):
    def __init__(self, lang, size=1000):
        self.lang = lang
        self.size = size
        self.data = [self.lang.generate_question() for _ in range(size)]
        self.max_len = max(len(q) for q, _ in self.data)  # Track max length
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        question, answer = self.data[idx]
        
        # Convert question to indices
        question_indices = [self.lang.char_to_idx[c] for c in question]
        
        # Pad with zeros if necessary
        padding = [0] * (self.max_len - len(question_indices))
        question_indices.extend(padding)
        
        # Convert to tensor
        question_tensor = torch.tensor(question_indices, dtype=torch.long)
        
        return question_tensor, torch.tensor(answer, dtype=torch.float32)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Parameter(torch.randn(1, 50, embed_dim))  # Max length 50
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        # Add positional encoding
        pos = self.position[:, :x.size(1)]
        x = self.embedding(x) + pos
        
        # Apply transformer
        x = self.transformer(x)
        
        # Pool and classify
        x = x.mean(dim=1)
        x = self.fc(x)
        return torch.sigmoid(x.squeeze(-1))

def train():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Initialize language and dataset
    lang = ArithmeticLanguage()
    train_dataset = ArithmeticDataset(lang, size=1000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model, optimizer, and loss
    model = SimpleTransformer(lang.vocab_size, EMBED_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_questions, batch_answers in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_questions)
            loss = criterion(outputs, batch_answers)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_answers).sum().item()
            total += len(batch_answers)
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Accuracy: {accuracy:.2%}")
            print("--------------------")

if __name__ == "__main__":
    train()
    