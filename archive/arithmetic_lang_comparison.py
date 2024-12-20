import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import time

# Constants
MAX_NUM = 100
CUSTOM_VOCAB = "0123456789↑↓→←=[]?"
NATURAL_VOCAB = "0123456789 abcdefghijklmnopqrstuvwxyzIsby"  # For natural language
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_EPOCHS = 50
BATCH_SIZE = 32

def number_to_words(n):
    """Convert a number to words for natural language"""
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    if n < 10:
        return units[n]
    elif n < 20:
        return teens[n-10]
    elif n < 100:
        unit = n % 10
        ten = n // 10
        return tens[ten] + (f" {units[unit]}" if unit > 0 else "")
    return str(n)  # For numbers >= 100, just return the numerical representation

class ArithmeticLanguage:
    def __init__(self, use_natural_language=False):
        self.use_natural = use_natural_language
        self.vocab = NATURAL_VOCAB if use_natural_language else CUSTOM_VOCAB
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    
    def encode_question(self, num1, op, num2, comparison, num3):
        """Convert arithmetic question to either custom or natural language"""
        if self.use_natural:
            op_map = {
                '*': 'multiplied by',
                '/': 'divided by',
                '>': 'greater than',
                '<': 'less than'
            }
            # Convert numbers to words for smaller numbers, keep digits for larger ones
            n1 = number_to_words(num1) if num1 < 100 else str(num1)
            n2 = number_to_words(num2) if num2 < 100 else str(num2)
            n3 = number_to_words(num3) if num3 < 100 else str(num3)
            
            question = f"Is {n1} {op_map[op]} {n2} {op_map[comparison]} {n3}"
            return question
        else:
            op_map = {'*': '↑', '/': '↓', '>': '→', '<': '←'}
            return f"[{num1}{op_map[op]}{num2}]{op_map[comparison]}{num3}?"
    
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
        
        # Find max length for padding
        self.max_length = max(len(q) for q, _ in self.data)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        question, answer = self.data[idx]
        
        # Convert question to tensor of indices and pad
        indices = [self.lang.char_to_idx[c] for c in question]
        padded = indices + [0] * (self.max_length - len(indices))
        question_tensor = torch.tensor(padded, dtype=torch.long)
        
        return question_tensor, torch.tensor(answer, dtype=torch.float32)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Parameter(torch.randn(1, 100, embed_dim))  # Increased max length
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

def train_and_evaluate(use_natural_language):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Initialize language and dataset
    lang = ArithmeticLanguage(use_natural_language)
    train_dataset = ArithmeticDataset(lang, size=1000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model, optimizer, and loss
    model = SimpleTransformer(lang.vocab_size, EMBED_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    # Training metrics
    all_losses = []
    all_accuracies = []
    epoch_times = []
    
    print(f"\nTraining with {'Natural' if use_natural_language else 'Custom'} Language:")
    print(f"Vocabulary size: {lang.vocab_size}")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
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
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        all_losses.append(avg_loss)
        all_accuracies.append(accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print("--------------------")
    
    return {
        'final_accuracy': accuracy,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'epochs_to_95': next((i for i, acc in enumerate(all_accuracies) if acc >= 0.95), -1) + 1,
        'model_size': sum(p.numel() for p in model.parameters()),
        'vocab_size': lang.vocab_size
    }

if __name__ == "__main__":
    # Run experiments with both languages
    custom_results = train_and_evaluate(use_natural_language=False)
    natural_results = train_and_evaluate(use_natural_language=True)
    
    # Print comparison
    print("\nResults Comparison:")
    print(f"{'Metric':<20} {'Custom':<15} {'Natural':<15}")
    print("-" * 50)
    metrics = [
        ('Vocabulary Size', 'vocab_size', 'd'),
        ('Model Size', 'model_size', 'd'),
        ('Avg Epoch Time', 'avg_epoch_time', '.3f'),
        ('Epochs to 95%', 'epochs_to_95', 'd'),
        ('Final Accuracy', 'final_accuracy', '.2%')
    ]
    
    for metric_name, metric_key, format_spec in metrics:
        custom_val = custom_results[metric_key]
        natural_val = natural_results[metric_key]
        format_str = f"{{:<20}} {{:{format_spec}}} {{:{format_spec}}}"
        print(format_str.format(metric_name, custom_val, natural_val))
        