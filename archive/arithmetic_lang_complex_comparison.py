import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import time
from decimal import Decimal, ROUND_HALF_UP

# Enhanced Constants
MAX_NUM = 1000
MAX_DECIMAL_PLACES = 2
NOISE_PROBABILITY = 0.05
CUSTOM_VOCAB = "0123456789↑↓→←=[](){}⊕⊖⊗⊘⊙⊚?., -"  # Added hyphen/minus
NATURAL_VOCAB = "0123456789 abcdefghijklmnopqrstuvwxyzIsby()+-*/^%,. ?"  # Added space at end
EMBED_DIM = 64  # Increased for more complex patterns
HIDDEN_DIM = 128
NUM_EPOCHS = 100
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
    
    def add_noise(self, question):
        """Add random noise to the question"""
        if random.random() < NOISE_PROBABILITY:
            pos = random.randint(0, len(question) - 1)
            chars = list(question)
            chars[pos] = random.choice(self.vocab)
            return ''.join(chars)
        return question
    
    def generate_question(self):
        """Generate a complex arithmetic question"""
        # Generate main expression
        result, (custom_expr, natural_expr) = Expression.generate_random_expression(max_depth=3)
        
        # Generate comparison value
        comparison_val = random.uniform(result - MAX_NUM/2, result + MAX_NUM/2)
        
        # Round results to prevent floating point complexity
        result = float(Decimal(str(result)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        comparison_val = float(Decimal(str(comparison_val)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        
        # Create question
        if self.use_natural:
            question = f"Is {natural_expr} greater than {comparison_val}?"
        else:
            question = f"{custom_expr}→{comparison_val}?"
        
        # Add noise
        question = self.add_noise(question)
        
        return question, 1 if result > comparison_val else 0

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

class EnhancedTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Parameter(torch.randn(1, 200, embed_dim))  # Increased max length
        
        # Multiple transformer layers
        self.transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,  # Increased heads
                dim_feedforward=hidden_dim,
                dropout=0.1,  # Added dropout
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        pos = self.position[:, :x.size(1)]
        x = self.embedding(x) + pos
        
        # Apply multiple transformer layers
        for transformer in self.transformers:
            x = transformer(x)
        
        # Enhanced classification head
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x.squeeze(-1))

def train_and_evaluate(use_natural_language):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    lang = ArithmeticLanguage(use_natural_language)
    train_dataset = ArithmeticDataset(lang, size=2000)  # Increased dataset size
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = EnhancedTransformer(lang.vocab_size, EMBED_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
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

class Expression:
    """Handles complex mathematical expressions"""
    @staticmethod
    def generate_random_expression(max_depth=3, current_depth=0):
        if current_depth >= max_depth or random.random() < 0.3:
            num = random.randint(1, MAX_NUM)
            return float(num), (str(num), str(num))  # Ensure we return float
        
        ops = [
            ('+', '⊕', 'plus'),
            ('-', '⊖', 'minus'),
            ('*', '⊗', 'multiplied by'),
            ('/', '⊘', 'divided by'),
            ('%', '⊙', 'modulo'),
            ('^', '⊚', 'to the power of')
        ]
        
        op, custom_op, natural_op = random.choice(ops)
        
        # Generate sub-expressions
        left_val, (left_custom, left_natural) = Expression.generate_random_expression(max_depth, current_depth + 1)
        right_val, (right_custom, right_natural) = Expression.generate_random_expression(max_depth, current_depth + 1)
        
        # Add safety checks and limits
        try:
            if op == '+': 
                result = min(left_val + right_val, MAX_NUM)
            elif op == '-': 
                result = max(left_val - right_val, -MAX_NUM)
            elif op == '*': 
                result = min(left_val * right_val, MAX_NUM)
            elif op == '/': 
                right_val = 1 if abs(right_val) < 0.0001 else right_val
                result = min(left_val / right_val, MAX_NUM)
            elif op == '%':
                right_val = 1 if abs(right_val) < 0.0001 else right_val
                result = left_val % right_val
            elif op == '^':
                right_val = min(right_val, 2)  # Further limit power to avoid huge numbers
                result = min(left_val ** right_val, MAX_NUM)
            
            # Ensure result is within reasonable bounds
            result = max(min(float(result), MAX_NUM), -MAX_NUM)
            
        except:
            # Fallback for any calculation errors
            result = random.randint(1, MAX_NUM)
            return float(result), (str(result), str(result))
        
        # Format expressions
        custom_expr = f"({left_custom}{custom_op}{right_custom})"
        natural_expr = f"({left_natural} {natural_op} {right_natural})"
        
        return float(result), (custom_expr, natural_expr)

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
        