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
CUSTOM_VOCAB = "0123456789↑↓→←=[](){}⊕⊖⊗⊘⊙⊚?:,. -SRmax" \
               "∈∉∪∩⊆≡∀∃◇□↣⊲⊳⊥∥∂θδ"  # Removed angle brackets since we're using existing brackets
NATURAL_VOCAB = "0123456789 abcdefghijklmnopqrstuvwxyzIsby()+-*/^%,. ?[]{}maxinufthelsr"  # Added [], {}
EMBED_DIM = 64  # Increased for more complex patterns
HIDDEN_DIM = 128
NUM_EPOCHS = 100
BATCH_SIZE = 32
MAX_MEMORY_SLOTS = 5
MEMORY_OPS = ['store', 'recall']
CONDITIONAL_OPS = ['if', 'max', 'min']

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
        self.expression = Expression()
    
    def add_noise(self, question):
        """Add random noise to the question"""
        if random.random() < NOISE_PROBABILITY:
            pos = random.randint(0, len(question) - 1)
            chars = list(question)
            chars[pos] = random.choice(self.vocab)
            return ''.join(chars)
        return question
    
    def generate_question(self):
        """Generate a complex arithmetic question with memory and conditional operations"""
        # Initialize memory for this question
        memory = [0.0] * MAX_MEMORY_SLOTS
        
        # Generate main expression
        result, (custom_expr, natural_expr), _ = Expression.generate_random_expression(
            max_depth=4,  # Increased depth for more complex expressions
            memory=memory,
            use_natural=self.use_natural
        )
        
        # Generate comparison value
        comparison_val = random.uniform(result - MAX_NUM/2, result + MAX_NUM/2)
        
        # Round results
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
        self.position = nn.Parameter(torch.randn(1, 512, embed_dim))
        
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
    """Handles complex mathematical expressions with memory and conditional operations"""
    def __init__(self):
        self.memory = [0.0] * MAX_MEMORY_SLOTS  # Initialize memory slots

    @staticmethod
    def generate_set():
        """Generate a random set of numbers"""
        size = random.randint(1, 5)
        elements = [random.randint(1, MAX_NUM) for _ in range(size)]
        return Set(elements)
    
    @staticmethod
    def generate_temporal_value():
        """Generate a random temporal value"""
        value = random.random() > 0.5
        timestamp = random.randint(0, 100)
        return TemporalValue(value, timestamp)
    
    @staticmethod
    def generate_point():
        """Generate a random 2D point"""
        x = random.uniform(-MAX_NUM, MAX_NUM)
        y = random.uniform(-MAX_NUM, MAX_NUM)
        return Point(x, y)

    @staticmethod
    def generate_random_expression(max_depth=3, current_depth=0, memory=None, use_natural=False):
        if memory is None:
            memory = [0.0] * MAX_MEMORY_SLOTS

        # Base case for recursion
        if current_depth >= max_depth or random.random() < 0.3:
            num = random.randint(1, MAX_NUM)
            return float(num), (str(num), str(num)), memory

        # Expanded operations list
        ops = [
            # Existing arithmetic ops
            ('+', '⊕', 'plus'),
            ('-', '⊖', 'minus'),
            ('*', '⊗', 'multiplied by'),
            ('/', '⊘', 'divided by'),
            ('%', '⊙', 'modulo'),
            ('^', '⊚', 'to the power of'),
            
            # New logical ops
            ('in', '∈', 'is in'),
            ('not_in', '∉', 'is not in'),
            ('union', '∪', 'union'),
            ('intersect', '∩', 'intersection'),
            ('subset', '⊆', 'subset of'),
            ('equiv', '≡', 'equivalent to'),
            
            # New temporal ops
            ('eventually', '◇', 'eventually'),
            ('always', '□', 'always'),
            ('leads_to', '↣', 'leads to'),
            ('before', '⊲', 'before'),
            ('after', '⊳', 'after'),
            
            # New spatial ops
            ('perp', '⊥', 'perpendicular to'),
            ('parallel', '∥', 'parallel to'),
            ('distance', 'δ', 'distance'),
            ('angle', 'θ', 'angle'),
            ('boundary', '∂', 'boundary')
        ]

        op, custom_op, natural_op = random.choice(ops)

        try:
            # Handle new operator types
            if op in ['in', 'not_in', 'subset']:
                # Generate actual sets instead of simple values
                set1 = Expression.generate_set()
                if op == 'subset':
                    set2 = Expression.generate_set()
                    result = set1.is_subset(set2)
                    custom_expr = f"{set1}{custom_op}{set2}"
                    natural_expr = f"{str(set1) if not use_natural else repr(set1)} {natural_op} {str(set2) if not use_natural else repr(set2)}"
                else:
                    elem = random.randint(1, MAX_NUM)
                    result = (elem in set1) if op == 'in' else (elem not in set1)
                    custom_expr = f"{elem}{custom_op}{set1}"
                    natural_expr = f"{elem} {natural_op} {str(set1) if not use_natural else repr(set1)}"
                return float(result), (custom_expr, natural_expr), memory

            elif op in ['eventually', 'always', 'before', 'after']:
                # Generate temporal values with actual timestamps
                temporal1 = Expression.generate_temporal_value()
                if op in ['before', 'after']:
                    temporal2 = Expression.generate_temporal_value()
                    result = temporal1.before(temporal2) if op == 'before' else temporal1.after(temporal2)
                    custom_expr = f"({temporal1}{custom_op}{temporal2})"
                    natural_expr = f"({temporal1} {natural_op} {temporal2})"
                else:
                    time_window = random.randint(1, 10)
                    result = temporal1.eventually(time_window) if op == 'eventually' else temporal1.always(time_window)
                    custom_expr = f"{custom_op}[{time_window}]({temporal1})"
                    natural_expr = f"{natural_op} within {time_window} units ({temporal1})"
                return float(result), (custom_expr, natural_expr), memory

            elif op in ['distance', 'angle', 'perp', 'parallel']:
                # Generate actual geometric points and relationships
                point1 = Expression.generate_point()
                point2 = Expression.generate_point()
                
                if op == 'distance':
                    result = point1.distance(point2)
                elif op == 'angle':
                    result = point1.angle(point2)
                elif op == 'perp':
                    result = float(point1.perpendicular(point2))
                else:  # parallel
                    result = float(point1.parallel(point2))
                
                custom_expr = f"{custom_op}{point1}{point2}"
                natural_expr = f"{natural_op} between {str(point1) if not use_natural else repr(point1)} and {str(point2) if not use_natural else repr(point2)}"
                return result, (custom_expr, natural_expr), memory

            # Handle memory operations
            elif op == 'store':
                val, (custom_expr, natural_expr), memory = Expression.generate_random_expression(
                    max_depth, current_depth + 1, memory, use_natural)
                slot = random.randint(0, MAX_MEMORY_SLOTS - 1)
                memory[slot] = val
                return (val,
                        (f"S{slot}({custom_expr})", f"store {natural_expr} in memory {slot}"),
                        memory)

            elif op == 'recall':
                slot = random.randint(0, MAX_MEMORY_SLOTS - 1)
                return (memory[slot],
                        (f"R{slot}", f"recall from memory {slot}"),
                        memory)

            # Handle conditional operations
            elif op == 'if':
                # Generate condition, true_case, and false_case
                cond_val, (cond_custom, cond_natural), memory = Expression.generate_random_expression(
                    max_depth - 1, current_depth + 1, memory, use_natural)
                true_val, (true_custom, true_natural), memory = Expression.generate_random_expression(
                    max_depth - 1, current_depth + 1, memory, use_natural)
                false_val, (false_custom, false_natural), memory = Expression.generate_random_expression(
                    max_depth - 1, current_depth + 1, memory, use_natural)

                result = true_val if cond_val > 0 else false_val
                custom_expr = f"?:({cond_custom},{true_custom},{false_custom})"
                natural_expr = f"if {cond_natural} is positive then {true_natural} else {false_natural}"
                return result, (custom_expr, natural_expr), memory

            elif op in ('max', 'min'):
                # Generate two values and return their max/min
                left_val, (left_custom, left_natural), memory = Expression.generate_random_expression(
                    max_depth, current_depth + 1, memory, use_natural)
                right_val, (right_custom, right_natural), memory = Expression.generate_random_expression(
                    max_depth, current_depth + 1, memory, use_natural)

                result = max(left_val, right_val) if op == 'max' else min(left_val, right_val)
                # Use symbols instead of text for custom mode
                custom_expr = f"[{left_custom},{right_custom}]" if op == 'max' else f"{{{left_custom},{right_custom}}}"
                natural_expr = f"{op}imum of {left_natural} and {right_natural}"
                return result, (custom_expr, natural_expr), memory

            # Handle regular arithmetic operations
            else:
                left_val, (left_custom, left_natural), memory = Expression.generate_random_expression(
                    max_depth, current_depth + 1, memory, use_natural)
                right_val, (right_custom, right_natural), memory = Expression.generate_random_expression(
                    max_depth, current_depth + 1, memory, use_natural)

                # Original arithmetic operations
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
                    right_val = min(right_val, 2)
                    result = min(left_val ** right_val, MAX_NUM)

                result = max(min(float(result), MAX_NUM), -MAX_NUM)
                custom_expr = f"({left_custom}{custom_op}{right_custom})"
                natural_expr = f"({left_natural} {natural_op} {right_natural})"
                return result, (custom_expr, natural_expr), memory

        except:
            # Fallback for any calculation errors
            result = random.randint(1, MAX_NUM)
            return float(result), (str(result), str(result)), memory

class Set:
    def __init__(self, elements=None):
        self.elements = set(elements) if elements else set()
    
    def __str__(self):
        # Use square brackets for both custom and natural
        elements_str = ",".join(str(int(x)) for x in self.elements)
        return f"[{elements_str}]"
    
    def __repr__(self):
        # Use a more descriptive format for natural language
        elements_str = ", ".join(str(int(x)) for x in self.elements)
        return f"set of {elements_str}"
    
    def __contains__(self, item):
        return item in self.elements
    
    def union(self, other):
        return Set(self.elements.union(other.elements))
    
    def intersection(self, other):
        return Set(self.elements.intersection(other.elements))
    
    def is_subset(self, other):
        return self.elements.issubset(other.elements)

class TemporalValue:
    def __init__(self, value, timestamp):
        self.value = value
        self.timestamp = timestamp
    
    def __str__(self):
        # Use parentheses instead of curly braces
        val_str = "1" if self.value else "0"
        return f"({val_str}.{self.timestamp})"  # Using only characters from our vocab
    
    def eventually(self, time_window):
        # Returns true if value becomes true within time_window
        return bool(self.value)  # Simplified implementation
    
    def always(self, time_window):
        # Returns true if value remains true throughout time_window
        return bool(self.value)  # Simplified implementation
    
    def before(self, other):
        return self.timestamp < other.timestamp
    
    def after(self, other):
        return self.timestamp > other.timestamp

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        x_str = f"{self.x:.2f}".rstrip('0').rstrip('.')
        y_str = f"{self.y:.2f}".rstrip('0').rstrip('.')
        return f"[{x_str},{y_str}]"
    
    def __repr__(self):
        x_str = f"{self.x:.2f}".rstrip('0').rstrip('.')
        y_str = f"{self.y:.2f}".rstrip('0').rstrip('.')
        return f"point ({x_str}, {y_str})"
    
    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def angle(self, other):
        return np.arctan2(other.y - self.y, other.x - self.x)
    
    def perpendicular(self, other):
        # Simplified check for perpendicularity
        dot_product = self.x * other.x + self.y * other.y
        return abs(dot_product) < 0.001
    
    def parallel(self, other):
        # Simplified check for parallel vectors
        cross_product = self.x * other.y - self.y * other.x
        return abs(cross_product) < 0.001

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
        