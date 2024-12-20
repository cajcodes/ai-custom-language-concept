# Custom Language Arithmetic Model

> ⚠️ **Note**: This project is currently in early development. Features and implementations may change significantly.

## Overview

This project explores the effectiveness of different language representations for mathematical operations using neural networks. It implements and compares two approaches:
- A custom symbolic language using mathematical symbols (⊕, ⊖, ⊗, etc.)
- Natural language descriptions of mathematical operations

## Initial Results

Comparison between Custom and Natural language approaches:

| Metric           | Custom | Natural |
|-----------------|--------|----------|
| Vocabulary Size | 56     | 70       |
| Model Size      | 145217 | 146113   |
| Avg Epoch Time  | 6.546  | 30.422   |
| Epochs to 95%   | 14     | 22       |
| Final Accuracy  | 99.85% | 98.95%   |

Key findings:
- Custom language approach achieves higher accuracy with fewer training epochs
- Significantly faster training time (approximately 4.6x faster per epoch)
- Smaller vocabulary size and slightly more compact model

## Features

Current implementation includes:

- 🧮 Complex arithmetic operations
- 🔄 Memory operations (store/recall)
- 🔀 Conditional logic
- 📐 Geometric operations
- 🔢 Set operations
- ⏱️ Temporal logic
- 🎯 Enhanced transformer-based model
- 🔍 Comparison capabilities

## Technical Details

- Built with PyTorch
- Uses transformer architecture
- Supports both custom symbolic and natural language processing
- Includes noise injection for robust training
- Implements dynamic batch processing

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- (Full requirements file coming soon)

## Current Status

The project is in active development with:

- [x] Basic arithmetic operations
- [x] Custom symbolic language
- [x] Natural language processing
- [x] Training infrastructure
- [x] Initial performance benchmarks
- [ ] Custom language integration with LLM
  - [ ] Chain-of-Thought reasoning using custom symbols
  - [ ] System prompt templates
  - [ ] Fine-tuning pipeline
- [ ] Comprehensive testing suite
- [ ] Performance optimizations
- [ ] Extended documentation
- [ ] Web interface

### Roadmap

The primary goal is to develop a small Language Model that can leverage the custom symbolic language for:
1. More efficient mathematical reasoning through Chain-of-Thought prompting
2. Compact system prompts using symbolic representations
3. Hybrid natural language + symbolic responses

## Getting Started

(Coming soon)

## Contributing

As this project is in early development, we welcome discussions and suggestions. Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Provide feedback on the current implementation

## License

(License information pending)
 -  Until then, all rights reserved.

## Contact

(Contact information pending) 
