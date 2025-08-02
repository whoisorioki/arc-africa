# ARC Challenge Africa - Neuro-Symbolic Solver

A state-of-the-art neuro-symbolic system for solving the Abstraction and Reasoning Corpus (ARC) challenge, designed specifically for The ARC Challenge Africa competition.

## Project Overview

This project implements a hybrid neuro-symbolic approach that combines:
- **Neural Guide**: A transformer-based model that predicts relevant primitives
- **Symbolic Search**: Beam search algorithm that composes programs from primitives
- **Test-Time Training (TTT)**: Adaptive fine-tuning during inference

## Architecture

### Core Components

1. **Enhanced Neural Guide** (`src/neural_guide/enhanced_architecture.py`)
   - Multi-scale attention mechanism
   - Spatial relation encoder
   - Program composition decoder
   - 39+ primitive support including cropping, resizing, and color operations

2. **Enhanced Neuro-Symbolic Solver** (`src/solver/enhanced_solver.py`)
   - Combines neural predictions with symbolic search
   - Supports 39+ primitives including missing operations
   - Test-time training integration
   - Beam search with adaptive pruning

3. **Domain-Specific Language** (`src/dsl/primitives.py`)
   - Core primitives: rotate, mirror, fill, colorfilter, replace_color
   - Enhanced primitives: crop, resize, remove_color, copy, identity
   - Extensible primitive system

## Project Structure

```
arc-africa/
├── src/                          # Core source code
│   ├── neural_guide/             # Neural guide implementation
│   ├── solver/                   # Solver implementations
│   ├── dsl/                      # Domain-specific language
│   ├── data_pipeline/            # Data processing
│   ├── symbolic_search/          # Symbolic search algorithms
│   └── external/                 # External libraries
├── scripts/                      # Production scripts
│   ├── generate_submission.py    # Main submission generator
│   ├── aws/                      # AWS deployment scripts
│   ├── training/                 # Training scripts
│   └── validation/               # Validation scripts
├── config/                       # Configuration files
│   ├── config.json              # Main configuration
│   ├── requirements.txt         # Dependencies
│   └── variable_description.csv # Variable descriptions
├── docs/                         # Documentation
│   ├── guides/                  # User guides
│   └── reports/                 # Project reports
├── outputs/                      # Output files
│   ├── submissions/             # Generated submissions
│   ├── results/                 # Training results
│   └── logs/                    # Training logs
├── data/                         # ARC datasets
├── models/                       # Trained models
├── notebooks/                    # Jupyter notebooks
└── tests/                        # Unit tests
```

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r config/requirements.txt

# Activate virtual environment (if using)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Basic Usage

```bash
# Generate submission for evaluation tasks
python scripts/generate_submission.py

# Test on specific task
python src/solver/enhanced_solver.py --task_path data/training/train_0000.json
```

### AWS Training

```bash
# Setup AWS credentials
python scripts/aws/setup_aws_credentials.py

# Launch training on AWS
python scripts/aws/setup_g4dn_spot.py
```

## Configuration

The system is configured via `config/config.json`:

```json
{
  "model_path": "models/enhanced_neural_guide_best.pth",
  "top_k_primitives": 15,
  "max_search_depth": 8,
  "beam_width": 30,
  "use_ttt": true,
  "use_enhanced_search": true
}
```

## Performance

The enhanced solver with 39+ primitives achieves:
- **Better coverage** of ARC task types including shape-changing operations
- **Improved accuracy** through test-time training
- **Robust performance** across diverse task patterns

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_solver.py
python -m pytest tests/test_neural_guide.py
```

## Documentation

- [Training Guide](docs/guides/TRAINING_GUIDE.md) - How to train the neural guide
- [Submission Guide](docs/guides/SUBMISSION_GUIDE.md) - How to generate submissions
- [AWS Deployment Guide](docs/guides/AWS_DEPLOYMENT_GUIDE.md) - AWS training setup
- [Solver Status Report](docs/reports/SOLVER_STATUS_REPORT.md) - Current solver performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ARC Challenge organizers for the dataset
- The ARC research community for inspiration
- PyTorch and Hugging Face for the neural architecture foundation
