# Neuro-Symbolic Solver for The ARC Challenge Africa

A state-of-the-art neuro-symbolic system for solving abstract reasoning tasks from the Abstraction and Reasoning Corpus (ARC). This system combines neural intuition with symbolic program search to achieve competitive performance on ARC tasks.

## 🏆 Project Overview

This project implements a complete neuro-symbolic solver that:
- **Perceives** ARC grids using object segmentation and representation
- **Reasons** using a symbolic program search guided by neural predictions
- **Adapts** through test-time training for task-specific optimization

The system is designed to compete in "The ARC Challenge Africa" hosted on the Zindi platform, with the goal of winning the Deep Learning Indaba 2025 sponsorship prize.

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 10GB+ disk space

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd arc-africa

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Generate synthetic training data
python scripts/generate_synthetic_data.py --num_samples 100000 --output_path data/synthetic/synthetic_dataset.json

# Clean the synthetic dataset
python scripts/clean_synthetic_data.py --input_path data/synthetic/synthetic_dataset.json --output_path data/synthetic/synthetic_dataset_cleaned.json
```

### 3. Train Neural Guide

```bash
# Train the neural guide model
python -m src.neural_guide.train --data_path data/synthetic/synthetic_dataset_cleaned.json --output_dir models --batch_size 8
```

### 4. Generate Competition Submission

```bash
# Generate submission for the test set
python scripts/generate_submission.py --test_path data/test.json --output_path submission.csv --model_path models/neural_guide_best.pth
```

## 📁 Project Structure

```
arc-africa/
├── data/                          # Data files
│   ├── synthetic/                 # Synthetic training data
│   ├── training/                  # ARC training tasks
│   └── test/                      # ARC test tasks
├── src/                           # Source code
│   ├── data_pipeline/            # Data processing modules
│   ├── dsl/                      # Domain-specific language
│   ├── neural_guide/             # Neural guide model
│   ├── symbolic_search/          # Symbolic search engine
│   └── solver/                   # Main solver integration
├── scripts/                      # Utility scripts
├── models/                       # Trained model weights
├── tests/                        # Unit tests
└── docs/                         # Documentation
```

## 🔧 Complete Pipeline

### Phase 1: Data Generation and Preparation

1. **Generate Synthetic Data**
   ```bash
   python scripts/generate_synthetic_data.py \
       --num_samples 100000 \
       --output_path data/synthetic/synthetic_dataset.json
   ```

2. **Clean Synthetic Data**
   ```bash
   python scripts/clean_synthetic_data.py \
       --input_path data/synthetic/synthetic_dataset.json \
       --output_path data/synthetic/synthetic_dataset_cleaned.json
   ```

### Phase 2: Model Training

1. **Train Neural Guide**
   ```bash
   python -m src.neural_guide.train \
       --data_path data/synthetic/synthetic_dataset_cleaned.json \
       --output_dir models \
       --batch_size 8 \
       --epochs 50 \
       --lr 1e-4
   ```

2. **Monitor Training**
   - Check `models/neural_guide_best.pth` for the best model
   - Training logs show loss and accuracy metrics

### Phase 3: Competition Submission

1. **Generate Submission**
   ```bash
   python scripts/generate_submission.py \
       --test_path data/test.json \
       --output_path submission.csv \
       --model_path models/neural_guide_best.pth
   ```

2. **Validate Submission**
   - Check that `submission.csv` follows the required format
   - Verify file size and row count are reasonable

## 🧪 Testing and Validation

### Run Unit Tests
```bash
python -m pytest tests/
```

### Test Individual Components
```bash
# Test data pipeline
python -m src.data_pipeline.segmentation

# Test DSL primitives
python -m src.dsl.primitives

# Test neural guide
python test_neural_guide.py

# Test full solver
python test_neurosymbolic_solver.py
```

## 📊 Performance Optimization

### For Limited GPU Memory
```bash
# Reduce batch size
python -m src.neural_guide.train --batch_size 4

# Use CPU training (slower but works)
CUDA_VISIBLE_DEVICES="" python -m src.neural_guide.train
```

### For Colab Training
```bash
# Upload to Colab and run with larger batch size
python -m src.neural_guide.train --batch_size 32
```

## 🏗️ Architecture Details

### Neural Guide
- **Architecture**: Transformer-based model
- **Input**: Object representations of ARC grids
- **Output**: Probability distribution over DSL primitives
- **Training**: Multi-label classification on synthetic data

### Symbolic Search
- **Algorithm**: Beam search with neural guidance
- **DSL**: Based on Michael Hodel's arc-dsl
- **Verification**: Exact match against demonstration outputs

### Test-Time Training
- **Adaptation**: Fine-tune neural guide on task demonstrations
- **Augmentation**: Geometric and color transformations
- **Integration**: Seamless combination with symbolic search

## 📈 Expected Performance

- **Training Time**: 2-4 hours on GPU
- **Inference Time**: 1-5 seconds per task
- **Memory Usage**: 2-4GB GPU memory during training
- **Accuracy**: Competitive with state-of-the-art ARC solvers

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 4`
   - Use CPU training: `CUDA_VISIBLE_DEVICES=""`

2. **Import Errors**
   - Ensure virtual environment is activated
   - Check that all dependencies are installed

3. **Data Loading Errors**
   - Verify file paths are correct
   - Check JSON file format

4. **Training Convergence Issues**
   - Adjust learning rate: `--lr 1e-3`
   - Increase training epochs: `--epochs 100`

### Getting Help

- Check the logs for detailed error messages
- Review the documentation in `docs/`
- Run unit tests to verify component functionality

## 📚 Documentation

- **Product Requirements Document**: [docs/arc-challenge.md](docs/arc-challenge.md)
- **Project Structure Guide**: [docs/init.md](docs/init.md)
- **Code Documentation**: All functions include detailed docstrings

## 🤝 Contributing

This project follows strict documentation and code quality standards. Please refer to [docs/init.md](docs/init.md) for contribution guidelines.

## 📄 License

This project is open source and available under the MIT License.

## 🏆 Competition Information

- **Platform**: Zindi
- **Challenge**: The ARC Challenge Africa
- **Prize**: Deep Learning Indaba 2025 sponsorship
- **Submission Format**: CSV with ID, row, col, value columns

---

**Good luck in the competition! 🚀**
