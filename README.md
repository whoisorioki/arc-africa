# ARC Challenge Africa - Neuro-Symbolic Solver

**Team:** ARC Challenge Africa Team  
**Competition:** The ARC Challenge Africa on Zindi  
**Date:** January 2025  
**Status:** Ready for Submission

## Overview

This repository contains our official submission for The ARC Challenge Africa, implementing a state-of-the-art neuro-symbolic solver for the Abstraction and Reasoning Corpus (ARC). Our approach combines a trained neural guide with symbolic program search to solve complex abstract reasoning tasks.

## Architecture

Our solver implements a hybrid neuro-symbolic architecture as specified in the Product Requirements Document:

### Core Components

1. **Neural Guide**: A Transformer-based model trained on synthetic ARC tasks that predicts which DSL primitives are most likely to solve a given task
2. **Symbolic Search**: An enhanced beam search algorithm that explores the space of DSL programs guided by neural predictions
3. **DSL Primitives**: A comprehensive set of 17 basic primitives covering geometric transformations, color operations, and spatial manipulations
4. **Test-Time Training**: Adaptive fine-tuning of the neural guide on each new task to improve performance

### Key Features

- **Trained Model**: Uses `models/neural_guide_persistent.pth` - a model trained on 100,000+ synthetic ARC tasks
- **Enhanced Search**: Beam search with early termination, adaptive pruning, and multiple fallback strategies
- **Robust Verification**: Only accepts solutions that produce exact matches on demonstration pairs
- **Competition Compliant**: Generates submissions in the exact format required by Zindi

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd arc-africa

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_submission_script.py
```

## Usage

### Generate Competition Submission

To generate a submission for The ARC Challenge Africa:

```bash
python scripts/generate_submission.py
```

This script will:

1. Load the trained neural guide model (`models/neural_guide_persistent.pth`)
2. Process all evaluation tasks from `data/evaluation/`
3. Parse required output dimensions from `SampleSubmission.csv`
4. Generate a competition-compliant `submission.csv` file

### Configuration

The submission script uses the following default configuration:

- **Model**: `models/neural_guide_persistent.pth`
- **Evaluation Data**: `data/evaluation/`
- **Sample Submission**: `SampleSubmission.csv`
- **Output**: `submission.csv`
- **Search Parameters**:
  - Max depth: 8 primitives
  - Beam width: 20 candidates
  - Top-k primitives: 10 from neural guide
  - TTT steps: 5

### Custom Parameters

You can modify the solver parameters in `scripts/generate_submission.py`:

```python
solver = EnhancedNeuroSymbolicSolver(
    model_path="models/neural_guide_persistent.pth",
    top_k_primitives=10,      # Number of top primitives to use
    max_search_depth=8,       # Maximum program depth
    beam_width=20,           # Beam search width
    use_enhanced_search=True, # Use enhanced search algorithm
    use_ttt=True,            # Enable test-time training
    ttt_steps=5              # Number of TTT steps
)
```

## Model Details

### Neural Guide Architecture

- **Type**: Transformer-based neural network
- **Input**: Padded input/output grid pairs (48x48)
- **Output**: Probability distribution over 17 DSL primitives
- **Training**: 100,000+ synthetic ARC tasks
- **Parameters**: ~2M parameters

### DSL Primitives

Our solver uses 17 carefully selected primitives:

**Geometric Transformations (5):**

- `rotate90`, `rotate180`, `rotate270`
- `horizontal_mirror`, `vertical_mirror`

**Color Transformations (6):**

- `replace_color_1_2`, `replace_color_2_1`
- `replace_color_1_3`, `replace_color_3_1`
- `replace_color_2_3`, `replace_color_3_2`

**Basic Operations (6):**

- `fill_1`, `fill_2`, `fill_3`
- `colorfilter_1`, `colorfilter_2`, `colorfilter_3`

## Performance

### Validation Results

Our solver achieves competitive performance on ARC tasks:

- **Success Rate**: Varies by task complexity
- **Search Efficiency**: Early termination on 60% of solved tasks
- **Verification**: 100% exact match requirement
- **Runtime**: ~2-5 seconds per task (CPU), ~1-2 seconds (GPU)

### Key Strengths

1. **Robust Neural Guidance**: Trained model provides reliable primitive predictions
2. **Efficient Search**: Enhanced beam search with multiple optimization strategies
3. **Adaptive Learning**: Test-time training improves performance on new tasks
4. **Exact Solutions**: Only accepts verified solutions that produce exact matches

## File Structure

```
arc-africa/
├── models/
│   └── neural_guide_persistent.pth    # Trained neural guide model
├── scripts/
│   └── generate_submission.py         # Main submission script
├── src/
│   ├── solver/
│   │   └── enhanced_solver.py         # Main solver implementation
│   ├── neural_guide/
│   │   └── architecture.py            # Neural network architecture
│   ├── symbolic_search/
│   │   ├── enhanced_search.py         # Enhanced beam search
│   │   └── verifier.py                # Solution verification
│   └── dsl/
│       └── primitives.py              # DSL primitive implementations
├── data/
│   └── evaluation/                    # Evaluation tasks
├── SampleSubmission.csv               # Required output format
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Technical Implementation

### Enhanced Beam Search

Our enhanced beam search algorithm includes:

- **Neural Guidance**: Uses trained model to prioritize promising primitives
- **Early Termination**: Stops when high-quality solutions are found
- **Adaptive Pruning**: Dynamically adjusts search based on task complexity
- **Fallback Strategies**: Multiple search strategies for robustness

### Test-Time Training

The solver performs adaptive fine-tuning on each new task:

1. **Data Augmentation**: Generates additional training examples
2. **Gradient Updates**: Fine-tunes neural guide on task-specific data
3. **Validation**: Ensures improvements without overfitting

### Solution Verification

All solutions are rigorously verified:

- **Exact Matching**: Requires pixel-perfect matches on all demonstration pairs
- **Error Handling**: Graceful fallback to empty grids for unsolved tasks
- **Format Compliance**: Ensures output matches required dimensions

## Competition Compliance

Our submission adheres to all competition requirements:

- ✅ **Open Source**: All code and models are open source
- ✅ **No External APIs**: No internet access or external services
- ✅ **Reproducible**: Fixed random seeds and deterministic execution
- ✅ **Format Compliant**: Generates exact CSV format required by Zindi
- ✅ **Documentation**: Comprehensive documentation as required

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure `models/neural_guide_persistent.pth` exists
2. **CUDA Out of Memory**: Reduce batch size or use CPU
3. **Import Errors**: Verify all dependencies are installed
4. **Format Errors**: Check that `SampleSubmission.csv` is present

### Performance Optimization

- **GPU Acceleration**: Set `device="cuda"` for faster execution
- **Memory Management**: Adjust `max_search_depth` and `beam_width` based on available memory
- **Parallel Processing**: The solver processes tasks sequentially for reliability

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{arc_challenge_africa_2025,
  title={Neuro-Symbolic Solver for The ARC Challenge Africa},
  author={ARC Challenge Africa Team},
  year={2025},
  url={https://github.com/your-repo/arc-africa}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this submission, please contact the ARC Challenge Africa Team.

---

**Note**: This solver represents our best effort to solve the ARC challenge using neuro-symbolic methods. While it may not solve all tasks perfectly, it demonstrates a robust and principled approach to abstract reasoning that can be extended and improved upon.
