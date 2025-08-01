# TTT Debug Action Plan

## Current Status Analysis

Based on the TTT test results analysis, we have identified critical issues:

- **Success Rate**: 20% (1/5 tasks have predictions)
- **Prediction Diversity**: Only "fill" primitive predicted
- **Empty Predictions**: 80% of tasks return empty arrays

## Immediate Action Items

### 1. Fix Import Issues (Priority: HIGH)

**Problem**: scipy import causing hangs with Python 3.12
**Solution**:

- Pin scipy version in requirements.txt
- Create alternative segmentation without scipy dependency
- Test basic imports

### 2. Debug Neural Guide Loading (Priority: HIGH)

**Problem**: Model may not be loading properly
**Actions**:

- Verify model file integrity
- Check model architecture compatibility
- Test basic model inference

### 3. Investigate Prediction Filtering (Priority: HIGH)

**Problem**: Most predictions are empty arrays
**Actions**:

- Check confidence thresholds
- Verify prediction post-processing
- Debug TTT adaptation process

### 4. Validate Input Processing (Priority: MEDIUM)

**Problem**: Input preprocessing may be failing
**Actions**:

- Test object segmentation independently
- Verify grid preprocessing
- Check feature extraction pipeline

## Detailed Debugging Steps

### Step 1: Fix Dependencies

```bash
# Update requirements.txt with pinned versions
pip install scipy==1.11.4  # Compatible with Python 3.12
```

### Step 2: Test Basic Model Loading

```bash
# Create minimal test script
python -c "
import torch
from src.neural_guide.architecture import create_neural_guide
model = create_neural_guide()
print('Model created successfully')
print(f'Parameters: {sum(p.numel() for p in model.parameters())}')
"
```

### Step 3: Test Model Inference

```bash
# Test basic forward pass
python -c "
import torch
import numpy as np
from src.neural_guide.architecture import create_neural_guide

model = create_neural_guide()
model.eval()

# Create dummy input
input_grid = torch.randint(0, 10, (1, 2, 48, 48))  # batch, pairs, height, width
output_grid = torch.randint(0, 10, (1, 2, 48, 48))

with torch.no_grad():
    predictions = model(input_grid, output_grid)
    print(f'Predictions shape: {predictions.shape}')
    print(f'Predictions: {predictions}')
    print(f'Max probability: {torch.max(predictions)}')
    print(f'Min probability: {torch.min(predictions)}')
"
```

### Step 4: Debug TTT Process

```bash
# Test TTT adaptation independently
python -c "
# Import and test TTT functions
from src.solver.main_solver import NeuroSymbolicSolver
solver = NeuroSymbolicSolver()
print('Solver created')
"
```

### Step 5: Test Input Processing

```bash
# Test segmentation without scipy
python -c "
import numpy as np
# Create simple test grid
test_grid = np.array([[1, 1, 0], [1, 0, 2], [0, 2, 2]])
print('Test grid created')
"
```

## Expected Outcomes

### Success Criteria:

1. **Model Loading**: Neural guide loads without errors
2. **Basic Inference**: Model produces non-zero predictions
3. **TTT Adaptation**: Test-time training completes successfully
4. **Prediction Diversity**: Multiple primitives predicted across tasks

### Success Metrics:

- Success rate > 80% (4/5 tasks have predictions)
- Prediction diversity > 3 unique primitives
- No empty prediction arrays

## Fallback Options

If TTT continues to fail:

1. **Disable TTT**: Use pre-trained model without adaptation
2. **Simplify Model**: Use smaller, more robust architecture
3. **Alternative Approach**: Implement rule-based fallback
4. **Manual Debugging**: Step-by-step debugging with logging

## Next Steps

1. Execute Step 1 (Fix Dependencies)
2. Execute Step 2 (Test Model Loading)
3. Execute Step 3 (Test Inference)
4. Based on results, proceed with remaining steps
5. Document findings and implement fixes

## Timeline

- **Immediate (Today)**: Steps 1-3
- **Short-term (This week)**: Steps 4-5 and fixes
- **Medium-term (Next week)**: Integration testing and optimization
