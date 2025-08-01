# Strategic Action Plan: Fix Critical ARC Solver Issues

**Date:** January 2025  
**Priority:** CRITICAL - System currently achieves 0% success rate

## 🚨 Critical Issues Identified

### 1. **Systematic Indexing Bug** (BLOCKING)

- **Error:** `"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"`
- **Impact:** Every single task fails with this error
- **Root Cause:** DSL primitives have incorrect array indexing
- **Status:** MUST FIX FIRST

### 2. **Over-Engineered Architecture** (COMPLEXITY)

- Too many unused components
- Complex synthetic data pipeline that isn't helping
- Multiple training scripts that don't work together
- **Status:** SIMPLIFY IMMEDIATELY

### 3. **Broken Validation Pipeline** (FEEDBACK)

- Validation scripts exist but don't provide useful feedback
- No clear success/failure metrics
- **Status:** FIX FOR RAPID ITERATION

## 🎯 Immediate Action Plan (Week 1)

### Day 1-2: Debug DSL Primitive Bug

**Command to run:**

```bash
# Activate virtual environment
source .venv/Scripts/activate  # Windows
# OR
source .venv/bin/activate      # Linux/Mac

# Create a minimal test script to debug primitives
python -c "
import numpy as np
from src.dsl.primitives import rotate90, horizontal_mirror, vertical_mirror

# Test with simple grid
grid = np.array([[1, 2], [3, 4]])
print('Original:', grid)

try:
    rotated = rotate90(grid)
    print('Rotated:', rotated)
except Exception as e:
    print('ERROR:', e)
    import traceback
    traceback.print_exc()
"
```

**Expected Outcome:** Identify which primitives are broken and fix them one by one.

### Day 3-4: Fix Neural Guide Integration

**Command to run:**

```bash
# Test neural guide loading and prediction
python -c "
import torch
from src.neural_guide.architecture import create_neural_guide
import numpy as np

# Test model loading
model = create_neural_guide()
model.load_state_dict(torch.load('models/neural_guide_best.pth'))
model.eval()

# Test prediction
input_grid = torch.randint(0, 10, (1, 2, 48, 48))  # batch, num_grids, height, width
output_grid = torch.randint(0, 10, (1, 2, 48, 48))

try:
    with torch.no_grad():
        predictions = model(input_grid, output_grid)
    print('Predictions shape:', predictions.shape)
    print('Predictions:', predictions[0])
except Exception as e:
    print('ERROR:', e)
    import traceback
    traceback.print_exc()
"
```

**Expected Outcome:** Ensure neural guide loads and produces valid predictions.

### Day 5-7: Create Working Validation Pipeline

**Command to run:**

```bash
# Create a simple validation script
python scripts/simple_validate.py --tasks 5 --verbose
```

**Expected Outcome:** Get accurate success/failure metrics for a small subset of tasks.

## 🧹 Cleanup Actions (Week 1)

### Remove Unused Components

**Files to delete:**

```bash
# Remove experimental RvNN system
rm -rf src/generative_models/

# Remove unused analysis modules
rm -rf src/analysis/

# Remove complex synthetic generation
rm src/data_pipeline/enhanced_synthetic_generation.py
rm src/data_pipeline/semantic_quality_filter.py

# Remove unused training scripts
rm scripts/run_rvnn_training.py
rm scripts/test_rvnn_system.py
rm scripts/train_quality_classifier.py
rm scripts/analyze_enhanced_synthetic.py
```

### Keep Essential Components

**Files to preserve:**

- `src/solver/main_solver.py` - Main solver (needs fixing)
- `src/neural_guide/architecture.py` - Neural guide (working)
- `src/dsl/primitives.py` - DSL primitives (needs fixing)
- `src/symbolic_search/search.py` - Beam search (needs testing)
- `src/data_pipeline/segmentation.py` - Segmentation (working)
- `models/neural_guide_best.pth` - Trained model (working)

## 📊 Success Metrics

### Week 1 Targets:

- [ ] Fix DSL primitive indexing bug
- [ ] Neural guide loads and predicts correctly
- [ ] Basic validation pipeline works
- [ ] System solves at least 1 task (0.25% success rate)

### Week 2 Targets:

- [ ] Achieve 1% success rate (4+ tasks solved)
- [ ] Simple TTT implementation working
- [ ] All core primitives tested and working

### Week 3 Targets:

- [ ] Achieve 5% success rate (20+ tasks solved)
- [ ] System ready for competition submission

## 🔧 Technical Strategy

### 1. **Minimal Viable System First**

- Start with only essential primitives: `rotate90`, `horizontal_mirror`, `vertical_mirror`, `replace_color`
- Use simple beam search with minimal parameters
- Focus on getting ANY task solved before optimization

### 2. **Incremental Testing**

- Test each component individually
- Add complexity only after basic functionality works
- Use small validation sets for rapid feedback

### 3. **Fallback Strategy**

- If neural guide integration fails, use pure symbolic search
- If complex primitives fail, use only basic transformations
- If beam search fails, use simple depth-first search

## 🚀 Next Steps

1. **Immediate:** Run the debugging commands above
2. **Today:** Fix the first broken primitive
3. **This Week:** Get the system to solve at least one task
4. **Next Week:** Scale up to 5% success rate

## 📝 Notes

- The current system has all the right components but they're not working together
- Focus on fixing bugs, not adding features
- Use the existing trained model - don't retrain until the system works
- Keep it simple - complexity is the enemy of debugging

---

**Remember:** The goal is to get from 0% to 5% success rate, not to build the perfect system. Fix what's broken first, then optimize.
