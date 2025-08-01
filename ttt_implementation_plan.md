# TTT Implementation Plan - Aligned with PRD v3.0

## 🎯 **Current Status vs PRD Requirements**

### **What the PRD Says:**

- System has trained model (`neural_guide_best.pth`) ✅
- Primary issue: DSL primitive indexing bugs ❌ (not pattern detection)
- Need TTT implementation (FR5) ❌
- Target: 5%+ success rate ❌

### **What We've Been Doing Wrong:**

- Building new pattern detection instead of fixing existing system
- Ignoring the trained model we already have
- Not implementing TTT as required
- Not focusing on the critical indexing bug

## 🚀 **Corrected Action Plan**

### **Phase 1: Fix Critical Bugs (Week 1)**

#### **1.1 Fix DSL Primitive Indexing Bug (FR1)**

```bash
# Debug the systematic indexing error
python scripts/debug_dsl_primitives.py
```

**Target:** Identify and fix the indexing error that causes every task to fail with:

```
"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"
```

#### **1.2 Fix Neural Guide Integration (FR2)**

```bash
# Test neural guide loading and prediction
python scripts/test_neural_guide_integration.py
```

**Target:** Ensure `neural_guide_best.pth` loads correctly and produces valid predictions for the symbolic search.

#### **1.3 Fix Validation Pipeline (FR4)**

```bash
# Create working validation script
python scripts/validate_system_fixed.py
```

**Target:** Get accurate success/failure metrics to measure progress.

### **Phase 2: Implement TTT (Week 2)**

#### **2.1 Simple TTT Implementation (FR5)**

**TTT Concept:**

- Use the trained model as a starting point
- For each new task, do a few gradient steps on the task's demo pairs
- Use the adapted model to guide the symbolic search
- Keep it computationally feasible (no AWS needed initially)

**Implementation Steps:**

1. **Load Pre-trained Model:**

```python
# Load the existing trained model
model = load_neural_guide("models/neural_guide_best.pth")
```

2. **Test-Time Adaptation:**

```python
# For each new task, adapt the model
def adapt_model_for_task(model, demo_pairs, learning_rate=0.001, steps=5):
    """
    Simple TTT: Few gradient steps on task-specific data
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for step in range(steps):
        for input_grid, output_grid in demo_pairs:
            # Forward pass
            predictions = model(input_grid)
            loss = compute_loss(predictions, output_grid)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

3. **Integration with Symbolic Search:**

```python
# Use adapted model to guide search
def solve_with_ttt(task):
    # Adapt model for this specific task
    adapted_model = adapt_model_for_task(pretrained_model, task.demo_pairs)

    # Use adapted model to predict promising primitives
    promising_primitives = adapted_model.predict_primitives(task.input)

    # Run symbolic search with neural guidance
    solution = beam_search_with_guidance(task, promising_primitives)

    return solution
```

#### **2.2 AWS Consideration**

**When AWS Might Be Needed:**

- If TTT requires more computational power than current hardware
- If we need to scale up the adaptation process
- If we want to experiment with more complex TTT approaches

**Current Approach (No AWS Needed):**

- Simple TTT with few gradient steps (5-10)
- Small learning rate (0.001)
- Focus on efficiency, not brute force

## 📊 **Success Metrics**

### **Primary Target:**

- **>5% success rate** on validation tasks (solve at least 20 of 400 tasks)

### **Secondary Targets:**

- Zero system crashes on valid inputs
- All DSL primitives pass unit tests
- Neural guide produces valid predictions
- TTT improves performance without breaking the system

## 🔧 **Implementation Scripts Needed**

### **1. Debug DSL Primitives**

```python
# scripts/debug_dsl_primitives.py
# - Test each primitive individually
# - Identify indexing errors
# - Fix edge cases
```

### **2. Test Neural Guide Integration**

```python
# scripts/test_neural_guide_integration.py
# - Load neural_guide_best.pth
# - Test prediction format
# - Verify integration with search
```

### **3. Implement TTT**

```python
# scripts/implement_ttt.py
# - Simple test-time training
# - Integration with symbolic search
# - Performance measurement
```

### **4. Validation Pipeline**

```python
# scripts/validate_system_fixed.py
# - Accurate success/failure metrics
# - Detailed error reporting
# - Progress tracking
```

## 🎯 **Next Immediate Actions**

1. **Debug the DSL primitive indexing bug** (highest priority)
2. **Test neural guide integration** with existing model
3. **Implement simple TTT** as specified in FR5
4. **Create working validation pipeline** to measure progress

## 💡 **Key Insights**

- **No retraining needed** - we have a trained model
- **No AWS needed initially** - TTT should be computationally feasible
- **Focus on fixing, not rebuilding** - the system exists, it just has bugs
- **TTT is the key innovation** - adapting the model to each specific task

This approach aligns perfectly with the PRD and should get us to the 5%+ success rate target.
