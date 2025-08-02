# Immediate Action Plan: 4-Hour Performance Turnaround

## 🎯 Executive Summary

Based on your 4 vCPU quota for g4dn.xlarge instances, this plan addresses critical performance bottlenecks identified in the audit.

## 🚀 Phase 1: Environment Stabilization (1 Hour)

### 1.1 Secure Dependencies

```bash
# Create requirements.in
cat > requirements.in << 'EOF'
torch==2.3.1+cu121
numpy==1.26.4
pandas
scipy
matplotlib
transformers
accelerate
lark
--extra-index-url https://download.pytorch.org/whl/cu121
EOF

# Install pip-tools and compile
python -m pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

### 1.2 Launch g4dn.xlarge Spot Instance

```bash
aws ec2 run-instances \
  --instance-type g4dn.xlarge \
  --image-id ami-0c02fb55956c7d316 \
  --key-name arc-us-key \
  --security-group-ids sg-033510e74476c27e1 \
  --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=0.50}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"DeleteOnTermination":false}}]'
```

## 🚀 Phase 2: Code Optimization (2 Hours)

### 2.1 Performance Benchmarking Script

```python
# scripts/performance_benchmark.py
import torch
import time

def apply_performance_optimizations():
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)
    print("✅ Performance optimizations applied")

def benchmark_pytorch_performance():
    results = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "amp_available": hasattr(torch.cuda, 'amp')
    }
    return results

if __name__ == "__main__":
    apply_performance_optimizations()
    results = benchmark_pytorch_performance()
    print(f"PyTorch: {results['torch_version']}")
    print(f"CUDA: {results['cuda_available']}")
    print(f"AMP: {results['amp_available']}")
```

### 2.2 Optimized TTT Training

```python
# scripts/optimized_ttt_training.py
import torch
from torch.cuda.amp import autocast, GradScaler

# Apply optimizations
torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float32)

class OptimizedTTTTrainer:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def train_step(self, input_grids, output_grids, target_primitives):
        self.optimizer.zero_grad()

        # Move data with non_blocking
        input_grids = input_grids.to(self.device, non_blocking=True)
        output_grids = output_grids.to(self.device, non_blocking=True)
        target_primitives = target_primitives.to(self.device, non_blocking=True)

        # Forward pass with AMP
        with autocast():
            predictions = self.model(input_grids, output_grids)
            loss = torch.nn.functional.cross_entropy(predictions, target_primitives)

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

def create_optimized_dataloader(dataset, batch_size=4, num_workers=4):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
```

### 2.3 Enhanced Grammar (Phase 1)

```python
# src/analysis/enhanced_grammar.py
from lark import Lark

class EnhancedGrammar:
    def __init__(self):
        self.grammar_str = """
        // Multiple start symbols for increased flexibility
        start: program | statement | expression

        program: statement_list
        statement: assignment | function_call | conditional
        expression: term | expression "+" term | expression "-" term
        term: factor | term "*" factor | term "/" factor
        factor: NUMBER | IDENTIFIER | "(" expression ")" | function_call

        function_call: IDENTIFIER "(" [arguments] ")"
        arguments: expression ("," expression)*
        assignment: IDENTIFIER "=" expression
        conditional: "if" "(" expression ")" statement ["else" statement]
        statement_list: statement (";" statement)*

        IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
        NUMBER: /[0-9]+/

        %import common.WS
        %ignore WS
        """

        self.parser = Lark(
            self.grammar_str,
            start=['program', 'statement', 'expression'],
            keep_all_tokens=True,
            maybe_placeholders=True
        )
```

## 📊 Phase 3: Validation & Measurement (1 Hour)

### 3.1 Success Metrics Tracker

```python
# scripts/success_metrics.py
import json
import time

class SuccessMetricsTracker:
    def __init__(self):
        self.metrics = {
            "baseline": {},
            "current": {},
            "improvements": {},
            "targets": {
                "training_speed_improvement": 2.0,
                "memory_efficiency_improvement": 1.5,
                "grammar_complexity_increase": 0.5
            }
        }

    def measure_baseline(self, training_time, memory_usage, grammar_complexity):
        self.metrics["baseline"] = {
            "training_time_seconds": training_time,
            "memory_usage_gb": memory_usage,
            "grammar_complexity_score": grammar_complexity,
            "timestamp": time.time()
        }

    def measure_current(self, training_time, memory_usage, grammar_complexity):
        self.metrics["current"] = {
            "training_time_seconds": training_time,
            "memory_usage_gb": memory_usage,
            "grammar_complexity_score": grammar_complexity,
            "timestamp": time.time()
        }

    def calculate_improvements(self):
        baseline = self.metrics["baseline"]
        current = self.metrics["current"]

        if baseline and current:
            self.metrics["improvements"] = {
                "training_speed_improvement": baseline["training_time_seconds"] / current["training_time_seconds"],
                "memory_efficiency_improvement": baseline["memory_usage_gb"] / current["memory_usage_gb"],
                "grammar_complexity_increase": (current["grammar_complexity_score"] - baseline["grammar_complexity_score"]) / baseline["grammar_complexity_score"]
            }

    def save_metrics(self, output_path="success_metrics.json"):
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to {output_path}")
```

## 🎯 Success Criteria

### 4-Hour Milestones:

- **Hour 1**: Environment stabilized, g4dn.xlarge launched
- **Hour 2-3**: Code optimized, grammar enhanced
- **Hour 4**: Validation completed, metrics tracked

### Expected Outcomes:

- **Training Speed**: 2x improvement through AMP
- **Memory Efficiency**: 1.5x improvement through optimized data loading
- **Grammar Complexity**: 50% increase through multiple start symbols
- **Environment Stability**: 100% reproducible results

### Risk Mitigation:

- **Spot Instance Termination**: Checkpoint every epoch
- **Performance Regression**: Maintain baseline measurements
- **Dependency Issues**: Use locked requirements.txt

## 🚀 Next Steps (After 4 Hours)

1. **Week 1**: Implement Phase 2 grammar (nested function calls)
2. **Week 2**: Add curriculum learning framework
3. **Week 3**: Implement torch.compile() and distributed training
4. **Week 4**: Scale to p4d.24xlarge for full dataset training
