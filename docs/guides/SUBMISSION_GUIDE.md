# ARC Challenge Africa - Submission Guide

## Quick Start

To generate your competition submission:

```bash
# 1. Ensure you have the required files
ls models/neural_guide_persistent.pth  # Trained model
ls data/evaluation/                    # Evaluation tasks
ls SampleSubmission.csv               # Required format

# 2. Run the submission script
python scripts/generate_submission.py

# 3. Check the output
ls submission.csv
```

## What This Solver Does

Our neuro-symbolic solver combines:

1. **Neural Guide**: A trained Transformer model that predicts which DSL primitives are most likely to solve a task
2. **Symbolic Search**: Enhanced beam search that explores program combinations guided by neural predictions
3. **Test-Time Training**: Adaptive fine-tuning on each new task to improve performance

## Key Features

- ✅ **Trained Model**: Uses `models/neural_guide_persistent.pth` (trained on 100K+ synthetic tasks)
- ✅ **Competition Compliant**: Generates exact CSV format required by Zindi
- ✅ **Robust**: Multiple fallback strategies and error handling
- ✅ **Verified**: Only accepts solutions that produce exact matches
- ✅ **Open Source**: No external APIs or proprietary components

## Expected Performance

- **Success Rate**: Varies by task complexity (typically 5-15% on complex ARC tasks)
- **Runtime**: ~2-5 seconds per task
- **Memory**: ~2GB RAM, ~1GB GPU memory (if using CUDA)
- **Output**: Competition-compliant `submission.csv`

## Troubleshooting

### Common Issues

1. **"Model not found"**

   - Ensure `models/neural_guide_persistent.pth` exists
   - Check file permissions

2. **"Evaluation directory not found"**

   - Ensure `data/evaluation/` contains ARC task JSON files
   - Check directory structure

3. **"Sample submission not found"**

   - Ensure `SampleSubmission.csv` is in the root directory
   - Verify file format

4. **CUDA Out of Memory**
   - The solver will automatically fall back to CPU
   - Or modify `device="cpu"` in the script

### Performance Tips

- **GPU Acceleration**: Set `device="cuda"` for faster execution
- **Memory Management**: Reduce `beam_width` or `max_search_depth` if needed
- **Parallel Processing**: The solver processes tasks sequentially for reliability

## File Requirements

Your submission directory must contain:

```
arc-africa/
├── models/
│   └── neural_guide_persistent.pth    # REQUIRED: Trained model
├── scripts/
│   └── generate_submission.py         # REQUIRED: Submission script
├── src/                               # REQUIRED: Source code
├── data/
│   └── evaluation/                    # REQUIRED: Evaluation tasks
├── SampleSubmission.csv               # REQUIRED: Format specification
├── requirements.txt                   # REQUIRED: Dependencies
└── README.md                         # REQUIRED: Documentation
```

## Competition Compliance

Our submission meets all Zindi requirements:

- ✅ **Open Source**: All code and models are open source
- ✅ **No External APIs**: No internet access or external services
- ✅ **Reproducible**: Fixed random seeds and deterministic execution
- ✅ **Documentation**: Comprehensive documentation as required
- ✅ **Format Compliant**: Generates exact CSV format required

## Expected Output

The script will generate `submission.csv` with format:

```csv
ID,Target
task_0000_0_0,0
task_0000_0_1,0
task_0000_0_2,0
task_0000_1_0,0
task_0000_1_1,1
...
```

Where:

- `ID`: Task ID and cell position (e.g., "task_0000_0_1" for task_0000, row 0, column 1)
- `Target`: Individual cell value (0, 1, 2, etc.)

## Validation

The script includes built-in validation:

- ✅ Checks that all required tasks are present
- ✅ Verifies output dimensions match requirements
- ✅ Ensures no missing or extra tasks
- ✅ Reports success rate and timing statistics

## Final Notes

- The solver is designed to be robust and handle edge cases gracefully
- Unsolved tasks will output empty grids (all zeros) rather than crashing
- The neural guide provides intelligent guidance but doesn't guarantee solutions
- Performance varies by task complexity - simple tasks are more likely to be solved

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all required files are present
3. Run `python test_submission_script.py` to test components
4. Check the logs for detailed error messages

---

**Good luck in the competition! 🚀**
