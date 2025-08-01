#!/usr/bin/env python3
"""
Debug script to analyze TTT test results and identify issues with the neural guide predictions.
"""

import json
import numpy as np
from pathlib import Path

def analyze_ttt_results(results_file: str = "results/ttt_test_results.json"):
    """Analyze the TTT test results to identify patterns and issues.
    
    Args:
        results_file: Path to the TTT test results JSON file.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("=== TTT Test Results Analysis ===\n")
    
    # Basic statistics
    total_tasks = len(results)
    tasks_with_predictions = sum(1 for r in results if any(r['predictions']))
    tasks_without_predictions = total_tasks - tasks_with_predictions
    
    print(f"Total tasks: {total_tasks}")
    print(f"Tasks with predictions: {tasks_with_predictions}")
    print(f"Tasks without predictions: {tasks_without_predictions}")
    print(f"Success rate: {tasks_with_predictions/total_tasks*100:.1f}%\n")
    
    # Analyze prediction patterns
    all_predictions = []
    for result in results:
        for pred_list in result['predictions']:
            all_predictions.extend(pred_list)
    
    if all_predictions:
        unique_predictions = set(all_predictions)
        print(f"Unique primitives predicted: {unique_predictions}")
        print(f"Total predictions made: {len(all_predictions)}")
        
        # Count frequency of each primitive
        from collections import Counter
        pred_counts = Counter(all_predictions)
        print("\nPrediction frequencies:")
        for pred, count in pred_counts.most_common():
            print(f"  {pred}: {count}")
    else:
        print("No predictions were made at all!")
    
    # Analyze task structure
    print(f"\nTask structure analysis:")
    num_pairs_counts = Counter(r['num_pairs'] for r in results)
    for num_pairs, count in num_pairs_counts.items():
        print(f"  Tasks with {num_pairs} pairs: {count}")
    
    # Detailed analysis of each task
    print(f"\n=== Detailed Task Analysis ===")
    for i, result in enumerate(results):
        task_id = result['task_id']
        predictions = result['predictions']
        num_pairs = result['num_pairs']
        
        has_predictions = any(predictions)
        prediction_count = sum(len(pred_list) for pred_list in predictions)
        
        print(f"\nTask {i+1}: {task_id}")
        print(f"  Number of pairs: {num_pairs}")
        print(f"  Has predictions: {has_predictions}")
        print(f"  Total predictions: {prediction_count}")
        print(f"  Prediction lists: {predictions}")
    
    # Identify potential issues
    print(f"\n=== Potential Issues Identified ===")
    
    if tasks_without_predictions > 0:
        print(f"❌ {tasks_without_predictions}/{total_tasks} tasks have no predictions")
        print("   This suggests the neural guide is not providing meaningful guidance")
    
    if len(all_predictions) > 0 and len(set(all_predictions)) == 1:
        print(f"❌ Only one primitive type predicted: {all_predictions[0]}")
        print("   This suggests the model lacks diversity in its predictions")
    
    if len(all_predictions) == 0:
        print("❌ No predictions made at all")
        print("   This could indicate:")
        print("   - Model confidence threshold too high")
        print("   - Input processing issues")
        print("   - Model not properly loaded")
        print("   - TTT adaptation not working")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    if tasks_without_predictions > 0:
        print("1. Check neural guide model loading and initialization")
        print("2. Verify input preprocessing (segmentation, feature extraction)")
        print("3. Lower prediction confidence thresholds")
        print("4. Debug TTT adaptation process")
    
    if len(all_predictions) > 0:
        print("5. Analyze model confidence scores")
        print("6. Check if predictions are being filtered incorrectly")
        print("7. Verify the model was trained on diverse data")

if __name__ == "__main__":
    analyze_ttt_results() 