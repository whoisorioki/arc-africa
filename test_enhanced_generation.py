#!/usr/bin/env python3
"""
Quick test script for the improved enhanced synthetic generation.
"""

import time
from src.data_pipeline.enhanced_synthetic_generation import EnhancedSyntheticGenerator


def test_improved_generation():
    """Test the improved generation with better success rate."""
    print("Testing improved enhanced synthetic generation...")

    # Create generator with relaxed settings
    generator = EnhancedSyntheticGenerator(
        target_samples=1000,  # Small test
        min_task_complexity=1,
        max_task_complexity=3,  # Lower complexity for better success
        quality_threshold=0.3,  # Very relaxed threshold
        balance_primitives=True,
    )

    # Test single sample generation
    print("Testing single sample generation...")
    start_time = time.time()
    sample = generator.generate_high_quality_sample()
    elapsed = time.time() - start_time

    if sample:
        print(f"✓ Successfully generated sample in {elapsed:.2f}s")
        print(f"  Quality: {sample['quality']:.2f}")
        print(f"  Complexity: {sample['complexity']}")
        print(f"  Primitives: {sample['program']}")
    else:
        print("✗ Failed to generate sample")

    # Test batch generation
    print("\nTesting batch generation (100 samples)...")
    start_time = time.time()
    samples = generator.generate_balanced_dataset(100)
    elapsed = time.time() - start_time

    print(f"Generated {len(samples)} samples in {elapsed:.2f}s")
    print(f"Success rate: {len(samples)/100:.2%}")

    if samples:
        avg_quality = sum(s["quality"] for s in samples) / len(samples)
        print(f"Average quality: {avg_quality:.2f}")

        # Show primitive distribution
        primitive_counts = {}
        for sample in samples:
            for prim in sample["program"]:
                primitive_counts[prim] = primitive_counts.get(prim, 0) + 1

        print("\nTop 5 primitives:")
        for prim, count in sorted(
            primitive_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"  {prim}: {count}")


if __name__ == "__main__":
    test_improved_generation()
