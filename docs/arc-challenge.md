# Product Requirements Document v3.0: Critical Fixes for ARC Challenge Solver

**Version:** 3.0  
**Date:** January 2025  
**Status:** Critical Fixes Required

## 1. Introduction

### 1.1. Project Status - CRITICAL

The current implementation has a **fundamental bug** that prevents the solver from working at all. Despite having:
- A trained neural guide model (`neural_guide_best.pth`)
- A sophisticated beam search algorithm
- A complete DSL implementation
- Data augmentation capabilities

**The system achieves 0% success rate** due to a systematic indexing error in the DSL primitives that causes every task to fail with the same error: `"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"`.

### 1.2. Revised Problem Statement

The primary challenge is **NOT** to add new features, but to **fix the existing system** so it can actually solve ARC tasks. The current implementation is over-engineered with many components that don't work together. We need to:

1. **Fix the critical indexing bug** in DSL primitives
2. **Simplify the architecture** to focus on what works
3. **Achieve a baseline working system** before adding enhancements
4. **Target 5%+ success rate** on validation tasks

### 1.3. Vision

Our vision is to create a **simple, working neuro-symbolic solver** that can actually solve ARC tasks. We will start with a minimal viable system and incrementally improve it based on empirical results.

### 1.4. Scope - RADICALLY SIMPLIFIED

#### In Scope (CRITICAL FIXES ONLY):

*   **Fix DSL Primitive Indexing Bug**: Debug and fix the systematic error in primitive implementations
*   **Simplify Neural Guide Integration**: Ensure the trained model actually works with the solver
*   **Basic Beam Search**: Use the existing beam search but ensure it works correctly
*   **Minimal TTT Implementation**: Simple test-time training that actually improves performance
*   **Working Validation Pipeline**: Fix the validation scripts to provide accurate feedback

#### Out of Scope (REMOVE/DEPRIORITIZE):

*   **RvNN and Experimental Models**: Remove `src/generative_models/rvnn.py` and related scripts
*   **Complex Analysis Modules**: Remove unused grammar analysis files
*   **Enhanced Synthetic Generation**: The complex pipeline isn't helping
*   **Multiple Training Variations**: Consolidate to one working training approach
*   **Advanced Primitives**: Focus on fixing basic primitives first

## 2. Goals and Objectives

| Goal | Objective |
| :--- | :--- |
| **Fix Critical System Bug** | Debug and resolve the indexing error that prevents any task from being solved |
| **Achieve Working Baseline** | Get the system to solve at least 5% of validation tasks |
| **Simplify Architecture** | Remove unused components and focus on core functionality |
| **Enable Rapid Iteration** | Create a reliable validation pipeline for quick feedback |

## 3. Functional Requirements

### Phase 1: Critical Bug Fixes (IMMEDIATE)

#### FR1: Fix DSL Primitive Indexing Bug
*   **FR1.1:** Debug the systematic indexing error in all DSL primitives
*   **FR1.2:** Ensure all primitives handle edge cases correctly (empty grids, single pixels, etc.)
*   **FR1.3:** Add comprehensive unit tests for each primitive
*   **FR1.4:** Verify that primitives work with the actual ARC grid formats

#### FR2: Fix Neural Guide Integration
*   **FR2.1:** Ensure the trained model loads correctly and produces valid predictions
*   **FR2.2:** Fix the interface between neural predictions and symbolic search
*   **FR2.3:** Verify that the model's output format matches what the search expects

#### FR3: Simplify and Fix Beam Search
*   **FR3.1:** Use the existing beam search but ensure it handles errors gracefully
*   **FR3.2:** Add proper error handling and logging
*   **FR3.3:** Ensure the search can actually find and return valid solutions

#### FR4: Fix Validation Pipeline
*   **FR4.1:** Create a simple, reliable validation script that works
*   **FR4.2:** Ensure it provides accurate success/failure metrics
*   **FR4.3:** Add detailed error reporting for debugging

### Phase 2: Basic Improvements (After Fixes)

#### FR5: Simple TTT Implementation
*   **FR5.1:** Implement basic test-time training that actually works
*   **FR5.2:** Ensure it improves performance without breaking the system
*   **FR5.3:** Keep it simple and computationally feasible

#### FR6: Core DSL Enhancement
*   **FR6.1:** Add only the most essential primitives that are missing
*   **FR6.2:** Focus on primitives that appear frequently in ARC tasks
*   **FR6.3:** Ensure all additions are thoroughly tested

## 4. Non-Functional Requirements

*   **NFR1: Reliability:** The system must not crash on any valid ARC task
*   **NFR2: Simplicity:** Remove all unused components to reduce complexity
*   **NFR3: Debuggability:** Add comprehensive logging and error reporting
*   **NFR4: Testability:** Every component must have unit tests

## 5. Success Metrics

*   **Primary Metric:** Achieve >5% success rate on validation tasks (solve at least 20 of 400 tasks)
*   **Secondary Metrics:**
    *   Zero system crashes on valid inputs
    *   All DSL primitives pass unit tests
    *   Neural guide produces valid predictions
    *   Validation pipeline provides accurate metrics

## 6. Implementation Plan

### Week 1: Critical Bug Fixes
1. Debug DSL primitive indexing error
2. Fix neural guide integration
3. Create working validation pipeline
4. Achieve baseline working system

### Week 2: Basic Improvements
1. Implement simple TTT
2. Add essential missing primitives
3. Optimize beam search parameters
4. Target 5%+ success rate

### Week 3: Validation and Refinement
1. Comprehensive testing on validation set
2. Performance optimization
3. Documentation and cleanup
4. Prepare for competition submission

## 7. Risk Mitigation

*   **High Risk:** The indexing bug might be deeper than expected
    *   **Mitigation:** Start with minimal primitives and add complexity gradually
*   **Medium Risk:** Neural guide might not be compatible with current architecture
    *   **Mitigation:** Create fallback to pure symbolic search if needed
*   **Low Risk:** Performance might be too slow
    *   **Mitigation:** Optimize only after achieving working baseline 