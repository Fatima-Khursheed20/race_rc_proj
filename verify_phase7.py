#!/usr/bin/env python3
"""
PHASE 7 IMPLEMENTATION VERIFICATION SCRIPT

This script verifies that all Phase 7 deliverables are in place and functional.
Run this to confirm the complete implementation.
"""

import sys
from pathlib import Path
import importlib

def check_files():
    """Verify all required files exist."""
    print("\n" + "="*70)
    print("FILE VERIFICATION")
    print("="*70)
    
    files_to_check = {
        "Source Code": [
            "src/hint_generator.py",
            "src/hint_inference_demo.py",
            "src/model_b_train.py",
        ],
        "Trained Models": [
            "models/model_b/hint_generator/hint_ranker.pkl",
            "models/model_b/hint_generator/hint_vectorizer.pkl",
        ],
        "Documentation": [
            "PHASE_7_README.md",
            "PHASE_7_IMPLEMENTATION_REPORT.md",
            "PHASE_7_FINAL_SUMMARY.md",
            "notebooks/experiments.ipynb",
        ],
        "Utility Scripts": [
            "train_hints_fast.py",
        ],
        "Examples": [
            "models/model_b/hint_generator/inference_examples.json",
        ],
    }
    
    all_ok = True
    for category, files in files_to_check.items():
        print(f"\n{category}:")
        for file in files:
            path = Path(file)
            exists = path.exists()
            status = "✅" if exists else "❌"
            size = f" ({path.stat().st_size:,} bytes)" if exists else ""
            print(f"  {status} {file}{size}")
            if not exists:
                all_ok = False
    
    return all_ok

def check_imports():
    """Verify key modules can be imported."""
    print("\n" + "="*70)
    print("IMPORT VERIFICATION")
    print("="*70)
    
    sys.path.insert(0, str(Path.cwd() / 'src'))
    
    modules_to_check = [
        "hint_generator",
        "preprocessing",
        "model_b_train",
    ]
    
    all_ok = True
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            all_ok = False
    
    return all_ok

def check_models():
    """Verify trained models are loadable."""
    print("\n" + "="*70)
    print("MODEL VERIFICATION")
    print("="*70)
    
    sys.path.insert(0, str(Path.cwd() / 'src'))
    
    try:
        from hint_generator import load_hint_model
        ranker, vectorizer = load_hint_model(Path('models/model_b/hint_generator'))
        print(f"✅ Hint ranker loaded: {type(ranker).__name__}")
        print(f"✅ Vectorizer loaded: {type(vectorizer).__name__}")
        
        # Test prediction
        import numpy as np
        X_test = np.random.randn(1, 8)
        pred = ranker.predict(X_test)
        proba = ranker.predict_proba(X_test)
        print(f"✅ Model prediction works: pred={pred[0]}, proba={proba[0]}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def check_inference():
    """Verify inference pipeline works."""
    print("\n" + "="*70)
    print("INFERENCE VERIFICATION")
    print("="*70)
    
    sys.path.insert(0, str(Path.cwd() / 'src'))
    
    try:
        from hint_generator import generate_hints, load_hint_model
        
        ranker, vectorizer = load_hint_model(Path('models/model_b/hint_generator'))
        
        # Simple test
        article = "The quick brown fox jumps over the lazy dog. The fox is very fast. Many animals cannot catch the fox."
        question = "What is the fox?"
        correct_answer = "quick brown fox"
        
        hints = generate_hints(article, question, correct_answer, ranker, vectorizer)
        
        print(f"✅ Hint 1 (general): {hints['hint_1'][:50]}...")
        print(f"✅ Hint 2 (specific): {hints['hint_2'][:50]}...")
        print(f"✅ Hint 3 (strong): {hints['hint_3'][:50]}...")
        
        # Verify format
        assert isinstance(hints, dict), "Hints should be dict"
        assert 'hint_1' in hints and 'hint_2' in hints and 'hint_3' in hints, "Missing hint keys"
        assert all(isinstance(h, str) for h in hints.values()), "Hints should be strings"
        
        print("✅ Inference pipeline works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def check_datasets():
    """Verify dataset files exist."""
    print("\n" + "="*70)
    print("DATASET VERIFICATION")
    print("="*70)
    
    datasets = [
        "data/raw/train.csv",
        "data/raw/dev.csv",
        "data/raw/test.csv",
    ]
    
    all_ok = True
    for dataset in datasets:
        path = Path(dataset)
        if path.exists():
            lines = sum(1 for _ in open(path))
            print(f"✅ {dataset}: {lines:,} lines")
        else:
            print(f"❌ {dataset}: not found")
            all_ok = False
    
    return all_ok

def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("PHASE 7 IMPLEMENTATION VERIFICATION")
    print("="*70)
    
    results = {
        "Files": check_files(),
        "Imports": check_imports(),
        "Datasets": check_datasets(),
        "Models": check_models(),
        "Inference": check_inference(),
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("Phase 7 implementation is complete and functional.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed.")
        print("See details above for issues to fix.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
