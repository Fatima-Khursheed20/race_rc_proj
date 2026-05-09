"""Example inference script demonstrating the hint generator on a test case."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from preprocessing import load_raw_splits
from hint_generator import generate_hints, load_hint_model
import json

def main():
    """Generate hints for example test cases."""
    
    # Load model
    try:
        ranker, vectorizer = load_hint_model(Path('models/model_b/hint_generator'))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first: python src/model_b_train.py --train-hints")
        return
    
    # Load test data
    splits = load_raw_splits(Path('data/raw'))
    test_df = splits['test']
    
    results = []
    
    # Generate hints for first 10 examples
    for idx in range(min(10, len(test_df))):
        row = test_df.iloc[idx]
        
        article = str(row['article'])
        question = str(row['question'])
        correct_answer = str(row[row['answer']])
        
        hints = generate_hints(
            article,
            question,
            correct_answer,
            ranker,
            vectorizer
        )
        
        result = {
            'question': question,
            'correct_answer': correct_answer,
            'hints': hints,
        }
        results.append(result)
        
        print(f"\n{'='*70}")
        print(f"Example {idx + 1}")
        print(f"{'='*70}")
        print(f"Question: {question}")
        print(f"Correct Answer: {correct_answer}")
        print(f"\nGenerated Hints:")
        print(f"  Level 1 (General):  {hints['hint_1']}")
        print(f"  Level 2 (Specific): {hints['hint_2']}")
        print(f"  Level 3 (Strong):   {hints['hint_3']}")
    
    # Save results to JSON
    output_file = Path('models/model_b/hint_generator/inference_examples.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to {output_file}")

if __name__ == '__main__':
    main()
