"""
Input Data Structure:
Dataset from 'xDAN2099/xDAN-Terrible-level-math-collection' with format:
{
    'problem': str,  # math problem
    'answer': str,   # answer
    'source': str,   # source (e.g., cn_k12)
    'domain': list,  # domain classification
    'llama8b_solve_rate': float64  # solve rate
}

Output Data Structure:
List of ChatML entries with format:
{
    'prompt': str,  # ChatML formatted prompt with system, user and assistant tags
    'answer': str,  # answer in latex format
    'level': int  # difficulty level (1-5)
}
"""

import json
from datasets import load_dataset
import random
import argparse

def convert_to_chatml(dataset, sample_ratio=1.0):
    if not 0 < sample_ratio <= 1:
        raise ValueError("Sample ratio must be between 0 and 1")
    
    total_samples = len(dataset)
    num_samples = int(total_samples * sample_ratio)
    sampled_indices = random.sample(range(total_samples), num_samples)
    sampled_dataset = [dataset[i] for i in sorted(sampled_indices)]
    
    # System message template
    system_msg = "You are a helpful assistant good at solving math problems with step-by-step reasoning. You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>."
    
    chatml_data = []
    for item in sampled_dataset:
        problem = item['problem'].strip()
        answer = item['answer'].strip()
        
        # Convert llama8b_solve_rate to difficulty level (1-5)
        solve_rate = item['llama8b_solve_rate']
        if solve_rate <= 0.05:
            level = 5
        elif solve_rate <= 0.15:
            level = 4
        elif solve_rate <= 0.30:
            level = 3
        elif solve_rate <= 0.50:
            level = 2
        else:
            level = 1
        
        # Format the prompt in ChatML format with <|im_start|> and <|im_end|> tags
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
        
        chatml_entry = {
            "prompt": prompt,
            "answer": answer,
            "level": level
        }
        
        chatml_data.append(chatml_entry)
    
    return chatml_data

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to ChatML format with sampling')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                      help='Ratio of data to sample (between 0 and 1)')
    parser.add_argument('--output', type=str, default='mathlv345_8k_chatml.json',
                      help='Output file name')
    args = parser.parse_args()
    
    dataset = load_dataset("xDAN2099/xDAN-Terrible-level-math-collection", split="train")
    chatml_data = convert_to_chatml(dataset, args.sample_ratio)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(chatml_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(chatml_data)} samples to ChatML format")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    random.seed(42)
    main()
