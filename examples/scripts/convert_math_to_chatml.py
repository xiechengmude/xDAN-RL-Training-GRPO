import json
from datasets import load_dataset
import random
import argparse

def convert_to_chatml(dataset, sample_ratio=1.0):
    # Convert sample_ratio to percentage for better readability
    if not 0 < sample_ratio <= 1:
        raise ValueError("Sample ratio must be between 0 and 1")
    
    # Calculate number of samples
    total_samples = len(dataset)
    num_samples = int(total_samples * sample_ratio)
    
    # Randomly sample the dataset
    sampled_indices = random.sample(range(total_samples), num_samples)
    sampled_dataset = [dataset[i] for i in sorted(sampled_indices)]
    
    # System message template (exactly match the template format)
    system_msg = '''You are a helpful assistant good at solving math problems with step-by-step reasoning. You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>. '''
    
    chatml_data = []
    for item in sampled_dataset:
        messages = item['messages']
        # Extract user message
        user_message = next(msg['value'] for msg in messages if msg['from'] == 'user')
        
        # Use original answer
        answer = item['answer'].strip()
        
        # Create ChatML format entry (exactly match the template format)
        chatml_entry = {
            "message": f"{system_msg}\n\nassistant\n{user_message.strip()}",
            "answer": answer,
            "level": 3  # Default difficulty level
        }
        
        chatml_data.append(chatml_entry)
    
    return chatml_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert dataset to ChatML format with sampling')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                      help='Ratio of data to sample (between 0 and 1)')
    parser.add_argument('--output', type=str, default='mathlv345_8k_chatml.json',
                      help='Output file name')
    args = parser.parse_args()
    
    # Load dataset from Hugging Face
    dataset = load_dataset("xDAN2099/xDAN-Agentic-openMath-r1", split="train")
    
    # Convert to ChatML format
    chatml_data = convert_to_chatml(dataset, args.sample_ratio)
    
    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(chatml_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(chatml_data)} samples to ChatML format")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
