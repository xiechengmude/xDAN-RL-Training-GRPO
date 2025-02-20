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
    
    chatml_data = []
    for item in sampled_dataset:
        messages = item['messages']
        # Extract user message and assistant response
        user_message = next(msg['value'] for msg in messages if msg['from'] == 'user')
        assistant_response = next(msg['value'] for msg in messages if msg['from'] == 'assistant')
        
        # Create ChatML format entry
        chatml_entry = {
            "message": json.dumps([{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": user_message
                }]
            }]),
            "answer": assistant_response
        }
        
        chatml_data.append(chatml_entry)
    
    return chatml_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert dataset to ChatML format with sampling')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                      help='Ratio of data to sample (between 0 and 1)')
    args = parser.parse_args()
    
    # Load dataset from Hugging Face
    dataset = load_dataset("xDAN2099/xDAN-Agentic-openMath-r1", split="train")
    
    # Convert to ChatML format with sampling
    chatml_data = convert_to_chatml(dataset, args.sample_ratio)
    
    # Generate output path with sample ratio in filename
    ratio_percent = int(args.sample_ratio * 100)
    output_path = f'./xDAN-Agentic-openMath-r1-{ratio_percent}percent-chatml.json'
    
    # Ensure the output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chatml_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted data saved to {output_path}")
    print(f"Total examples in original dataset: {len(dataset)}")
    print(f"Total examples sampled and converted: {len(chatml_data)} ({ratio_percent}%)")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
