import json
import sys

def check_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    print("\nChecking answers format...")
    
    issues = []
    for i, item in enumerate(data):
        answer = item.get("answer", "")
        if not answer:
            issues.append((i, "Empty answer"))
            continue
            
        # Check if answer starts with "$"
        if not isinstance(answer, str):
            issues.append((i, f"Answer is not a string: {type(answer)}"))
        elif not answer.strip().startswith("$"):
            issues.append((i, f"Answer does not start with '$': {answer}"))
    
    if issues:
        print(f"\nFound {len(issues)} issues:")
        for idx, issue in issues[:20]:  # Show first 20 issues
            print(f"Entry {idx}: {issue}")
        
        if len(issues) > 20:
            print(f"... and {len(issues) - 20} more issues.")
    else:
        print("No issues found!")
    
    # Print some examples
    print("\nSample entries:")
    for i in range(min(5, len(data))):
        print(f"\nEntry {i}:")
        print(f"Prompt: {data[i]['prompt'][:100]}...")
        print(f"Answer: {data[i]['answer']}")

if __name__ == "__main__":
    file_path = "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"
    check_dataset(file_path)
