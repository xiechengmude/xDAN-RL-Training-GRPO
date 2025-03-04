import pandas as pd
import json

def convert_conversation_to_pairs(conversations):
    """Convert a single conversation to instruction-output pairs"""
    pairs = []
    for i in range(0, len(conversations)-1, 2):
        if (conversations[i]['from'].lower() == 'human' and 
            conversations[i+1]['from'].lower() == 'assistant'):
            pair = {
                'instruction': conversations[i]['value'].strip(),
                'input': "",
                'output': conversations[i+1]['value'].strip()
            }
            pairs.append(pair)
    return pairs

def convert_sharegpt_to_alpaca(df):
    """
    Convert ShareGPT format to Alpaca format using pandas apply.
    
    ShareGPT format:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "assistant", "value": "..."},
            ...
        ]
    }
    
    Alpaca format:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }
    """
    print(f"Processing {len(df)} rows...")
    
    # 确保conversations列的数据是正确的格式
    def process_row(row):
        convs = row['conversations']
        # 如果conversations是嵌套的列表，取第一个元素
        if isinstance(convs, list) and len(convs) > 0 and isinstance(convs[0], list):
            convs = convs[0]
        return convert_conversation_to_pairs(convs)
    
    # 转换每一行
    result = df.apply(process_row, axis=1)
    print(f"Converted to {len(result)} pairs")
    
    # 展开列表成为单独的行
    result = result.explode()
    
    # 将字典转换为单独的列
    result = pd.json_normalize(result)
    print(f"Final result: {len(result)} rows")
    
    # 只显示前几行数据作为示例
    if len(result) > 0:
        print("\nFirst row example:")
        print(result.iloc[0])
    
    return result

def save_to_jsonl(df, output_file):
    """Save DataFrame to JSONL format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved {len(df)} entries to {output_file}")

# Example usage
if __name__ == "__main__":
    # Example ShareGPT data
    data = {
        'conversations': [
            {"from": "human", "value": "What is Python?"},
            {"from": "assistant", "value": "Python is a programming language."},
            {"from": "human", "value": "Is it easy to learn?"},
            {"from": "assistant", "value": "Yes, Python is known for being easy to learn."}
        ]
    }
    
    # Create DataFrame with the example data
    df = pd.DataFrame([data])
    
    # Convert to Alpaca format
    alpaca_df = convert_sharegpt_to_alpaca(df)
    
    # Save to JSONL file
    save_to_jsonl(alpaca_df, 'alpaca_format.jsonl')
