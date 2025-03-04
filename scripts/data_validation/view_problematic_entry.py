import json

def view_entry(file_path, index):
    """
    查看数据集中特定索引的完整条目
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 0 <= index < len(data):
        print(f"条目索引 {index} 的完整内容:")
        print(json.dumps(data[index], indent=2))
        
        # 显示前后的条目以便对比
        if index > 0:
            print(f"\n前一条目 (索引 {index-1}) 的答案:")
            print(f"Answer: '{data[index-1].get('answer', '')}'")
        
        if index < len(data) - 1:
            print(f"\n后一条目 (索引 {index+1}) 的答案:")
            print(f"Answer: '{data[index+1].get('answer', '')}'")
    else:
        print(f"索引 {index} 超出范围，数据集大小为 {len(data)}")

if __name__ == "__main__":
    file_path = "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"
    view_entry(file_path, 1078)
