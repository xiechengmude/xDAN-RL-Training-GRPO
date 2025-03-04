import json

def find_empty_answers(file_path):
    """
    查找数据集中答案为空字符串的条目
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"数据集总条目数: {len(data)}")
    
    empty_answers = []
    for i, item in enumerate(data):
        answer = item.get("answer", "")
        if answer == "" or (isinstance(answer, str) and answer.strip() == ""):
            empty_answers.append(i)
    
    if empty_answers:
        print(f"\n找到 {len(empty_answers)} 条空答案记录:")
        for idx in empty_answers:
            print(f"条目索引 {idx}:")
            print(f"Prompt: {data[idx]['prompt'][:150]}...")
            print(f"Answer: '{data[idx].get('answer', '')}'")
            print("-" * 50)
    else:
        print("\n未找到空答案记录!")
        
        # 检查最短的几个答案
        answer_lengths = [(i, len(str(item.get("answer", "")))) for i, item in enumerate(data)]
        answer_lengths.sort(key=lambda x: x[1])
        
        print("\n最短的5个答案:")
        for idx, length in answer_lengths[:5]:
            print(f"条目索引 {idx}: 长度={length}")
            print(f"Answer: '{data[idx].get('answer', '')}'")
            print("-" * 30)

if __name__ == "__main__":
    file_path = "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"
    find_empty_answers(file_path)
