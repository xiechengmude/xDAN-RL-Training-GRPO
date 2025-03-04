import json
import os

def view_empty_answers(file_path, indices):
    """查看指定文件中特定索引的空答案记录"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件: {os.path.basename(file_path)}")
        print(f"总记录数: {len(data)}")
        
        for idx in indices:
            if 0 <= idx < len(data):
                item = data[idx]
                print(f"\n记录索引: {idx}")
                print(f"Prompt: {item.get('prompt', '')[:200]}...")
                print(f"Answer: '{item.get('answer', '')}'")
                
                # 显示前后的记录以便对比
                if idx > 0:
                    print(f"\n前一条记录 (索引 {idx-1}) 的答案:")
                    print(f"Answer: '{data[idx-1].get('answer', '')}'")
                
                if idx < len(data) - 1:
                    print(f"\n后一条记录 (索引 {idx+1}) 的答案:")
                    print(f"Answer: '{data[idx+1].get('answer', '')}'")
                
                print("-" * 50)
            else:
                print(f"索引 {idx} 超出范围，数据集大小为 {len(data)}")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

if __name__ == "__main__":
    # 查看xDAN-Hardest-level-math-collection_chatml_rl.json中的三条空答案记录
    hardest_file = "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/examples/data/xDAN-Hardest-level-math-collection_chatml_rl.json"
    view_empty_answers(hardest_file, [18473, 19720, 19861])
