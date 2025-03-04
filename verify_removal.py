import json
import os

def verify_removal(file_path):
    """验证数据集中是否还存在空答案记录"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"数据集: {os.path.basename(file_path)}")
        print(f"记录数: {len(data)}")
        
        empty_count = 0
        for i, item in enumerate(data):
            answer = item.get("answer", "")
            if answer == "" or (isinstance(answer, str) and answer.strip() == ""):
                empty_count += 1
                print(f"发现空答案记录，索引: {i}")
        
        if empty_count == 0:
            print("验证成功: 数据集中不存在空答案记录")
        else:
            print(f"验证失败: 数据集中仍存在 {empty_count} 条空答案记录")
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

def verify_all_datasets():
    """验证所有修改过的数据集文件"""
    datasets = [
        "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/examples/data/xDAN-Hardest-level-math-collection_chatml_rl.json",
        "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"
    ]
    
    for file_path in datasets:
        print(f"\n验证文件: {os.path.basename(file_path)}")
        verify_removal(file_path)

if __name__ == "__main__":
    verify_all_datasets()
