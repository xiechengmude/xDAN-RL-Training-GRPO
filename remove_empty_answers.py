import json
import os
import shutil

def remove_empty_answers(file_path):
    """
    找出并移除数据集中答案为空的记录
    """
    # 创建备份
    backup_path = file_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"已创建备份: {backup_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_count = len(data)
        print(f"原始数据集大小: {original_count} 条记录")
        
        # 找出空答案记录
        empty_indices = []
        for i, item in enumerate(data):
            answer = item.get("answer", "")
            if answer == "" or (isinstance(answer, str) and answer.strip() == ""):
                empty_indices.append(i)
        
        if not empty_indices:
            print(f"未发现空答案记录，无需修改")
            return
        
        print(f"发现 {len(empty_indices)} 条空答案记录")
        
        # 移除空答案记录
        new_data = [item for i, item in enumerate(data) if i not in empty_indices]
        
        # 保存修改后的数据集
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2)
        
        print(f"已成功移除所有空答案记录")
        print(f"新数据集大小: {len(new_data)} 条记录")
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return

def process_all_datasets():
    """处理所有包含空答案的数据集文件"""
    datasets_with_empty = [
        "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/examples/data/xDAN-Hardest-level-math-collection_chatml_rl.json",
        "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/examples/data/xDAN-level5-math-aime-chatml.json"
    ]
    
    for file_path in datasets_with_empty:
        print(f"\n处理文件: {os.path.basename(file_path)}")
        remove_empty_answers(file_path)

if __name__ == "__main__":
    process_all_datasets()
