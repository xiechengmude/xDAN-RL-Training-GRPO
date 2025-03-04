import json
import os
import glob

def check_dataset_file(file_path):
    """检查单个数据集文件中的空答案"""
    try:
        # 根据文件扩展名决定如何加载
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"错误: 无法解析JSON文件 {file_path}")
                    return []
        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"不支持的文件格式: {file_path}")
            return []
        
        # 检查是否为列表格式
        if not isinstance(data, list):
            print(f"警告: {file_path} 不是列表格式的数据集")
            return []
        
        empty_answers = []
        for i, item in enumerate(data):
            # 检查是否有answer字段
            if "answer" not in item:
                continue
                
            answer = item.get("answer", "")
            if answer == "" or (isinstance(answer, str) and answer.strip() == ""):
                empty_answers.append((i, item.get("prompt", "")[:150]))
        
        return empty_answers
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return []

def check_all_datasets(directory):
    """检查目录中所有数据集文件"""
    # 获取所有json和jsonl文件
    json_files = glob.glob(os.path.join(directory, "**/*.json"), recursive=True)
    jsonl_files = glob.glob(os.path.join(directory, "**/*.jsonl"), recursive=True)
    all_files = json_files + jsonl_files
    
    print(f"找到 {len(all_files)} 个数据集文件")
    
    all_empty_answers = {}
    for file_path in all_files:
        print(f"\n检查文件: {file_path}")
        empty_answers = check_dataset_file(file_path)
        
        if empty_answers:
            rel_path = os.path.relpath(file_path, directory)
            all_empty_answers[rel_path] = empty_answers
            print(f"  - 发现 {len(empty_answers)} 条空答案记录")
        else:
            print("  - 未发现空答案记录")
    
    # 输出汇总结果
    print("\n===== 汇总结果 =====")
    if all_empty_answers:
        total_empty = sum(len(items) for items in all_empty_answers.values())
        print(f"总共发现 {total_empty} 条空答案记录，分布在 {len(all_empty_answers)} 个文件中")
        
        for file_path, empty_answers in all_empty_answers.items():
            print(f"\n文件: {file_path}")
            print(f"空答案记录数: {len(empty_answers)}")
            print("示例记录:")
            for i, (idx, prompt) in enumerate(empty_answers[:3]):
                print(f"  索引 {idx}: {prompt}...")
            if len(empty_answers) > 3:
                print(f"  ... 以及其他 {len(empty_answers) - 3} 条记录")
    else:
        print("所有数据集文件中均未发现空答案记录")

if __name__ == "__main__":
    data_dir = "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/examples/data"
    check_all_datasets(data_dir)
