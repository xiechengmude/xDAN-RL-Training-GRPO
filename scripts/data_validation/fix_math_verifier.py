import json
import os
import shutil

def fix_math_verifier():
    """
    修复math_verifier.py中的IndexError问题
    """
    file_path = "/Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/openrlhf/models/remote_rm/math_verifier.py"
    backup_path = file_path + ".bak"
    
    # 创建备份
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 查找并修复问题代码
    old_code = """        answer = item["answer"].strip()
        # we require the answer to be in latex format
        if answer[0] != "$":
            answer = "$" + answer + "$" """
    
    new_code = """        answer = item["answer"].strip()
        # we require the answer to be in latex format
        if answer and answer[0] != "$":
            answer = "$" + answer + "$"
        elif not answer:
            # 处理空答案的情况
            print(f"Warning: Empty answer found in dataset item: {item}")
            answer = "$\\text{Empty Answer}$" """
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w') as f:
            f.write(content)
        print("Successfully fixed math_verifier.py")
    else:
        print("Could not find the exact code pattern to replace. Manual fix may be required.")
        # 尝试查找可能的位置
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "answer[0] != \"$\"" in line:
                print(f"Potential issue found at line {i+1}: {line}")

if __name__ == "__main__":
    fix_math_verifier()
