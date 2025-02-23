import json

def print_formatted_item(item, title):
    print(f"\n=== {title} ===")
    print("Prompt:")
    print("-" * 50)
    print(item["prompt"][:500] + "..." if len(item["prompt"]) > 500 else item["prompt"])
    print("-" * 50)
    print("\nAnswer:", item["answer"])
    print("\nLevel:", item["level"])
    print("=" * 50)

def check_format(item):
    has_system = "<|im_start|>system" in item["prompt"]
    has_user = "<|im_start|>user" in item["prompt"]
    has_assistant = "<|im_start|>assistant" in item["prompt"]
    has_think = "<div className=\"think-block\">" in item["prompt"] or "<think>" in item["prompt"]
    has_answer = "<answer>" in item["prompt"]
    return all([has_system, has_user, has_assistant, has_think, has_answer])

def main():
    # 读取原始样板文件
    print("\nReading template file...")
    with open('../data/mathlv345_8k_chatml.json', 'r') as f:
        template_data = json.load(f)

    # 读取生成的文件
    print("\nReading generated file...")
    with open('test_output.json', 'r') as f:
        generated_data = json.load(f)

    # 显示两个文件的第一个样本进行对比
    print("\n比较第一个样本:")
    print_formatted_item(template_data[0], "Template Data")
    print_formatted_item(generated_data[0], "Generated Data")

    # 比较基本统计信息
    print("\n基本统计信息:")
    print(f"Template data count: {len(template_data)}")
    print(f"Generated data count: {len(generated_data)}")

    # 检查格式一致性
    template_format = all(check_format(item) for item in template_data[:10])
    generated_format = all(check_format(item) for item in generated_data[:10])

    print("\n格式检查 (前10条):")
    print(f"Template format consistent: {template_format}")
    print(f"Generated format consistent: {generated_format}")

if __name__ == "__main__":
    main()
