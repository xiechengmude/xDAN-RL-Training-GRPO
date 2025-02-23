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
    has_system = "
