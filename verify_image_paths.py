#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
from pathlib import Path

def verify_image_paths(json_file, image_dir):
    """
    验证JSON文件中的图像路径是否与实际图像文件匹配
    
    Args:
        json_file: 转换后的JSON文件路径
        image_dir: 图像目录路径
    """
    # 获取图像目录中的所有文件
    image_files = set(os.listdir(image_dir))
    print(f"图像目录中共有 {len(image_files)} 个文件")
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"JSON文件中共有 {len(data)} 条数据")
    
    # 提取JSON中的图像路径
    image_paths_in_json = []
    image_pattern = re.compile(r'"image": "([^"]+)"')
    
    for item in data:
        message = item.get('message', '')
        matches = image_pattern.findall(message)
        for match in matches:
            # 从路径中提取文件名
            filename = os.path.basename(match)
            image_paths_in_json.append(filename)
    
    print(f"JSON文件中共引用了 {len(image_paths_in_json)} 个图像")
    
    # 检查是否所有JSON中引用的图像都存在于图像目录中
    missing_images = []
    for image_path in image_paths_in_json:
        if image_path not in image_files:
            missing_images.append(image_path)
    
    if missing_images:
        print(f"警告: 有 {len(missing_images)} 个图像在JSON中引用但在图像目录中不存在")
        print("前5个缺失的图像:")
        for i, image in enumerate(missing_images[:5]):
            print(f"  {i+1}. {image}")
    else:
        print("所有JSON中引用的图像都存在于图像目录中")
    
    # 检查是否所有图像目录中的图像都被JSON引用
    unused_images = []
    for image_file in image_files:
        if image_file not in image_paths_in_json and not image_file.startswith('.'):
            unused_images.append(image_file)
    
    if unused_images:
        print(f"信息: 有 {len(unused_images)} 个图像在图像目录中但未被JSON引用")
        print("前5个未使用的图像:")
        for i, image in enumerate(unused_images[:5]):
            print(f"  {i+1}. {image}")
    else:
        print("所有图像目录中的图像都被JSON引用")
    
    # 检查重复引用的图像
    duplicate_images = {}
    for image_path in image_paths_in_json:
        if image_paths_in_json.count(image_path) > 1:
            duplicate_images[image_path] = image_paths_in_json.count(image_path)
    
    if duplicate_images:
        print(f"信息: 有 {len(duplicate_images)} 个图像在JSON中被多次引用")
        print("前5个重复引用的图像:")
        for i, (image, count) in enumerate(list(duplicate_images.items())[:5]):
            print(f"  {i+1}. {image} (引用了 {count} 次)")
    else:
        print("所有图像都只被引用一次")
    
    # 总结
    print("\n验证结果摘要:")
    print(f"- 图像目录中共有 {len(image_files)} 个文件")
    print(f"- JSON文件中共有 {len(data)} 条数据")
    print(f"- JSON文件中共引用了 {len(image_paths_in_json)} 个图像")
    print(f"- 缺失的图像: {len(missing_images)}")
    print(f"- 未使用的图像: {len(unused_images)}")
    print(f"- 重复引用的图像: {len(duplicate_images)}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='验证JSON文件中的图像路径是否与实际图像文件匹配')
    parser.add_argument('--json', type=str, required=True, help='转换后的JSON文件路径')
    parser.add_argument('--images', type=str, required=True, help='图像目录路径')
    
    args = parser.parse_args()
    
    verify_image_paths(args.json, args.images)
