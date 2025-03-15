#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import argparse
import re

def remove_entries_with_missing_images(input_file, output_file, image_dir):
    """
    从JSON文件中移除引用了不存在图像的条目
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
        image_dir: 图像目录路径
    """
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据集中共有 {len(data)} 条数据")
    
    # 获取图像目录中所有文件的列表
    image_files = set(os.listdir(image_dir))
    
    # 用于存储保留的条目
    filtered_data = []
    removed_entries = []
    
    # 正则表达式用于从message字段中提取图像路径
    image_pattern = r'"image":\s*"([^"]+)"'
    
    # 检查每个条目
    for entry in data:
        message = entry.get("message", "")
        
        # 提取图像路径
        image_matches = re.findall(image_pattern, message)
        missing_images = []
        
        # 检查每个图像是否存在
        for image_path in image_matches:
            image_filename = os.path.basename(image_path)
            if image_filename not in image_files:
                missing_images.append(image_filename)
        
        # 如果没有缺失的图像，则保留该条目
        if not missing_images:
            filtered_data.append(entry)
        else:
            removed_entries.append({
                "id": entry.get("id", ""),
                "missing_images": missing_images
            })
    
    # 将过滤后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"移除了 {len(removed_entries)} 条引用缺失图像的数据")
    print(f"移除的条目ID和缺失图像:")
    for entry in removed_entries:
        print(f"  ID: {entry['id']}, 缺失图像: {', '.join(entry['missing_images'])}")
    print(f"过滤后的数据集中共有 {len(filtered_data)} 条数据")
    print(f"已将过滤后的数据保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='移除引用了不存在图像的数据条目')
    parser.add_argument('--input', required=True, help='输入的JSON文件路径')
    parser.add_argument('--output', required=True, help='输出的JSON文件路径')
    parser.add_argument('--images', required=True, help='图像目录路径')
    
    args = parser.parse_args()
    
    remove_entries_with_missing_images(args.input, args.output, args.images)

if __name__ == "__main__":
    main()
