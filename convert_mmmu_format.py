#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import argparse
from pathlib import Path
import re

def extract_image_references(question):
    """从问题中提取图像引用，例如 <image 1>"""
    image_refs = re.findall(r'<image (\d+)>', question)
    return [int(ref) for ref in image_refs]

def convert_mmmu_to_desired_format(input_file, output_file, image_dir=None):
    """
    将MMMU数据集转换为期望的格式
    
    Args:
        input_file: MMMU数据集的jsonl文件路径
        output_file: 输出文件路径
        image_dir: 可选，本地图像目录路径，如果提供，将使用本地图像路径
    """
    converted_data = []
    
    # 读取MMMU数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                
                # 只处理单图情况，检查是否只有image_1字段且不为None
                if 'image_1' not in item or item['image_1'] is None:
                    continue
                
                # 检查是否有多图，如果有则跳过
                has_multiple_images = False
                for i in range(2, 8):
                    image_key = f'image_{i}'
                    if image_key in item and item[image_key] is not None:
                        has_multiple_images = True
                        break
                
                if has_multiple_images:
                    continue
                
                # 提取问题和图像引用
                question = item.get('question', '')
                
                # 清理问题文本，移除图像引用
                clean_question = re.sub(r'<image \d+>', '', question).strip()
                
                # 构建内容数组
                content = []
                
                # 添加图像
                image_path = item['image_1'].get('path', '')
                
                # 从URL中提取文件名
                image_filename = os.path.basename(image_path)
                
                # 使用简单的相对路径，不带file://前缀
                if image_dir:
                    # 使用指定的图像目录
                    local_path = f"images/{image_filename}"
                else:
                    # 默认使用MMMU-Reasoning-Distill-Validation/images目录
                    local_path = f"MMMU-Reasoning-Distill-Validation/images/{image_filename}"
                
                content.append({
                    "type": "image",
                    "image": local_path
                })
                
                # 添加文本问题
                if clean_question:
                    content.append({
                        "type": "text",
                        "text": clean_question
                    })
                
                # 如果有选项，添加到问题中
                if 'options' in item and item['options']:
                    options_text = "\n选项:\n"
                    for i, option in enumerate(item['options']):
                        option_letter = chr(65 + i)  # A, B, C, ...
                        options_text += f"{option_letter}: {option}\n"
                    
                    # 更新最后一个文本内容或添加新的
                    if content and content[-1]["type"] == "text":
                        content[-1]["text"] += "\n" + options_text
                    else:
                        content.append({
                            "type": "text",
                            "text": options_text
                        })
                
                # 构建消息对象
                message_obj = [{
                    "role": "user",
                    "content": content
                }]
                
                # 获取答案
                answer = item.get('answer', '')
                
                # 如果答案是选项字母，可以格式化为数学符号格式
                if answer and len(answer) == 1 and 'A' <= answer <= 'Z':
                    formatted_answer = f"${answer}$"
                else:
                    # 尝试从assistant的回复中提取答案
                    for msg in item.get('messages', []):
                        if msg.get('role') == 'assistant':
                            answer_content = msg.get('content', '')
                            answer_match = re.search(r'<answer>(.*?)</answer>', answer_content, re.DOTALL)
                            if answer_match:
                                answer_text = answer_match.group(1).strip()
                                # 提取\boxed{}中的内容
                                boxed_match = re.search(r'\\boxed{(.*?)}', answer_text)
                                if boxed_match:
                                    answer = boxed_match.group(1).strip()
                                    formatted_answer = f"${answer}$"
                                    break
                    else:
                        formatted_answer = f"${answer}$"
                
                # 创建最终的转换对象
                converted_item = {
                    "message": json.dumps(message_obj),
                    "answer": formatted_answer,
                    "id": item.get('id', ''),
                    "source": item.get('source', ''),
                    "topic_difficulty": item.get('topic_difficulty', ''),
                    "question_type": item.get('question_type', '')
                }
                
                converted_data.append(converted_item)
                
            except json.JSONDecodeError:
                print(f"警告: 跳过无效的JSON行: {line[:100]}...")
                continue
    
    # 写入转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成! 共转换 {len(converted_data)} 条数据到 {output_file}")

def download_images(input_file, output_dir):
    """
    下载MMMU数据集中的图像到本地目录
    
    Args:
        input_file: MMMU数据集的jsonl文件路径
        output_dir: 图像保存目录
    """
    import requests
    from tqdm import tqdm
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取MMMU数据集
    image_urls = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                
                # 收集所有图像URL
                for i in range(1, 8):  # image_1 到 image_7
                    image_key = f'image_{i}'
                    if image_key in item and item[image_key] is not None:
                        image_url = item[image_key].get('path', '')
                        if image_url:
                            image_urls.add(image_url)
                
                # 也检查images数组
                for image_obj in item.get('images', []):
                    image_url = image_obj.get('path', '')
                    if image_url:
                        image_urls.add(image_url)
                
            except json.JSONDecodeError:
                continue
    
    # 下载图像
    print(f"开始下载 {len(image_urls)} 张图像...")
    for url in tqdm(image_urls):
        try:
            # 从URL中提取文件名
            filename = os.path.basename(url)
            output_path = os.path.join(output_dir, filename)
            
            # 如果文件已存在，跳过
            if os.path.exists(output_path):
                continue
            
            # 下载图像
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        except Exception as e:
            print(f"下载图像失败 {url}: {str(e)}")
    
    print(f"图像下载完成! 保存到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='将MMMU数据集转换为期望的格式')
    parser.add_argument('--input', type=str, required=True, help='输入MMMU jsonl文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--download-images', action='store_true', help='是否下载图像到本地')
    parser.add_argument('--image-dir', type=str, help='本地图像保存目录')
    parser.add_argument('--use-local-images', action='store_true', help='使用本地图像路径')
    
    args = parser.parse_args()
    
    # 如果需要下载图像
    if args.download_images:
        image_dir = args.image_dir or os.path.join(os.path.dirname(args.output), 'images')
        download_images(args.input, image_dir)
        # 使用本地图像路径进行转换
        convert_mmmu_to_desired_format(args.input, args.output, image_dir)
    elif args.use_local_images:
        # 使用本地图像路径
        image_dir = args.image_dir or os.path.join(os.path.dirname(args.input), 'images')
        convert_mmmu_to_desired_format(args.input, args.output, image_dir)
    else:
        # 不下载图像，使用原始URL
        convert_mmmu_to_desired_format(args.input, args.output)

if __name__ == '__main__':
    main()
