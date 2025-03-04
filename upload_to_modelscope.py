#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ModelScope 本地文件上传脚本
用于将本地文件上传至ModelScope平台
"""

import os
import argparse
from modelscope.hub.api import HubApi
from modelscope.utils.constant import ModelFile


def parse_args():
    parser = argparse.ArgumentParser(description='Upload files to ModelScope')
    parser.add_argument('--model_id', type=str, required=True, 
                        help='Model ID in format of {organization}/{model_name}')
    parser.add_argument('--local_path', type=str, required=True,
                        help='Local file or directory path to upload')
    parser.add_argument('--remote_path', type=str, default='',
                        help='Remote path in the model repo (default: same as local)')
    parser.add_argument('--token', type=str, default=None,
                        help='ModelScope API token (can also be set via MODELSCOPE_API_TOKEN env var)')
    parser.add_argument('--commit_message', type=str, default='Upload files via script',
                        help='Commit message for the upload')
    parser.add_argument('--create_repo', action='store_true',
                        help='Create the repository if it does not exist')
    return parser.parse_args()


def upload_to_modelscope(model_id, local_path, remote_path=None, token=None, 
                         commit_message='Upload files via script', create_repo=False):
    """
    上传本地文件到ModelScope
    
    Args:
        model_id (str): 模型ID，格式为 {组织}/{模型名称}
        local_path (str): 本地文件或目录路径
        remote_path (str, optional): 远程仓库中的路径，默认与本地路径相同
        token (str, optional): ModelScope API令牌，如果未提供则尝试从环境变量获取
        commit_message (str, optional): 提交消息
        create_repo (bool, optional): 如果仓库不存在是否创建
    python /Users/gumpcehng/CascadeProjects/xDAN-RL-Training-GRPO/upload_to_modelscope.py 
    --model_id xDAN2099/xDAN-Video-testing 
    --local_path /Users/gumpcehng/Documents/xDAN/xDAN-Video-testing-0301 
    --token 721ec736-ec61-45ad-a1ec-b3b339ef016d --create_repo
    Returns:
        bool: 上传是否成功
    """
    # 获取API令牌
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
        if token is None:
            raise ValueError("API token must be provided either through --token argument "
                             "or MODELSCOPE_API_TOKEN environment variable")
    
    # 初始化API
    api = HubApi()
    api.login(token)
    
    # 检查本地路径是否存在
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local path does not exist: {local_path}")
    
    # 如果remote_path未指定，使用local_path的文件名或目录名
    if not remote_path:
        remote_path = os.path.basename(local_path)
    
    try:
        # 检查仓库是否存在
        try:
            api.get_model(model_id)
            print(f"Repository {model_id} exists.")
        except Exception as e:
            if create_repo:
                print(f"Creating repository {model_id}...")
                api.create_model(model_id)
            else:
                raise ValueError(f"Repository {model_id} does not exist. "
                                 f"Use --create_repo to create it. Error: {str(e)}")
        
        # 执行上传
        print(f"Uploading {local_path} to {model_id}/{remote_path}...")
        
        if os.path.isdir(local_path):
            # 上传目录
            print(f"Uploading directory: {local_path} -> {remote_path}")
            api.upload_folder(
                repo_id=model_id,
                folder_path=local_path,
                path_in_repo=remote_path,
                commit_message=commit_message
            )
        else:
            # 上传单个文件
            print(f"Uploading file: {local_path} -> {remote_path}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=model_id,
                commit_message=commit_message
            )
        
        print(f"Upload completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error uploading to ModelScope: {str(e)}")
        return False


def main():
    args = parse_args()
    upload_to_modelscope(
        model_id=args.model_id,
        local_path=args.local_path,
        remote_path=args.remote_path,
        token=args.token,
        commit_message=args.commit_message,
        create_repo=args.create_repo
    )


if __name__ == "__main__":
    main()
