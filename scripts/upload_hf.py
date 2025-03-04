import os
from huggingface_hub import HfApi, create_repo, login

def upload_to_hf(
    path: str,
    repo_name: str,
    repo_type: str = "model",  # 可以是 "model" 或 "dataset"
    token: str = None,
    private: bool = False
) -> None:
    """
    将本地文件或目录上传到Hugging Face。
    
    Args:
        path: 本地文件或目录的路径
        repo_name: 要创建的仓库名称
        repo_type: 仓库类型，可以是 "model" 或 "dataset"
        token: Hugging Face的访问令牌
        private: 是否创建私有仓库
    """
    if repo_type not in ["model", "dataset"]:
        raise ValueError('repo_type must be either "model" or "dataset"')

    try:
        # 登录HuggingFace
        print("正在登录HuggingFace...")
        if token:
            login(token=token)
        
        # 创建API实例
        api = HfApi()
        
        # 创建仓库（如果不存在）
        print(f"正在创建{repo_type}仓库: {repo_name}")
        create_repo(
            repo_id=repo_name,
            repo_type=repo_type,
            private=private,
            token=token,
            exist_ok=True
        )
        print(f"创建仓库成功: {repo_name}")
        
        # 上传文件或目录
        print(f"正在上传: {path}")
        if os.path.isdir(path):
            # 如果是目录，使用upload_folder
            api.upload_folder(
                folder_path=path,
                repo_id=repo_name,
                repo_type=repo_type,
                token=token
            )
        else:
            # 如果是文件，使用upload_file
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=os.path.basename(path),
                repo_id=repo_name,
                repo_type=repo_type,
                token=token
            )
        print(f"上传成功！")
        base_url = "models" if repo_type == "model" else "datasets"
        print(f"你可以在这里查看你的{repo_type}: https://huggingface.co/{base_url}/{repo_name}")
        
    except Exception as e:
        print(f"上传过程中出现错误: {str(e)}")

if __name__ == "__main__":
    # 使用示例 - 上传模型
    path = "/data/vayu/train/models/ckpts/xDAN-L2-RL-32B-Instruct-0219-RL-step20-stage2-e1"
    repo_name = "xDAN2099/xDAN-L2-RL-32B-Instruct-0219-RL-step20-stage2-e1"
    token = os.environ.get("HF_TOKEN", None)  # Get token from environment variable
    
    if token is None:
        print("Please set the HF_TOKEN environment variable or provide it as an argument")
        import sys
        sys.exit(1)
    
    upload_to_hf(
        path=path,
        repo_name=repo_name,
        repo_type="model",  # 指定为model类型
        token=token,
        private=True
    )
    
    # 使用示例 - 上传数据集
    # upload_to_hf(
    #     path="/path/to/dataset",
    #     repo_name="username/dataset-name",
    #     repo_type="dataset",  # 指定为dataset类型
    #     token="your-token",
    #     private=True
    # )
