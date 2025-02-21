import os
import torch
import json
from collections import OrderedDict
try:
    from safetensors.torch import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("safetensors not available. Will use PyTorch format only.")

def get_size_in_gb(tensor):
    """计算tensor的大小（GB）"""
    return tensor.nelement() * tensor.element_size() / (1024**3)

def split_model(model_path, num_splits, use_safetensors=False):
    """
    将模型切分成指定数量的部分，并保存为多个bin文件或safetensor文件
    Args:
        model_path: 模型路径
        num_splits: 要切分的份数
        use_safetensors: 是否使用safetensors格式保存
    """
    if use_safetensors and not SAFETENSORS_AVAILABLE:
        print("safetensors not available, falling back to PyTorch format")
        use_safetensors = False

    # 加载模型的state dict
    print(f"Loading model from {model_path}")
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
    
    # 计算每个tensor的大小并按大小排序
    tensors_with_size = [(name, tensor, get_size_in_gb(tensor)) 
                        for name, tensor in state_dict.items()]
    total_size = sum(size for _, _, size in tensors_with_size)
    target_size_per_split = total_size / num_splits
    
    # 创建分片
    current_dict = OrderedDict()
    current_size = 0
    split_idx = 0
    split_info = {
        "num_splits": num_splits,
        "total_size_gb": total_size,
        "splits": {}
    }

    # 确保输出目录存在
    os.makedirs(model_path, exist_ok=True)
    
    def save_current_split(split_idx, current_dict):
        if use_safetensors:
            filename = f"model-{split_idx:05d}-of-{num_splits:05d}.safetensors"
            save_path = os.path.join(model_path, filename)
            save_safetensors(current_dict, save_path)
        else:
            filename = f"pytorch_model-{split_idx:05d}-of-{num_splits:05d}.bin"
            save_path = os.path.join(model_path, filename)
            torch.save(current_dict, save_path)
        
        return filename, save_path

    # 分配tensors到不同的分片
    for name, tensor, size in tensors_with_size:
        current_dict[name] = tensor
        current_size += size
        
        # 当当前分片大小接近目标大小时保存
        if current_size >= target_size_per_split and split_idx < num_splits - 1:
            filename, save_path = save_current_split(split_idx, current_dict)
            
            # 记录分片信息
            split_info["splits"][filename] = {
                "file": filename,
                "size_gb": current_size,
                "weight_map": list(current_dict.keys())
            }
            
            print(f"Split {split_idx}: {len(current_dict)} weights, {current_size:.2f}GB saved to {filename}")
            
            # 重置当前分片
            current_dict = OrderedDict()
            current_size = 0
            split_idx += 1
    
    # 保存最后一个分片
    if current_dict:
        filename, save_path = save_current_split(split_idx, current_dict)
        split_info["splits"][filename] = {
            "file": filename,
            "size_gb": current_size,
            "weight_map": list(current_dict.keys())
        }
        print(f"Split {split_idx}: {len(current_dict)} weights, {current_size:.2f}GB saved to {filename}")
    
    # 创建权重映射文件
    weight_map = {}
    for split_data in split_info["splits"].values():
        for weight_name in split_data["weight_map"]:
            weight_map[weight_name] = split_data["file"]
    
    # 保存index文件
    index = {
        "metadata": {"total_size": total_size * (1024**3)},  # 转换为bytes
        "weight_map": weight_map
    }
    index_file = "model.safetensors.index.json" if use_safetensors else "pytorch_model.bin.index.json"
    with open(os.path.join(model_path, index_file), 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\nModel split into {num_splits} parts")
    print(f"Index file saved as {index_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split a PyTorch model into N parts")
    parser.add_argument("--model_path", type=str, 
                      default="/data/vayu/train/xDAN-RL-Training-GRPO/trans/mla/saves/qwen_eye_matrix_kv_proj",
                      help="Path to the model directory")
    parser.add_argument("--num_splits", type=int, required=True,
                      help="Number of splits to create")
    parser.add_argument("--use_safetensors", action="store_true",
                      help="Use safetensors format instead of PyTorch binary format")
    
    args = parser.parse_args()
    split_model(args.model_path, args.num_splits, args.use_safetensors)
