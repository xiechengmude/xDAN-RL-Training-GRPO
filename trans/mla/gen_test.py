from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os

def load_model_and_tokenizer(model_path):
    """加载模型和tokenizer"""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用float16减少显存占用
        device_map="auto",          # 自动处理模型部署
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, **kwargs):
    """生成回复"""
    # 设置默认生成参数
    default_kwargs = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    # 更新生成参数
    generate_kwargs = {**default_kwargs, **kwargs}
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 如果响应包含原始提示，则移除
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Test model generation")
    parser.add_argument("--model_path", type=str, 
                      default="/data/vayu/train/xDAN-RL-Training-GRPO/trans/mla/saves/qwen_eye_matrix_kv_proj",
                      help="Path to the model directory")
    parser.add_argument("--prompt", type=str, 
                      default="如何做西红柿炒鸡蛋？",
                      help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                      help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                      help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # 生成回复
    response = generate_response(
        model, 
        tokenizer, 
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("输入提示：", args.prompt)
    print("-"*50)
    print("模型回答：", response)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
