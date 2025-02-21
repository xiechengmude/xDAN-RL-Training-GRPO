from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ds_path = "/data/h3030118/data/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/3865e12a1eb7cbd641ab3f9dfc28c588c6b0c1e9"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(ds_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ds_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # 自动分配GPU/CPU
    trust_remote_code=True
)
print(model)

# 输入文本
prompt = "如何做西红柿炒鸡蛋？"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成配置
generate_kwargs = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1
}

# 推理
outputs = model.generate(**inputs, **generate_kwargs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("模型回答：", response)