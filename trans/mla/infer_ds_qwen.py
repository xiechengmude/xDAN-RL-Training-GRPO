from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from qwen2.modeling_qwen2_v1 import Qwen2ForCausalLM

ds_path = "/data/vayu/train/models/DeepSeek-R1-Distill-Qwen-32B"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(ds_path, trust_remote_code=True)
#model = Qwen2ForCausalLM.from_pretrained(
#    ds_path,attn_implementation="sdpa", device_map='auto'
#)
model = Qwen2ForCausalLM.from_pretrained(ds_path, attn_implementation="eager", partial_rotary_factor=1, rope_repeat=True)
#tokenizer = AutoTokenizer.from_pretrained(ds_path)

hidden_size = model.config.hidden_size
n_heads = model.config.num_attention_heads
kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size//model.config.num_attention_heads
latent_dim = kv_heads * head_dim
kv_groups = model.config.num_attention_heads // model.config.num_key_value_heads
model.config.partial_rotary_factor

for name,module in model.named_modules():
    if 'k_up_proj' in name or "v_up_proj" in name:
        module.weight.data = torch.stack([torch.eye(kv_heads*head_dim).reshape(kv_heads, head_dim, kv_heads*head_dim)]*kv_groups,dim=1).reshape(hidden_size, kv_heads*head_dim).contiguous().to(module.weight.data.device,module.weight.data.dtype)

# 保存修改后的模型和tokenizer
import os
save_dir = "saves"
model_name = "qwen_eye_matrix_kv_proj"  # 反映了对k_up_proj和v_up_proj使用eye matrix的改动
save_path = os.path.join(save_dir, model_name)
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to {save_path}")

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