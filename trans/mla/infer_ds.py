from qwen2.modeling_qwen2_v1 import Qwen2ForCausalLM
from transformers import AutoTokenizer
import torch
from copy import deepcopy
from tqdm import tqdm

ds_path = "/data/h3030118/data/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/3865e12a1eb7cbd641ab3f9dfc28c588c6b0c1e9"

model = Qwen2ForCausalLM.from_pretrained(ds_path, attn_implementation="eager", partial_rotary_factor=1, rope_repeat=True)
tokenizer = AutoTokenizer.from_pretrained(ds_path)

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

output = model.generate(**tokenizer("给我讲一个故事吧",return_tensors="pt").to("cuda:1"), max_new_tokens=500, do_sample=False)
print(tokenizer.batch_decode(output)[0])