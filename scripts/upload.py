from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'd8a5c361-28c1-4d54-9a3d-d6d7badd9fc5'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="xDAN-AI/LLama3-Pro-Series",
    model_dir="llama3-temp" # 本地模型目录，要求目录中必须包含configuration.json
)