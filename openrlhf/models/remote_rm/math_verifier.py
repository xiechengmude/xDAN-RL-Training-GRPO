import argparse
import json
import os
from flask import Flask, request, jsonify
from typing import List, Dict, Any, Union

app = Flask(__name__)

class MathVerifier:
    def __init__(self, dataset_path: str, prompt_template: str = "chatml", input_key: str = "message"):
        """初始化数学验证器

        Args:
            dataset_path: 训练数据集路径
            prompt_template: 提示模板类型
            input_key: 输入消息的key
        """
        self.dataset_path = dataset_path
        self.prompt_template = prompt_template
        self.input_key = input_key
        self.examples = self._load_dataset()
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """加载数据集"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _verify_format(self, messages: List[Dict[str, str]]) -> bool:
        """验证消息格式是否正确

        Args:
            messages: 消息列表

        Returns:
            bool: 格式是否正确
        """
        if not isinstance(messages, list) or len(messages) < 2:
            return False
            
        # 检查消息格式
        for msg in messages:
            if not isinstance(msg, dict) or 'from' not in msg or 'value' not in msg:
                return False
            if msg['from'] not in ['user', 'assistant']:
                return False
                
        # 检查最后一条是否是assistant的回答
        if messages[-1]['from'] != 'assistant':
            return False
            
        return True
    
    def _verify_math_content(self, response: str) -> float:
        """验证数学内容的质量

        Args:
            response: assistant的回答

        Returns:
            float: 质量分数 (0-1)
        """
        score = 0.0
        
        # 检查是否包含思考过程
        if '<think>' in response and '</think>' in response:
            score += 0.3
            
        # 检查是否有清晰的步骤
        if any(str(i) + '.' in response for i in range(1, 10)):
            score += 0.2
            
        # 检查是否有最终答案
        if '\\boxed{' in response and '}' in response:
            score += 0.3
            
        # 检查LaTeX格式
        if '\\' in response and '{' in response and '}' in response:
            score += 0.2
            
        return min(1.0, score)

    def get_reward(self, messages: List[Dict[str, str]]) -> Dict[str, Union[float, str]]:
        """获取回答的奖励分数

        Args:
            messages: 消息列表

        Returns:
            Dict: 包含分数和反馈的字典
        """
        # 验证格式
        if not self._verify_format(messages):
            return {
                'score': 0.0,
                'feedback': 'Invalid message format'
            }
        
        # 获取assistant的回答
        response = messages[-1]['value']
        
        # 计算质量分数
        score = self._verify_math_content(response)
        
        return {
            'score': score,
            'feedback': f'Response quality score: {score:.2f}'
        }

@app.route('/get_reward', methods=['POST'])
def get_reward():
    """处理获取奖励的HTTP请求"""
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': 'Invalid request data'}), 400
            
        reward = verifier.get_reward(data['messages'])
        return jsonify(reward)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--prompt-template', type=str, required=True, help='Prompt template type')
    parser.add_argument('--input_key', type=str, default='message', help='Key for input messages')
    
    args = parser.parse_args()
    
    global verifier
    verifier = MathVerifier(
        dataset_path=args.dataset,
        prompt_template=args.prompt_template,
        input_key=args.input_key
    )
    
    # 从环境变量获取端口
    port = int(os.environ.get('REWARD_MODEL_PORT', 5001))
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()