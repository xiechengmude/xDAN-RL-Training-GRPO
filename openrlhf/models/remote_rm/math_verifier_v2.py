"""
数学推理任务的奖励函数模块

本模块包含三个核心奖励函数：
1. ngram_repetition_penalty: 用于检测和惩罚文本中的重复模式
2. accuracy_reward: 验证数学答案的正确性
3. format_reward: 检查回答格式的规范性

主要用于强化学习训练过程中的奖励计算。

配置建议：
1. v1兼容配置（默认）:
   format_weight = 0.5
   accuracy_weight = 1.0
   repetition_weight = 0.0
   min_similarity = 0.0

2. v2增强配置:
   format_weight = 0.3     # 降低格式权重，因为增加了重复惩罚
   accuracy_weight = 0.5   # 准确性权重
   repetition_weight = 0.2 # 启用重复惩罚
   min_similarity = 0.8    # 提高相似度匹配要求
   ngram_length = 50       # 重复检测窗口大小
   ngram_penalty = -0.025  # 重复惩罚系数

启用v2配置的命令示例：
```bash
python math_verifier_v2.py \
    --dataset path/to/dataset.json \
    --format-weight 0.3 \
    --accuracy-weight 0.5 \
    --repetition-weight 0.2
```

v2配置的优势：
1. 更严格的质量控制（通过重复惩罚）
2. 更精确的相似度匹配（通过min_similarity阈值）
3. 更均衡的奖励分配
"""

import json
import os
import random
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import List, Dict, Optional, Union, Tuple

import Levenshtein
from flask import Flask, jsonify, request
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# 配置数据类
@dataclass
class VerifierConfig:
    """验证器配置类
    
    权重配置说明：
    1. format_weight + accuracy_weight = 1.0，确保基础分数在合理范围内
    2. accuracy_weight占比最高，因为数学正确性最重要
    3. repetition_weight作为额外的惩罚项，不占用基础权重
       - 重复内容和重复格式最多扣除30%
    """
    format_weight: float = 0.2     # 格式奖励权重
    accuracy_weight: float = 0.8    # 准确性奖励权重
    repetition_weight: float = 0.3  # 重复惩罚权重（作为额外惩罚，最多扣除30%）
    ngram_length: int = 5          # n-gram长度，用于检测短文本中的重复
    ngram_penalty: float = -0.05   # n-gram惩罚系数
    min_similarity: float = 0.0    # 最小相似度阈值

    def __post_init__(self):
        """验证权重配置的合法性"""
        base_weight = self.format_weight + self.accuracy_weight
        if not abs(base_weight - 1.0) < 1e-6:  # 使用浮点数比较的安全方式
            raise ValueError(f"基础权重之和（format_weight + accuracy_weight）必须为1.0，当前为{base_weight}")

class RewardCalculator:
    """奖励计算器基类"""
    def calculate(self, content: str, **kwargs) -> float:
        raise NotImplementedError

class FormatReward(RewardCalculator):
    """格式奖励计算器"""
    def __init__(self):
        # 更新正则表达式以允许标签之间有可选的空格
        self.format_pattern = r"^\s*<think>(?:(?!</think>).)*</think>\s*<answer>(?:(?!</answer>).)*</answer>\s*\Z"

    def calculate(self, content: str, **kwargs) -> float:
        """验证回答格式"""
        think_count = content.count("<think>")
        answer_count = content.count("<answer>")
        if bool(re.match(self.format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1:
            return 1.0
        return 0.0

class AccuracyReward(RewardCalculator):
    """准确性奖励计算器"""
    def calculate(self, content: str, solution: str, **kwargs) -> float:
        """验证答案正确性"""
        try:
            gold_parsed = parse(solution, extraction_mode="first_match", 
                              extraction_config=[LatexExtractionConfig()])
            
            if len(gold_parsed) == 0:
                print(f"Warning: Gold solution cannot be parsed: {solution}")
                return 0.0  # 改为返回0分，标记出问题样本
                
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            
            if len(answer_parsed) == 0:
                print(f"Warning: Generated answer cannot be parsed: {content}")
                return 0.0  # 生成的答案也无法解析时返回0分
            
            reward = float(verify(answer_parsed, gold_parsed))
            return reward
        except Exception as e:
            print(f"Math verification error: {e}")
            return 0.0  # 验证出错时返回0分，而不是1分

class RepetitionPenalty(RewardCalculator):
    """重复内容惩罚计算器"""
    def __init__(self, config: VerifierConfig):
        self.config = config

    def calculate(self, content: str) -> float:
        """计算重复内容惩罚分数
        
        计算逻辑：
        1. 检查格式标签重复
        2. 检查内容重复（使用n-gram）
        3. 返回更严重的惩罚值
        """
        format_penalty = 0.0
        content_penalty = 0.0
        
        # 检查格式标签重复
        think_count = content.count("<think>")
        answer_count = content.count("<answer>")
        if think_count > 1 or answer_count > 1:
            format_penalty = -0.3  # 格式标签重复，30%惩罚
            
        # 检查内容重复
        # 移除标签，只检查实际内容
        clean_content = re.sub(r'<[^>]+>', '', content)
        words = clean_content.split()
        
        if len(words) < self.config.ngram_length:
            content_penalty = 0.0
        else:
            # 1. 检查完整句子重复
            sentences = [s.strip() for s in clean_content.split(',') if s.strip()]
            unique_sentences = set(sentences)
            if len(sentences) > len(unique_sentences):
                # 句子重复，严重惩罚
                content_penalty = -0.3
            else:
                # 2. 检查n-gram重复
                ngrams = {}
                for i in range(len(words) - self.config.ngram_length + 1):
                    ngram = ' '.join(words[i:i + self.config.ngram_length])
                    ngrams[ngram] = ngrams.get(ngram, 0) + 1
                    
                # 计算重复程度
                max_repeat = max(ngrams.values())
                if max_repeat > 1:
                    # 根据重复次数计算惩罚
                    # 每次重复增加15%的惩罚，最多30%
                    content_penalty = -min(0.3, (max_repeat - 1) * 0.15)
            
        # 返回更严重的惩罚
        return min(format_penalty, content_penalty)

class MathVerifier:
    """数学验证器主类"""
    def __init__(self, config: VerifierConfig):
        self.config = config
        self.problem_to_answer: Dict[str, str] = {}
        self.format_reward = FormatReward()
        self.accuracy_reward = AccuracyReward()
        self.repetition_penalty = RepetitionPenalty(config)

    def load_dataset(self, dataset_paths: List[str], input_key: str = "prompt"):
        """加载数据集"""
        for path in dataset_paths:
            if path.endswith("json"):
                with open(path, "r") as f:
                    dataset = json.load(f)
            elif path.endswith("jsonl"):
                with open(path, "r") as f:
                    dataset = [json.loads(l) for l in f.readlines()]
            else:
                raise ValueError(f"Unsupported file format: {path}")

            for item in dataset:
                problem = item[input_key]
                answer = item["answer"].strip()
                if not answer.startswith("$"):
                    answer = "$" + answer + "$"
                self.problem_to_answer[problem] = answer

    def find_similar_problem(self, problem: str) -> Optional[str]:
        """查找相似问题"""
        max_sim = -1
        target_problem = None
        for p in self.problem_to_answer.keys():
            sim = Levenshtein.ratio(problem, p)
            if sim > max_sim:  # 移除最小相似度阈值检查，与v1保持一致
                max_sim = sim
                target_problem = p
        return target_problem

    def calculate_reward(self, content: str, solution: str) -> float:
        """计算综合奖励
        
        计算逻辑：
        1. 基础分数 = format_weight * format_score + accuracy_weight * accuracy_score
        2. 重复惩罚 = repetition_weight * repetition_score
        3. 最终分数 = 基础分数 * (1 + 重复惩罚)
        """
        format_score = self.format_reward.calculate(content)
        accuracy_score = self.accuracy_reward.calculate(content, solution)
        repetition_score = self.repetition_penalty.calculate(content)

        # 计算基础分数
        base_score = (
            self.config.format_weight * format_score +
            self.config.accuracy_weight * accuracy_score
        )

        # 应用重复惩罚（作为乘法因子）
        penalty_factor = 1 + (self.config.repetition_weight * repetition_score)
        
        return max(0.0, base_score * penalty_factor)  # 确保分数不会小于0

def get_response_from_query(q: str, response_prefix: str):
    """从查询中提取回答内容
    Args:
        q: 原始查询字符串
        response_prefix: 回答的前缀模式
    Returns:
        提取出的回答内容
    """
    ends_of_sentence = ["。", "<｜end▁of▁sentence｜>", "\n"]
    response = q[len(response_prefix):].strip()
    for eos in ends_of_sentence:
        response = response.split(eos)[0]
    return response

def get_template_response_from_query(q: str, template: str):
    """从查询中提取模板回答内容
    Args:
        q: 原始查询字符串
        template: 回答模板
    Returns:
        提取出的回答内容
    """
    template_pattern = re.compile(template, re.DOTALL)
    match = template_pattern.search(q)
    if match:
        return match.group(0)
    return ""

def create_app(config: VerifierConfig) -> Tuple[Flask, MathVerifier]:
    """创建Flask应用和验证器"""
    app = Flask(__name__)
    verifier = MathVerifier(config)

    @app.route("/get_reward", methods=["POST"])
    def get_reward():
        try:
            data = request.get_json()
            if not data or "query" not in data or "prompts" not in data:
                return jsonify({"error": "Invalid request format"}), 400

            rewards = []
            for query, problem in zip(data["query"], data["prompts"]):
                if not problem:
                    return jsonify({"error": f"Problem not found in query: {query}"}), 400

                if problem not in verifier.problem_to_answer:
                    problem = verifier.find_similar_problem(problem)
                    if not problem:
                        return jsonify({"error": f"No similar problem found"}), 400

                solution = verifier.problem_to_answer[problem]
                response = get_response_from_query(query, "答案：")
                template_response = get_template_response_from_query(query, r"答案：.*")
                reward = verifier.calculate_reward(response, solution)
                rewards.append(reward)

                # 随机打印日志
                if random.randint(1, 20) == 1:
                    print(f"Query: {query}\nProblem: {problem}\nSolution: {solution}\nReward: {reward}\n")

            return jsonify({"rewards": rewards})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app, verifier

def main():
    """主函数"""
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                       help="Datasets to use (comma separated)")
    parser.add_argument("--input-key", type=str, default="prompt",
                       help="The key name of prompt")
    parser.add_argument("--prompt-template", type=str, required=True,
                        help="Chat template format (chatml, qwen1, or base)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Server port")
    parser.add_argument("--format-weight", type=float, default=0.3,
                       help="Weight for format reward")
    parser.add_argument("--accuracy-weight", type=float, default=0.5,
                       help="Weight for accuracy reward")
    parser.add_argument("--repetition-weight", type=float, default=0.2,
                       help="Weight for repetition penalty")
    args = parser.parse_args()

    # 设置模板格式
    if args.prompt_template == "chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template == "qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template == "base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    else:
        raise ValueError(f"Unknown chat format: {args.prompt_template}")

    # 创建配置
    config = VerifierConfig(
        format_weight=args.format_weight,
        accuracy_weight=args.accuracy_weight,
        repetition_weight=args.repetition_weight
    )

    # 创建应用
    app, verifier = create_app(config)
    
    # 加载数据集
    dataset_paths = [p.strip() for p in args.dataset.split(',')]
    verifier.load_dataset(dataset_paths, args.input_key)
    print(f"Loaded {len(verifier.problem_to_answer)} problems from dataset")

    # 启动服务器
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
