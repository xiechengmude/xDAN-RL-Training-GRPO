"""
测试math_verifier_v2的功能
"""

from math_verifier_v2 import create_app, VerifierConfig

def test_basic_functionality():
    """测试基本功能"""
    # 创建默认配置
    config = VerifierConfig()
    
    # 创建应用和验证器
    app, verifier = create_app(config)
    
    # 添加测试数据
    test_cases = [
        {
            "prompt": "What is 2 + 2?",
            "solution": "$4$",
            "response": "<think> 2 + 2 = 4 </think><answer> $\\boxed{4}$ </answer>",
            "expected_score": 1.0,
            "description": "基础正确性测试：格式正确，答案正确，无重复内容"
        },
        {
            "prompt": "What is the integral of x²?",
            "solution": "$\\frac{x^3}{3} + C$",
            "response": "<think> The integral of x² is x³/3 + C </think><answer> $\\boxed{\\frac{x^3}{3} + C}$ </answer>",
            "expected_score": 1.0,
            "description": "复杂表达式测试：格式正确，答案正确，无重复内容"
        },
        {
            "prompt": "What is 2 + 2?",
            "solution": "$4$",
            "response": "The answer is 4",  # 格式错误
            "expected_score": 0.0,
            "description": "格式错误测试：完全不符合格式要求"
        },
        {
            "prompt": "What is 2 + 2?",
            "solution": "$4$",
            "response": "<think> 2 + 2 = 5 </think><answer> $\\boxed{5}$ </answer>",  # 答案错误
            "expected_score": 0.2,
            "description": "答案错误测试：格式正确（得0.2分），答案错误（得0分）"
        },
        {
            "prompt": "What is 2 + 2?",
            "solution": "$4$",
            "response": "<think>2 + 2 = 4</think><answer>$\\boxed{4}$</answer>",  # 无空格也是合法的
            "expected_score": 1.0,
            "description": "格式变体测试：无空格也是合法的格式"
        },
        {
            "prompt": "What is 2 + 2?",
            "solution": "$4$",
            "response": "<think>2 + 2 = 4, 2 + 2 = 4, 2 + 2 = 4, let me repeat: 2 + 2 = 4</think><answer>$\\boxed{4}$</answer>",
            "expected_score": 0.7,  # 基础分1.0，但因重复内容受到30%的惩罚
            "description": "重复内容测试：答案正确但存在严重重复内容，扣除30%"
        },
        {
            "prompt": "What is 2 + 2?",
            "solution": "$4$",
            "response": "<think>2 + 2 = 4</think><answer>$\\boxed{4}$</answer><think>Let me think again</think><answer>The answer is 4</answer>",
            "expected_score": 0.7,  # 基础分1.0，但因重复格式标签受到30%的惩罚
            "description": "重复格式测试：重复使用think和answer标签，扣除30%"
        },
        {
            "prompt": "What is 2 + 2?",
            "solution": "$4$",
            "response": "<think>2 + 2 = 4, let me verify</think><answer>$\\boxed{4}$</answer>",
            "expected_score": 1.0,  # 轻微重复不扣分
            "description": "轻微重复测试：内容稍有重复但在可接受范围内"
        }
    ]
    
    # 测试每个用例
    print("\n开始测试 math_verifier_v2...")
    print("=" * 50)

    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {case['description']}")
        print(f"提示: {case['prompt']}")
        print(f"回答: {case['response']}")
        
        # 获取各个组件的分数
        format_score = verifier.format_reward.calculate(case['response'])
        accuracy_score = verifier.accuracy_reward.calculate(case['response'], case['solution'])
        repetition_score = verifier.repetition_penalty.calculate(case['response'])
        
        # 计算总分
        actual_score = (
            config.format_weight * format_score +
            config.accuracy_weight * accuracy_score +
            config.repetition_weight * repetition_score
        )
        
        print(f"\n分数明细:")
        print(f"格式分数: {format_score} * {config.format_weight} = {format_score * config.format_weight}")
        print(f"准确性分数: {accuracy_score} * {config.accuracy_weight} = {accuracy_score * config.accuracy_weight}")
        print(f"重复惩罚: {repetition_score} * {config.repetition_weight} = {repetition_score * config.repetition_weight}")
        print(f"总分: {actual_score}")
        
        print(f"\n期望分数: {case['expected_score']}")
        print(f"实际分数: {actual_score}")
        print(f"结果: {'通过' if abs(actual_score - case['expected_score']) < 0.1 else '失败'}")

    print("\n测试完成!")

if __name__ == "__main__":
    test_basic_functionality()
