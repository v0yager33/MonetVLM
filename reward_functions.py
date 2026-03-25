"""
GRPO 奖励函数模块

奖励由两部分组成：
- 结果奖励（系数 1.0）：回复中 \\boxed{} 内的答案是否与目标一致
- 格式奖励（系数 0.1）：回复是否包含合规的 \\boxed{X} 格式
"""

import re

RESULT_REWARD_WEIGHT = 1.0
FORMAT_REWARD_WEIGHT = 0.1

# 匹配 \boxed{...} 中的内容
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_boxed_answer(text: str) -> str | None:
    """从回复中提取 \\boxed{} 内的答案，取最后一个匹配。"""
    matches = BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def compute_result_reward(content: str, target: str | None) -> float:
    """
    结果奖励：提取 \\boxed{} 中的答案，与目标精确匹配则得 1.0，否则 0.0。
    """
    if target is None:
        return 0.0

    predicted = _extract_boxed_answer(content)
    if predicted is None:
        return 0.0

    if predicted.upper().strip() == str(target).upper().strip():
        return 1.0

    return 0.0


def compute_format_reward(content: str) -> float:
    """
    格式奖励：回复中包含合规的 \\boxed{X} 格式则得 1.0，否则 0.0。
    仅检查格式是否存在，不校验内容正确性。
    """
    if BOXED_PATTERN.search(content):
        return 1.0
    return 0.0


def compute_reward(completions_text: list[str], target: str | None = None) -> list[float]:
    """
    计算一组 completion 的最终奖励分数。

    最终奖励 = 结果奖励 × 1.0 + 格式奖励 × 0.1

    Args:
        completions_text: G 个生成的回复文本
        target: 目标答案字母（如 "A"），用于结果匹配

    Returns:
        每个 completion 的最终总奖励
    """
    total_rewards = []

    for content in completions_text:
        result_score = compute_result_reward(content, target)
        format_score = compute_format_reward(content)

        total = RESULT_REWARD_WEIGHT * result_score + FORMAT_REWARD_WEIGHT * format_score
        total_rewards.append(total)

    return total_rewards