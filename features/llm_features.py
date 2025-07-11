import re
import math
from typing import Dict, List, Any
import numpy as np
import base64, io
import requests
import json

# 导入移出的函数和配置
from .llm_client import llm, llm_ali
from .config import dimensions


def llm_score(text):
    """
    对文本进行多维度评分，每次评估一个维度
    
    Args:
        text: 待评估的文本
        
    Returns:
        包含各维度得分的字典
    """
    
    scores_dict = {}
    
    # 为每个维度创建单独的评分任务
    for dim in dimensions["dimensions"]:
        dimension_id = dim["dimension_id"]
        
        # 为单个维度创建评分提示
        single_dim_prompt = f"""[评估维度]
维度ID: {dimension_id}
维度描述: {dim["description"]}

[评分标准]
0分标准: {dim["scoring_criteria"]["0_score"]}
1分标准: {dim["scoring_criteria"]["1_score"]}

[关键指标]
{', '.join(dim["key_indicators"])}

[用户指令]
请根据上述维度标准，对以下文本进行评分。得分为0-1之间的浮点数字。
请直接输出一个数字，不需要任何解释或说明。

[待评估文本]
{text}

[输出格式]
仅输出一个0-1之间的数字，例如：0.7
"""
        
        # 尝试获取该维度的评分
        max_retries = 3
        retry_count = 0
        score = 0.5  # 默认分数
        
        while retry_count < max_retries:
            try:
                res = llm_ali(single_dim_prompt)
                
                # 清理响应文本，提取数字
                cleaned_res = res.strip()
                
                # 尝试直接转换为浮点数
                try:
                    score = float(cleaned_res)
                    # 确保分数在0-1范围内
                    score = max(0.0, min(1.0, score))
                    break
                except ValueError:
                    # 如果直接转换失败，尝试提取数字
                    numbers = re.findall(r'[0-9]*\.?[0-9]+', cleaned_res)
                    if numbers:
                        score = float(numbers[0])
                        score = max(0.0, min(1.0, score))
                        break
                    else:
                        raise ValueError("无法从响应中提取有效数字")
                        
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"维度 {dimension_id} 评分失败，使用默认分数 0.5")
                    score = 0.5
                    break
        
        scores_dict[dimension_id] = score
    
    return scores_dict





