import numpy as np
import emoji
from collections import Counter, defaultdict
import re
import string
import math
import logging
from typing import Dict, Any, List
import base64, io
import requests
import json


try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    # 确保有句子分割器
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
except ImportError:
    nlp = None
    logging.warning("spaCy library not installed. Some features will be unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    model_path = '/data/Content_Moderation/all-MiniLM-L6-v2/'  # Updated to use local path
    sentence_transformer = SentenceTransformer(model_path)
except ImportError:
    sentence_transformer = None
    logging.warning("Sentence Transformers library not installed. Semantic coherence features will be disabled.")
except Exception as e:
    sentence_transformer = None
    logging.error(f"Failed to load sentence transformer model: {str(e)}")

# 配置日志
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# 内容词性标签
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def sentence_length(text: str) -> Dict[str, float]:
    """
    计算句子长度相关特征
    
    Args:
        text: 输入文本
        
    Returns:
        包含平均句子长度、最大句子长度、最小句子长度的字典
    """
    # 支持中英文标点符号的句子分割
    sentences = re.split(r'[.!?。！？]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return {
            'avg_sentence_length': 0.0,
            'max_sentence_length': 0.0,
            'min_sentence_length': 0.0,
            'sentence_length_std': 0.0
        }
    
    lengths = [len(s.split()) for s in sentences]
    
    return {
        'avg_sentence_length': np.mean(lengths),
        'max_sentence_length': max(lengths),
        'min_sentence_length': min(lengths),
        'sentence_length_std': np.std(lengths)
    }


def word_frequency(text: str) -> Dict[str, float]:
    """
    计算词频相关特征
    
    Args:
        text: 输入文本
        
    Returns:
        包含高频词比例、词频分布熵等特征的字典
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if not words:
        return {
            'high_freq_word_ratio': 0.0,
            'word_freq_entropy': 0.0,
            'unique_word_ratio': 0.0
        }
    
    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)
    
    # 计算高频词比例（出现次数>1的词）
    high_freq_words = sum(1 for count in word_counts.values() if count > 1)
    high_freq_ratio = high_freq_words / unique_words if unique_words > 0 else 0
    
    # 计算词频分布熵
    frequencies = np.array(list(word_counts.values()))
    probabilities = frequencies / total_words
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return {
        'high_freq_word_ratio': high_freq_ratio,
        'word_freq_entropy': entropy,
        'unique_word_ratio': unique_words / total_words
    }


def punctuation_ratio(text: str) -> Dict[str, float]:
    """
    计算标点符号相关特征
    
    Args:
        text: 输入文本
        
    Returns:
        包含标点符号比例、不同标点符号使用情况的字典
    """
    if not text:
        return {
            'punctuation_ratio': 0.0,
            'comma_ratio': 0.0,
            'period_ratio': 0.0,
            'question_ratio': 0.0,
            'exclamation_ratio': 0.0,
            'chinese_punct_ratio': 0.0
        }
    
    total_chars = len(text)
    
    # 中英文标点符号
    chinese_punct = '，。！？；：""''（）【】《》、'
    all_punct = string.punctuation + chinese_punct
    
    punctuation_count = sum(1 for char in text if char in all_punct)
    
    # 英文标点
    comma_count = text.count(',')
    period_count = text.count('.')
    question_count = text.count('?')
    exclamation_count = text.count('!')
    
    # 中文标点
    chinese_comma_count = text.count('，')
    chinese_period_count = text.count('。')
    chinese_question_count = text.count('？')
    chinese_exclamation_count = text.count('！')
    
    return {
        'punctuation_ratio': punctuation_count / total_chars,
        'comma_ratio': (comma_count + chinese_comma_count) / total_chars,
        'period_ratio': (period_count + chinese_period_count) / total_chars,
        'question_ratio': (question_count + chinese_question_count) / total_chars,
        'exclamation_ratio': (exclamation_count + chinese_exclamation_count) / total_chars,
        'chinese_punct_ratio': sum(1 for char in text if char in chinese_punct) / total_chars
    }


def emoji_frequency(text: str) -> Dict[str, float]:
    """
    计算emoji使用频率特征
    
    Args:
        text: 输入文本
        
    Returns:
        包含emoji使用频率相关特征的字典
    """
    if not text:
        return {
            'emoji_ratio': 0.0,
            'unique_emoji_ratio': 0.0,
            'emoji_density': 0.0
        }
    
    total_chars = len(text)
    emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    
    if not emoji_list:
        return {
            'emoji_ratio': 0.0,
            'unique_emoji_ratio': 0.0,
            'emoji_density': 0.0
        }
    
    emoji_counts = Counter(emoji_list)
    total_emojis = len(emoji_list)
    unique_emojis = len(emoji_counts)
    
    return {
        'emoji_ratio': total_emojis / total_chars,
        'unique_emoji_ratio': unique_emojis / total_emojis if total_emojis > 0 else 0,
        'emoji_density': total_emojis / len(text.split())  # 每个词中的emoji密度
    }


def avg_word_length(text: str) -> Dict[str, float]:
    """
    计算平均词长相关特征
    
    Args:
        text: 输入文本
        
    Returns:
        包含平均词长、词长标准差等特征的字典
    """
    words = re.findall(r'\b\w+\b', text)
    
    if not words:
        return {
            'avg_word_length': 0.0,
            'word_length_std': 0.0,
            'max_word_length': 0.0,
            'min_word_length': 0.0
        }
    
    word_lengths = [len(word) for word in words]
    
    return {
        'avg_word_length': np.mean(word_lengths),
        'word_length_std': np.std(word_lengths),
        'max_word_length': max(word_lengths),
        'min_word_length': min(word_lengths)
    }


def lexical_diversity(text: str) -> Dict[str, float]:
    """
    计算词汇多样性特征
    
    Args:
        text: 输入文本
        
    Returns:
        包含TTR、MTLD等词汇多样性指标的字典
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if not words:
        return {
            'ttr': 0.0,  # Type-Token Ratio
            'log_ttr': 0.0,
            'root_ttr': 0.0
        }
    
    unique_words = len(set(words))
    total_words = len(words)
    
    ttr = unique_words / total_words
    log_ttr = unique_words / np.log(total_words) if total_words > 1 else 0
    root_ttr = unique_words / np.sqrt(total_words)
    
    return {
        'ttr': ttr,
        'log_ttr': log_ttr,
        'root_ttr': root_ttr
    }


def repetition_patterns(text: str) -> Dict[str, float]:
    """
    计算重复模式特征
    检测文本中的各种重复模式
    
    Args:
        text: 输入文本
        
    Returns:
        包含重复模式特征的字典
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) < 4:
        return {
            'immediate_repetition': 0.0,
            'phrase_repetition': 0.0,
            'pattern_regularity': 0.0,
            'self_repetition_ratio': 0.0
        }
    
    # 计算直接重复（相邻词重复）
    immediate_reps = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
    immediate_repetition = immediate_reps / (len(words) - 1)
    
    # 计算短语重复（2-3词组合重复）
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)
    
    repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
    repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
    
    phrase_repetition = (repeated_bigrams + repeated_trigrams) / (len(bigrams) + len(trigrams))
    
    # 计算模式规律性
    word_distances = defaultdict(list)
    for i, word in enumerate(words):
        word_distances[word].append(i)
    
    regular_patterns = 0
    total_patterns = 0
    
    for word, positions in word_distances.items():
        if len(positions) >= 3:
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            gap_std = np.std(gaps)
            if gap_std < 2:  # 间隔比较规律
                regular_patterns += 1
            total_patterns += 1
    
    pattern_regularity = regular_patterns / total_patterns if total_patterns > 0 else 0.0
    
    # 计算自重复比例
    unique_words = len(set(words))
    total_words = len(words)
    self_repetition_ratio = 1 - (unique_words / total_words)
    
    return {
        'immediate_repetition': immediate_repetition,
        'phrase_repetition': phrase_repetition,
        'pattern_regularity': pattern_regularity,
        'self_repetition_ratio': self_repetition_ratio
    } 

def semantic_coherence(text: str) -> Dict[str, float]:
    """
    计算语义连贯性特征
    使用简化的方法评估文本的语义连贯性
    
    Args:
        text: 输入文本
        
    Returns:
        包含语义连贯性特征的字典
    """
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return {
            'lexical_cohesion': 0.0,
            'sentence_similarity': 0.0,
            'topic_consistency': 0.0
        }
    
    # 计算词汇衔接度
    sentence_words = []
    for sentence in sentences:
        words = set(re.findall(r'\b\w+\b', sentence.lower()))
        sentence_words.append(words)
    
    # 相邻句子的词汇重叠度
    overlaps = []
    for i in range(len(sentence_words) - 1):
        intersection = len(sentence_words[i] & sentence_words[i+1])
        union = len(sentence_words[i] | sentence_words[i+1])
        overlap = intersection / union if union > 0 else 0
        overlaps.append(overlap)
    
    lexical_cohesion = np.mean(overlaps) if overlaps else 0.0
    
    # 计算句子相似性（基于词汇重叠）
    similarities = []
    for i in range(len(sentence_words)):
        for j in range(i+1, len(sentence_words)):
            intersection = len(sentence_words[i] & sentence_words[j])
            union = len(sentence_words[i] | sentence_words[j])
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
    
    sentence_similarity = np.mean(similarities) if similarities else 0.0
    
    # 计算主题一致性（基于高频词分布）
    all_words = []
    for words in sentence_words:
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    top_words = set([word for word, count in word_counts.most_common(10)])
    
    topic_scores = []
    for words in sentence_words:
        topic_score = len(words & top_words) / len(words) if words else 0
        topic_scores.append(topic_score)
    
    topic_consistency = 1 - np.std(topic_scores) if len(topic_scores) > 1 else 0.0
    
    return {
        'lexical_cohesion': lexical_cohesion,
        'sentence_similarity': sentence_similarity,
        'topic_consistency': max(0, topic_consistency)  # 确保非负
    }

def calculate_burstiness(text: str) -> Dict[str, float]:
    """
    计算文本的突发性特征
    突发性衡量词汇使用的不均匀程度
    
    Args:
        text: 输入文本
        
    Returns:
        包含突发性相关特征的字典
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) < 10:  # 文本太短无法计算突发性
        return {
            'word_burstiness': 0.0,
            'sentence_burstiness': 0.0,
            'avg_gap_variance': 0.0
        }
    
    # 计算词汇突发性
    word_positions = defaultdict(list)
    for i, word in enumerate(words):
        word_positions[word].append(i)
    
    burstiness_scores = []
    gap_variances = []
    
    for word, positions in word_positions.items():
        if len(positions) >= 3:  # 至少出现3次才计算突发性
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            mean_gap = np.mean(gaps)
            var_gap = np.var(gaps)
            
            if mean_gap > 0:
                burstiness = (var_gap - mean_gap) / (var_gap + mean_gap)
                burstiness_scores.append(burstiness)
                gap_variances.append(var_gap)
    
    word_burstiness = np.mean(burstiness_scores) if burstiness_scores else 0.0
    avg_gap_variance = np.mean(gap_variances) if gap_variances else 0.0
    
    # 计算句子长度突发性
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) >= 3:
        sentence_lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(sentence_lengths)
        var_len = np.var(sentence_lengths)
        
        sentence_burstiness = (var_len - mean_len) / (var_len + mean_len) if (var_len + mean_len) > 0 else 0.0
    else:
        sentence_burstiness = 0.0
    
    return {
        'word_burstiness': word_burstiness,
        'sentence_burstiness': sentence_burstiness,
        'avg_gap_variance': avg_gap_variance
    }



def syntactic_complexity(text: str) -> Dict[str, float]:
    """
    计算句法复杂度特征
    
    Args:
        text: 输入文本
        
    Returns:
        包含句法复杂度特征的字典:
        - max_syntax_depth: 最大依存句法树深度
        - avg_branch_factor: 平均分支因子
    """
    if not text.strip() or nlp is None:
        return {
            'max_syntax_depth': 0.0,
            'avg_branch_factor': 0.0
        }
    
    try:
        doc = nlp(text)
        depths = []
        branches = []
        

        for sent in doc.sents:
            sent_doc = nlp(sent.text)
            
            # 计算依存深度
            for token in sent_doc:
                # 计算从当前词到根节点的距离
                depth = 0
                head = token.head
                while head != token and head != head.head:  # 防止无限循环
                    depth += 1
                    head = head.head
                depths.append(depth)
            

            for token in sent_doc:
                if token.pos_ in CONTENT_POS:
                    children = list(token.children)
                    if children:
                        branches.append(len(children))
        

        max_depth = max(depths) if depths else 0
        avg_branches = np.mean(branches) if branches else 0
        
        return {
            'max_syntax_depth': float(max_depth),
            'avg_branch_factor': float(avg_branches)
        }
    
    except Exception as e:
        logger.error(f"Syntactic complexity calculation error: {str(e)}")
        return {
            'max_syntax_depth': 0.0,
            'avg_branch_factor': 0.0
        }

"""
def semantic_coherence(text: str) -> Dict[str, float]:
    \"""
    计算语义一致性特征
    
    Args:
        text: 输入文本
        
    Returns:
        包含语义一致性特征的字典:
    \"""
    if not text.strip() or sentence_transformer is None:
        return {'semantic_variance': 0.0}
    
    try:
        # 分割句子 - 使用简单方法避免spaCy依赖
        sentences = [s.strip() for s in re.split(r'[.!?。！？\n]+', text.strip()) if len(s.strip()) > 5]
        
        if len(sentences) < 2:
            return {'semantic_variance': 0.0}
        
        # 生成句子嵌入
        embeddings = sentence_transformer.encode(sentences)
        
        # 计算相邻句子余弦相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            dot_product = np.dot(embeddings[i], embeddings[i+1])
            norm_a = np.linalg.norm(embeddings[i])
            norm_b = np.linalg.norm(embeddings[i+1])

            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
                similarities.append(similarity)
        
        # 计算相似度方差
        if len(similarities) > 1:
            variance = np.var(similarities)
        else:
            variance = 0.0
        
        return {'semantic_variance': float(variance)}
    
    except Exception as e:
        logger.error(f"Semantic coherence calculation error: {str(e)}")
        return {'semantic_variance': 0.0}
"""

def information_density(text: str) -> Dict[str, float]:
    """
    计算熵与信息密度特征
    
    Args:
        text: 输入文本
        
    Returns:
        包含信息密度特征的字典
    """
    if not text.strip():
        return {
            'char_entropy': 0.0,
            'content_word_ratio': 0.0
        }
    
    # 字符级熵
    try:
        char_counts = Counter(text)
        total_chars = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        char_entropy = entropy
    except Exception as e:
        logger.error(f"Character entropy calculation error: {str(e)}")
        char_entropy = 0.0
    
    # 内容词比例 
    try:
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {
                'char_entropy': float(char_entropy),
                'content_word_ratio': 0.0
            }
        
        # 简单的内容词识别
        content_words = set([
            'noun', 'verb', 'adj', 'adv', 
            'nouns', 'verbs', 'adjectives', 'adverbs',
        ])
        content_count = sum(1 for word in words if word in content_words)
        content_ratio = content_count / len(words)
    except Exception as e:
        logger.error(f"Content word ratio calculation error: {str(e)}")
        content_ratio = 0.0
    
    return {
        'char_entropy': float(char_entropy),
        'content_word_ratio': float(content_ratio)
    }
