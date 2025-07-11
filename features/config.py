dimensions = {
  "dimensions": [
    {
      "dimension_id": "emotional_intensity_control",
      "description": "情感表达的节制与自然度",
      "scoring_criteria": {
        "0_score": "情感表达极端化，过度使用积极或消极情感词，情感比例失衡，与语境不匹配",
        "1_score": "情感表达自然适度，平衡使用积极与消极情感词，具有语境意识",
      },
      "key_indicators": [
        "积极情感密度", 
        "消极情感密度",
        "情感平衡度",
        "情感词多样性",
        "高强度情感词占比",
        "语气助词分布",
        "特殊标点占比（例如！、？）"
      ],
      "liwc_dimensions": ["PosEmo", "NegEmo", "Affect"]
    },
    {
      "dimension_id": "personal_emotional_expression",
      "description": "情感与个人经历的结合度",
      "scoring_criteria": {
        "0_score": "缺乏个人情感表达，与自身经历脱节，流于形式和表面",
        "1_score": "情感表达紧密联系个人真实经历，反映了深度的自我觉察与反思",
      },
      "key_indicators": [
        "自我（第一人称代词）指涉频率",
        "情感多样性",
        "个人叙事深度",
        "感官、认知词语频度",
        "细节描写(语言、动作、神态、心理)数量",
        "自我反思、感悟"
      ],
      "liwc_dimensions": ["I", "EmoDiversity"]
    },
    {
      "dimension_id": "narrative_flexibility",
      "description": "叙述结构的灵活性与思维开放性",
      "scoring_criteria": {
        "0_score": "结构刻板/表达绝对化，缺乏思考过程和对其他观点的思考",
        "1_score": "自然展现思维过程，结构灵活留白得当，思考视角多元化，善用叙述技巧",
      },
      "key_indicators": [
        "思维开放性",
        "表达平衡度",
        "叙述流畅性",
        "因果逻辑词占比",
        "思维变化过程"
      ],
      "liwc_dimensions": ["Insight", "Tentat", "Certain", "Cause"]
    },
    {
      "dimension_id": "life_detail_vividness",
      "description": "生活细节的丰富性与真实感",
      "scoring_criteria": {
        "0_score": "描写高度抽象与概括，缺乏具体生活细节，空泛失真",
        "1_score": "细节丰富，多感官联动，真实还原生活场景，给人以画面感和沉浸感",
      },
      "key_indicators": [
        "多感官细节",
        "场景还原度",
        "生活元素密度",
        "俗语、俚语、口语词的运用",
        "时空方位词频",
        "感官词语",
        "具象名词"
      ],
      "liwc_dimensions": ["Percept", "Body", "Family", "Time"]
    },
    {
      "dimension_id": "vocabulary_personalization",
      "description": "用词的独特性和个人风格",
      "scoring_criteria": {
        "0_score": "用词大众化缺乏个人特色，文章前后风格各异",
        "1_score": "用词独特具有鲜明个人风格和偏好，上下文风格一致",
      },
      "key_indicators": [
        "风格化用词",
        "复杂词汇、低频词、特定领域词汇运用",
        "表达独特性",
        "词汇丰富度",
        "个人独创词",
        "反讽、用典、比喻等修辞手法运用"
      ],
      "liwc_dimensions": ["Sixltr", "Slang"]
    },
    {
      "dimension_id": "natural_language_flow",
      "description": "语言表达的自然程度",
      "scoring_criteria": {
        "0_score": "语言结构不自然不流畅，用词单一重复，上下文连接僵硬，错别字多，句子结构单一、重复，缺少必要的过渡词，存在大量不自然的停顿和重复",
        "1_score": "语言流畅符合自然对话特征，流畅且富有节奏感。能够娴熟地运用过渡词连接思想，句式长短结合、错落有致，整体语流一气呵成。",
      },
      "key_indicators": [
        "功能词比例",
        "实词比例",
        "语流自然度",
        "过渡词的运用",
        "词语重复率",
        "句式的变化",
        "语言得体"
      ],
      "liwc_dimensions": ["Function", "Content"]
    },
    {
      "dimension_id": "communication_naturalness",
      "description": "与读者交流的自然程度",
      "scoring_criteria": {
        "0_score": "缺乏读者互动意识，存在距离感，文字缺乏感染力与激情",
        "1_score": "自然建立读者互动关系，营造出一种亲切、平等的对话氛围",
      },
      "key_indicators": [
        "群体指涉（第二、第三人称）运用",
        "社交连接表达",
        "互动自然性",
        "特殊句式（疑问句、祈使句、设问句）频率"
      ],
      "liwc_dimensions": ["We", "Social"]
    },
    {
      "dimension_id": "humanized_error_level",
      "description": "语言错误的自然性",
      "scoring_criteria": {
        "0_score": "语言错误极端化（完全无错或错误过多），错误过于低级",
        "1_score": "语言错误符合自然表达特征，",
      },
      "key_indicators": [
        "错误分布",
        "错误密度",
        "错误自然度",
        "错误严重程度"
      ],
      "liwc_dimensions": ["Authentic"]
    }
  ]
} 