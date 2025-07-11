dimensions = {
  "dimensions": [
    {
      "dimension_id": "emotional_intensity_control",
      "description": "情感表达的节制与自然度",
      "scoring_criteria": {
        "0_score": "情感表达极端化，过度使用积极或消极情感词",
        "1_score": "情感表达自然适度，平衡使用积极与消极情感词",
      },
      "key_indicators": [
        "积极情感密度", 
        "消极情感密度",
        "情感平衡度"
      ],
      "liwc_dimensions": ["PosEmo", "NegEmo", "Affect"]
    },
    {
      "dimension_id": "personal_emotional_expression",
      "description": "情感与个人经历的结合度",
      "scoring_criteria": {
        "0_score": "缺乏个人情感表达，与自身经历脱节",
        "1_score": "情感表达紧密联系个人真实经历",
      },
      "key_indicators": [
        "自我指涉频率",
        "情感多样性",
        "个人叙事深度"
      ],
      "liwc_dimensions": ["I", "EmoDiversity"]
    },
    {
      "dimension_id": "narrative_flexibility",
      "description": "叙述结构的灵活性与思维开放性",
      "scoring_criteria": {
        "0_score": "结构刻板/表达绝对化，缺乏思考过程",
        "1_score": "自然展现思维过程，结构灵活留白得当",
      },
      "key_indicators": [
        "思维开放性",
        "表达平衡度",
        "叙述流畅性"
      ],
      "liwc_dimensions": ["Insight", "Tentat", "Certain", "Cause"]
    },
    {
      "dimension_id": "life_detail_vividness",
      "description": "生活细节的丰富性与真实感",
      "scoring_criteria": {
        "0_score": "缺乏具体生活细节，描写空泛失真",
        "1_score": "细节丰富多感官联动，真实还原生活场景",
      },
      "key_indicators": [
        "多感官细节",
        "场景还原度",
        "生活元素密度"
      ],
      "liwc_dimensions": ["Percept", "Body", "Family", "Time"]
    },
    {
      "dimension_id": "vocabulary_personalization",
      "description": "用词的独特性和个人风格",
      "scoring_criteria": {
        "0_score": "用词大众化缺乏个人特色",
        "1_score": "用词独特具有鲜明个人风格",
      },
      "key_indicators": [
        "风格化用词",
        "复杂词汇运用",
        "表达独特性"
      ],
      "liwc_dimensions": ["Sixltr", "Slang"]
    },
    {
      "dimension_id": "natural_language_flow",
      "description": "语言表达的自然程度",
      "scoring_criteria": {
        "0_score": "语言结构不自然不流畅",
        "1_score": "语言流畅符合自然对话特征",
      },
      "key_indicators": [
        "功能词比例",
        "实词比例",
        "语流自然度"
      ],
      "liwc_dimensions": ["Function", "Content"]
    },
    {
      "dimension_id": "communication_naturalness",
      "description": "与读者交流的自然程度",
      "scoring_criteria": {
        "0_score": "缺乏读者互动意识",
        "1_score": "自然建立读者互动关系",
      },
      "key_indicators": [
        "群体指涉运用",
        "社交连接表达",
        "互动自然性"
      ],
      "liwc_dimensions": ["We", "Social"]
    },
    {
      "dimension_id": "humanized_error_level",
      "description": "语言错误的自然性",
      "scoring_criteria": {
        "0_score": "语言错误极端化（完全无错或错误过多）",
        "1_score": "语言错误符合自然表达特征",
      },
      "key_indicators": [
        "错误分布",
        "错误密度",
        "错误自然度"
      ],
      "liwc_dimensions": ["Authentic"]
    }
  ]
} 