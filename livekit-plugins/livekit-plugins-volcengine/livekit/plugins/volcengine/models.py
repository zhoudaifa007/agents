from typing import Literal

# Volcengine supported voice types (refer to https://www.volcengine.com/docs/6561/1257544)
VolcengineVoiceTypes = Literal[
    "BV001_STREAMING",  # 普通女声
    "BV002_STREAMING",  # 普通男声
    "BV003_STREAMING",  # 温柔女声
    "BV004_STREAMING",  # 温柔男声
    # Add more voice types as needed
]

# Volcengine supported languages
VolcengineLanguages = Literal[
    "zh-CN",  # 普通话
    "zh-HK",  # 粤语
    "en-US",  # 英语（美式）
    "en-GB",  # 英语（英式）
    "ja-JP",  # 日语
    "ko-KR",  # 韩语
    "fr-FR",  # 法语
    "de-DE",  # 德语
    "es-ES",  # 西班牙语
    # Add more languages as needed
]