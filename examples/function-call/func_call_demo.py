from typing import List, Dict
import random
import json
import openai


def get_weather_and_recommendation(
        city: str,
        date: str,
        temperature_unit: str = "Celsius",
        preferred_conditions: List[str] = ["clear", "cloudy"],
        min_temperature: float = None,
        max_temperature: float = None
) -> Dict[str, any]:
    """
    查询城市的天气，并根据用户偏好推荐活动。

    参数:
        city (str): 查询天气的城市。
        date (str): 查询的日期（格式：YYYY-MM-DD）。
        temperature_unit (str): 温度单位（Celsius 或 Fahrenheit）。
        preferred_conditions (List[str]): 用户偏好的天气状况。
        min_temperature (float): 用户可接受的最低温度。
        max_temperature (float): 用户可接受的最高温度。

    返回:
        Dict[str, any]: 包含天气信息和推荐活动的结果。
    """

    # 模拟天气信息（实际应用中，这里会调用第三方 API 获取数据）
    simulated_weather = {
        "city": city,
        "date": date,
        "temperature": round(random.uniform(-10, 35), 2),
        "condition": random.choice(["clear", "cloudy", "rain", "snow", "storm"]),
    }

    # 转换温度单位
    temperature = simulated_weather["temperature"]
    if temperature_unit == "Fahrenheit":
        temperature = round((temperature * 9 / 5) + 32, 2)

    # 根据用户的条件推荐活动
    recommendations = []
    if simulated_weather["condition"] in preferred_conditions:
        if min_temperature is not None and temperature < min_temperature:
            recommendations.append("建议穿保暖衣物")
        elif max_temperature is not None and temperature > max_temperature:
            recommendations.append("适合户外活动，例如散步或骑行")
        else:
            recommendations.append("天气适中，可以安排一些室外活动")

    # 如果天气不理想，给出其他建议
    if "rain" in simulated_weather["condition"] or "storm" in simulated_weather["condition"]:
        recommendations.append("建议在室内阅读或观看电影")

    # 返回结构化的天气信息和推荐活动
    return {
        "city": simulated_weather["city"],
        "date": simulated_weather["date"],
        "temperature": temperature,
        "unit": temperature_unit,
        "condition": simulated_weather["condition"],
        "recommendations": recommendations,
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_and_recommendation",
            "description": "查询城市的天气，并根据用户偏好推荐活动。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "查询天气的城市"},
                    "date": {"type": "string", "description": "查询的日期（格式：YYYY-MM-DD）"},
                    "temperature_unit": {
                        "type": "string",
                        "enum": ["Celsius", "Fahrenheit"],
                        "description": "温度单位",
                    },
                    "preferred_conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "用户偏好的天气状况",
                    },
                    "min_temperature": {
                        "type": "number",
                        "description": "用户可接受的最低温度",
                    },
                    "max_temperature": {
                        "type": "number",
                        "description": "用户可接受的最高温度",
                    },
                },
                "required": ["city", "date"],
            }
        }
    }
]

messages = [
    {"role": "system", "content": "你是一个智能助手，能够查询天气并推荐活动。"},
    {"role": "user", "content": "请帮我查询 2024-10-28 在北京的天气，并告诉我适合做什么活动。"},
]

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)

# 检查模型的回复并执行函数
response_message = response.choices[0].message

tool_call = response.choices[0].message.tool_calls[0]


# 检查是否有函数调用
if tool_call:

    arguments = json.loads(tool_call.function.arguments)

    function_name = tool_call.function.name

    # 调用相应的函数
    if function_name == "get_weather_and_recommendation":
        result = get_weather_and_recommendation(**arguments)

        # 将函数调用结果添加到对话中
        messages.append(response_message)  # 添加函数调用信息
        messages.append({"role": "function", "name": function_name, "content": json.dumps(result)})

        # 获取最终回复
        final_response = openai.chat.completions.create(
            model="gpt-4-0613",
            messages=messages
        )

        # 输出最终回复
        print(final_response.choices[0].message["content"])
