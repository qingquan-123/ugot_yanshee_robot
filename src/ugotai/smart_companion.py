# -*- coding: utf-8 -*-
"""
smart_companion.py - 智能陪伴完整流程（完全独立，不依赖 ugot 目录下其他模块）
流程：开场 → N次陪伴对话（大模型回答+实时天气）→ 显示表情 → 模仿音效 → 呼吸灯
命令行用法：
  python smart_companion.py <chat_times> <emotion> <sound> <light_color> <light_duration> <repeat>
示例：
  python smart_companion.py 2 愤怒 "['大象','老虎']" 红色 3 False
  python smart_companion.py 2 微笑 大象 蓝色 3 True
"""

import sys
import os
import time
import ast
import re
import random

# ===== SSL 证书补丁（必须在 import qianfan 之前）=====
try:
    import ssl
    import certifi
    def patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cafile=certifi.where())
        return ctx
    ssl.create_default_context = patched_create_default_context
except ImportError:
    pass

# ===== 千帆大模型配置 =====
from api_config import *
import qianfan
os.environ["QIANFAN_ACCESS_KEY"] = QIANFAN_ACCESS_KEY
os.environ["QIANFAN_SECRET_KEY"] = QIANFAN_SECRET_KEY
MODEL = "DeepSeek-V4-Flash"
chat_comp = qianfan.ChatCompletion(model=MODEL)

# ===== requests（天气查询）=====
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[警告] requests 未安装，实时天气功能不可用")

from ugot import ugot

# ============================================================
#  配置常量
# ============================================================
ROBOT_IP = "10.233.104.242"
DEFAULT_CITY = "杭州"           # 默认天气查询城市
LLM_TIMEOUT = 8                 # 大模型超时秒数
WEATHER_TIMEOUT = 5             # 天气接口超时秒数
ASR_TIMEOUT = 15                 # 语音识别超时秒数（循环调用，给用户足够时间说话）
EMOTION_HOLD_TIME = 2.5         # 表情保持秒数

# 大模型系统提示词：强制简短回答
SYSTEM_PROMPT = """你是一个陪伴机器人，回答必须遵守以下规则：
1. 回答必须简短，1-2句话，不超过40个字
2. 口语化，适合语音播报，不要用书面语
3. 不要解释、不要列举、不要说"作为AI"、不要说"我无法"
4. 如果用户问天气，直接基于提供的天气数据用一句话回答
5. 如果用户讲笑话，你也回一个超短笑话
6. 如果用户自我介绍，你也做一句简短自我介绍
7. 如果用户问问题你不知道，就说"这个我不太清楚呢"然后反问一个轻松的问题
"""

# 兜底回答（大模型超时/失败时随机选一个）
FALLBACK_ANSWERS = [
    "这个问题有点难，不过开心最重要啦！",
    "哈哈你说的真有趣，我们换个话题吧？",
    "我正在思考人生呢，你觉得今天天气怎么样？",
    "这个我得想想，要不我给你讲个笑话吧！",
    "嗯嗯我听到了，你继续说，我在认真听呢。",
    "哇这个问题好有深度，我觉得跟着感觉走就对了！",
]

# ============================================================
#  映射表
# ============================================================
EMOTION_MAP = {
    "愤怒": "Anger", "生气": "Anger", "怒": "Anger",
    "微笑": "Smile", "高兴": "Smile", "开心": "Smile", "快乐": "Smile", "笑": "Smile",
    "疑惑": "Doubt", "疑问": "Doubt", "困惑": "Doubt",
    "搜索": "Search",
    "呼吸": "Breathe",
    "眨眼": "Blink",
    "抵抗": "Resist",
    "爱心": "Love", "爱": "Love", "比心": "Love",
    "骄傲": "Proud", "自豪": "Proud",
    "挠痒痒": "Ticklish",
    "虚弱": "Weakness",
    "睡眠": "Sleep", "睡觉": "Sleep", "困": "Sleep",
    "苏醒": "WakeUp", "醒来": "WakeUp",
}

SOUND_MAP = {
    "熊": "bear", "鸟": "bird", "鸡": "chicken", "牛": "cow",
    "狗": "dog", "小狗": "dog", "大象": "elephant", "长颈鹿": "giraffe",
    "马": "horse", "狮子": "lion", "猴子": "monkey", "猪": "pig",
    "老虎": "tiger", "大老虎": "tiger", "猫": "cat", "小猫": "cat",
    "鸭": "duck", "鸭子": "duck", "羊": "sheep", "绵羊": "sheep",
    "救护车": "ambulance", "警车": "police_car_1", "汽车喇叭": "carhorn",
    "门铃": "doorbell", "机器人": "robot", "引擎": "engine", "激光": "laser",
    "高兴": "happy", "生气": "angry", "惊讶": "surprise", "疑问": "doubt",
    "完成": "complete", "收到": "received", "遵命": "yes",
}

LIGHT_COLOR_MAP = {
    "红色": [255, 0, 0], "红": [255, 0, 0],
    "蓝色": [0, 0, 255], "蓝": [0, 0, 255],
    "绿色": [0, 255, 0], "绿": [0, 255, 0],
}

# 常见城市列表（用于从用户话中提取城市名）
KNOWN_CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都",
    "重庆", "西安", "东莞", "苏州", "天津", "长沙", "青岛", "大连",
    "厦门", "宁波", "无锡", "济南", "郑州", "合肥", "福州", "昆明",
    "哈尔滨", "沈阳", "长春", "石家庄", "太原", "南昌", "贵阳", "南宁",
    "兰州", "乌鲁木齐", "呼和浩特", "拉萨", "银川", "西宁", "海口", "三亚",
]


# ============================================================
#  工具函数
# ============================================================
def show_text_on_screen(got, text):
    """屏幕显示文字（自动截断，避免溢出）"""
    try:
        display = text[:30] if len(text) > 30 else text
        got.screen_clear()
        got.screen_display_background(1)
        got.screen_print_text(display, 0)
    except Exception as e:
        print(f"  屏幕显示失败: {e}")


def parse_sound_param(sound_str):
    """
    解析音效参数：
    - 单个字符串如 "大象" → ["大象"]
    - 列表字符串如 "['大象','老虎']" → ["大象", "老虎"]
    """
    sound_str = sound_str.strip()
    if sound_str.startswith("["):
        try:
            result = ast.literal_eval(sound_str)
            if isinstance(result, list):
                return [str(s) for s in result if s]
        except Exception:
            pass
        return [sound_str]
    return [sound_str]


def is_weather_question(text):
    """判断用户是否在问天气"""
    weather_keywords = [
        "天气", "气温", "温度", "下雨", "晴天", "阴天", "多云",
        "刮风", "大风", "冷不冷", "热不热", "冷吗", "热吗",
        "穿什么", "带伞", "下雪", "雪", "雾霾", "空气质量",
    ]
    return any(k in text for k in weather_keywords)


def extract_city(text):
    """从用户话中提取城市名，找不到返回默认城市"""
    for city in KNOWN_CITIES:
        if city in text:
            return city
    return DEFAULT_CITY


def get_weather(city):
    """
    调用免费天气接口 wttr.in 获取实时天气
    返回格式如 "Sunny +28°C 65% 12km/h"，失败返回 None
    """
    if not REQUESTS_AVAILABLE:
        return None
    try:
        url = f"https://wttr.in/{city}?format=%C+%t+%h+%w"
        resp = requests.get(url, timeout=WEATHER_TIMEOUT)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text.strip()
    except Exception as e:
        print(f"  天气查询失败: {e}")
    return None


def generate_answer(user_text, chat_history):
    """
    生成陪伴回答：
    - 天气问题 → 先查实时天气，再让大模型组织语言
    - 普通问题 → 直接大模型回答
    - 大模型超时/失败 → 随机兜底回答
    """
    # 1. 天气问题特殊处理
    if is_weather_question(user_text):
        city = extract_city(user_text)
        print(f"  检测到天气问题，查询城市: {city}")
        weather = get_weather(city)
        if weather:
            print(f"  实时天气: {weather}")
            user_prompt = (
                f"用户问：{user_text}\n"
                f"实时天气数据：{city} {weather}\n"
                f"请用一句话口语化地告诉用户天气情况，不要念原始数据，要说人话。"
            )
        else:
            user_prompt = (
                f"用户问：{user_text}\n"
                f"暂时无法获取实时天气，请用一句话告诉用户你暂时查不到天气，让用户看手机。"
            )
    else:
        user_prompt = user_text

    # 2. 构建消息（系统提示 + 历史 + 当前问题）
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    # 3. 调用大模型
    try:
        resp = chat_comp.do(
            messages=messages,
            top_p=0.8,
            temperature=0.7,
            penalty_score=1.0,
            request_timeout=LLM_TIMEOUT,
        )
        answer = resp["result"].strip()
        # 去掉可能的引号
        answer = answer.strip('"').strip("'")
        # 截断过长回答
        if len(answer) > 50:
            answer = answer[:50] + "。"
        print(f"  大模型回答: {answer}")
        return answer
    except Exception as e:
        print(f"  大模型调用失败: {e}，使用兜底回答")
        return random.choice(FALLBACK_ANSWERS)


def listen_user(got, timeout=ASR_TIMEOUT):
    """
    循环调用语音识别，直到识别到文字或超时。
    start_audio_asr() 是非阻塞的，必须循环调用，否则用户来不及说话。
    返回识别到的文字，超时返回 None
    """
    print(f"  开始聆听（超时{timeout}秒）...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = got.start_audio_asr()
            if result and result.strip():
                elapsed = time.time() - start_time
                print(f"  识别成功（用时{elapsed:.1f}秒）：{result.strip()}")
                return result.strip()
        except Exception as e:
            print(f"  语音识别异常: {e}")
        time.sleep(0.3)
    print(f"  聆听超时（{timeout}秒），未识别到语音")
    return None


# ============================================================
#  主流程
# ============================================================
def companion_full_flow(got, chat_times, emotion, sound_list, light_color, light_duration, repeat):
    """
    智能陪伴完整流程
    参数：
      got: ugot 机器人实例
      chat_times: 对话次数
      emotion: 表情名（中文）
      sound_list: 音效名列表（中文）
      light_color: 呼吸灯颜色（中文）
      light_duration: 呼吸灯时长（秒）
      repeat: 是否重复回答
    返回：True=成功, False=失败
    """
    print("\n" + "=" * 60)
    print(f" 智能陪伴流程：对话{chat_times}次 | 表情[{emotion}] | "
          f"音效{sound_list} | {light_color}呼吸灯{light_duration}秒 | 重复:{repeat}")
    print("=" * 60)

    try:
        # ===== 1. 开场 =====
        print("\n===== 1. 开场 =====")
        try:
            got.play_audio_tts("你好，我是你的陪伴机器人，我们开始聊天吧", 0, True)
        except:
            pass
        show_text_on_screen(got, "陪伴模式")
        time.sleep(1)

        # ===== 2. 陪伴对话 =====
        chat_history = []
        for i in range(chat_times):
            print(f"\n===== 2.{i+1} 第 {i+1}/{chat_times} 次对话 =====")

            # 提示用户说话
            try:
                got.play_audio_tts(f"第{i+1}次对话，你说", 0, True)
            except:
                pass
            time.sleep(0.3)

            # 识别用户说话
            user_text = listen_user(got)
            if not user_text:
                print("  未识别到用户说话，跳过本轮")
                try:
                    got.play_audio_tts("没有听到你说话，我们继续吧", 0, True)
                except:
                    pass
                continue

            print(f"  用户说：{user_text}")
            show_text_on_screen(got, user_text)

            # 生成回答
            answer = generate_answer(user_text, chat_history)

            # 语音播报回答
            try:
                got.play_audio_tts(answer, 0, True)
            except Exception as e:
                print(f"  语音播报失败: {e}")

            # 屏幕显示回答
            show_text_on_screen(got, answer)

            # 重复回答
            if repeat:
                time.sleep(0.5)
                try:
                    got.play_audio_tts(answer, 0, True)
                except:
                    pass

            # 保存对话历史（只保留最近2轮=4条消息）
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": answer})
            if len(chat_history) > 4:
                chat_history = chat_history[-4:]

            time.sleep(0.5)

        # ===== 3. 显示表情 =====
        print(f"\n===== 3. 显示表情：{emotion} =====")
        emo_en = EMOTION_MAP.get(emotion, "Smile")
        print(f"  表情英文名: {emo_en}")
        try:
            got.play_audio_tts(f"接下来，给你看一个{emotion}的表情", 0, True)
        except:
            pass
        try:
            got.screen_display_emotion(emo_en, loop=False)
            print(f"  表情已显示，保持 {EMOTION_HOLD_TIME} 秒")
        except Exception as e:
            print(f"  表情显示失败: {e}")
        time.sleep(EMOTION_HOLD_TIME)

        # ===== 4. 模仿音效 =====
        print(f"\n===== 4. 模仿音效：{sound_list} =====")
        for idx, snd in enumerate(sound_list):
            snd_en = SOUND_MAP.get(snd, "monkey")
            print(f"  播放音效 {idx+1}/{len(sound_list)}：{snd} → {snd_en}")
            try:
                got.play_sound(snd_en, wait=True)
                print(f"  音效播放完成")
            except Exception as e:
                print(f"  音效播放失败: {e}，用TTS兜底")
                try:
                    got.play_audio_tts(f"模仿{snd}的声音", 0, True)
                except:
                    pass
            time.sleep(0.5)

        # ===== 5. 呼吸灯 =====
        print(f"\n===== 5. {light_color}呼吸灯 {light_duration}秒 =====")
        rgb = LIGHT_COLOR_MAP.get(light_color, [255, 0, 0])
        print(f"  RGB: {rgb}")
        try:
            got.play_audio_tts(f"最后，给你展示{light_color}呼吸灯", 0, True)
        except:
            pass
        try:
            got.show_light_rgb_effect(rgb[0], rgb[1], rgb[2], 2)
            print(f"  呼吸灯已开启，持续 {light_duration} 秒")
            time.sleep(light_duration)
            got.show_light_rgb_effect(0, 0, 0, 1)
            print("  呼吸灯已关闭")
        except Exception as e:
            print(f"  呼吸灯失败: {e}")

        # ===== 结束 =====
        try:
            got.play_audio_tts("陪伴任务完成，谢谢你和我聊天，再见！", 0, True)
        except:
            pass
        show_text_on_screen(got, "陪伴完成 再见")

        print("\n" + "=" * 60)
        print(" 智能陪伴流程完成！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n陪伴流程出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 确保呼吸灯关闭
        try:
            got.show_light_rgb_effect(0, 0, 0, 1)
        except:
            pass


# ============================================================
#  命令行入口
#  用法：python smart_companion.py <chat_times> <emotion> <sound> <light_color> <light_duration> <repeat>
#  示例：python smart_companion.py 2 愤怒 "['大象','老虎']" 红色 3 False
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("用法: python smart_companion.py <chat_times> <emotion> <sound> <light_color> <light_duration> <repeat>")
        print("示例: python smart_companion.py 2 愤怒 \"['大象','老虎']\" 红色 3 False")
        print("  chat_times: 对话次数（整数）")
        print("  emotion: 表情（愤怒/微笑/高兴/疑惑/爱心/骄傲/睡眠/苏醒）")
        print("  sound: 音效（单个如'大象'，或列表如\"['大象','老虎']\"）")
        print("  light_color: 呼吸灯颜色（红色/蓝色/绿色）")
        print("  light_duration: 呼吸灯时长（秒，整数）")
        print("  repeat: 是否重复回答（True/False）")
        sys.exit(1)

    chat_times = int(sys.argv[1])
    emotion = sys.argv[2]
    sound_str = sys.argv[3]
    light_color = sys.argv[4]
    light_duration = int(sys.argv[5])
    repeat = sys.argv[6].lower() in ('true', '1', 'yes')

    sound_list = parse_sound_param(sound_str)

    print("=== smart_companion.py 启动 ===")
    print(f"对话次数: {chat_times}")
    print(f"表情: {emotion}")
    print(f"音效列表: {sound_list}")
    print(f"呼吸灯: {light_color} {light_duration}秒")
    print(f"重复回答: {repeat}")
    print(f"默认城市: {DEFAULT_CITY}")

    # 初始化机器人
    got = ugot.UGOT()
    got.initialize(ROBOT_IP)
    print("机器人连接成功")

    # 执行陪伴流程
    success = companion_full_flow(
        got, chat_times, emotion, sound_list, light_color, light_duration, repeat
    )

    print(f"\n执行结果: success={success}")
    sys.exit(0 if success else 1)
