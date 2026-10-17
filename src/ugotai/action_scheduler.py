# -*- coding: utf-8 -*-
"""
action_scheduler.py - 机器人动作调度器
四大场景任务通过 subprocess 调用对应的外部脚本执行实际动作：
  - 智能仓储 → smart_warehouse.py
  - 智能陪伴 → smart_companion.py
  - 智能侦察 → smart_recon.py
  - 智能解算 → smart_geometry.py
基础工具函数（移动、机械臂、屏幕音效、视觉识别）保留供外部脚本 import 使用。
"""

from ugot import ugot
import time
import ast
import math
import cv2
import numpy as np
from color import grab_color_block_full, detect_color_objects, color_ranges


# ========== 全局初始化 ==========
u = ugot.UGOT()
u.initialize("10.233.104.242")
u.open_camera()
u.load_models([
    "color_recognition",
    "apriltag_qrcode",
    "traffic_sign",
    "gesture",
    "word_recognition",
    "line_recognition"
])
time.sleep(2)


# ========== 公共配置 ==========
# tf-od 环境 python 路径（所有视觉脚本都在这个环境下运行）
TF_PYTHON = r"C:\Users\16231\anaconda3\envs\tf-od\python.exe"
# 脚本所在目录
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ========== 工具函数 ==========
def safe_int(val, default=0):
    try:
        return int(float(val))
    except:
        return default

def safe_str(val, default=""):
    try:
        return str(val)
    except:
        return default


# ========== 基础移动函数 ==========
def move_forward(distance=30):
    """前进指定厘米"""
    d = safe_int(distance, 30)
    print(f"前进 {d}cm")
    try:
        u.mecanum_move_speed_times(0, 20, d, 1)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f" 前进失败: {e}")
        return False

def move_backward(distance=30):
    """后退指定厘米"""
    d = safe_int(distance, 30)
    print(f"后退 {d}cm")
    try:
        u.mecanum_move_speed_times(1, 20, d, 1)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f" 后退失败: {e}")
        return False

def turn_left(angle=90):
    """左转指定角度"""
    a = safe_int(angle, 90)
    print(f"左转 {a}°")
    try:
        u.mecanum_turn_speed_times(2, 130, a, 2)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f" 左转失败: {e}")
        return False

def turn_right(angle=90):
    """右转指定角度"""
    a = safe_int(angle, 90)
    print(f"右转 {a}°")
    try:
        u.mecanum_turn_speed_times(3, 130, a, 2)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f" 右转失败: {e}")
        return False

def move_left(distance=30):
    """左平移指定厘米"""
    d = safe_int(distance, 30)
    print(f"左平移 {d}cm")
    try:
        u.mecanum_translate_speed_times(-90, 15, d, 1)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f" 左平移失败: {e}")
        return False

def move_right(distance=30):
    """右平移指定厘米"""
    d = safe_int(distance, 30)
    print(f"右平移 {d}cm")
    try:
        u.mecanum_translate_speed_times(90, 15, d, 1)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f" 右平移失败: {e}")
        return False


# ========== 机械臂控制 ==========
def release_block_action():
    """释放色块动作"""
    print("释放色块")
    try:
        u.mechanical_single_joint_control(2, -100, 50)
        u.mechanical_single_joint_control(3, 16, 50)
        time.sleep(0.6)
        u.mechanical_clamp_release()
        time.sleep(0.2)
        u.mechanical_single_joint_control(2, -25, 100)
        u.mechanical_single_joint_control(3, -70, 100)
        time.sleep(0.5)
        u.mechanical_single_joint_control(2, 80, 100)
        u.mechanical_single_joint_control(3, -80, 100)
        return True
    except Exception as e:
        print(f" 释放失败: {e}")
        return False

def move_arm_action(duration=3):
    """机械臂自定义动作，持续至少3秒"""
    d = safe_int(duration, 3)
    print(f"机械臂动作 {d}秒")
    try:
        start = time.time()
        while time.time() - start < d:
            u.mechanical_single_joint_control(2, 0, 100)
            time.sleep(0.5)
            u.mechanical_single_joint_control(2, 60, 100)
            time.sleep(0.5)
            u.mechanical_clamp_release()
            time.sleep(0.3)
            u.mechanical_clamp_close()
            time.sleep(0.3)
        u.mechanical_single_joint_control(2, 80, 100)
        u.mechanical_single_joint_control(3, -80, 100)
        return True
    except Exception as e:
        print(f" 机械臂动作失败: {e}")
        return False

def arm_reset():
    """机械臂复位到初始位置"""
    u.mechanical_clamp_release()
    u.mechanical_single_joint_control(3, -85, 100)
    u.mechanical_single_joint_control(2, 80, 100)
    time.sleep(1)


# ========== 屏幕与音效 ==========
def show_emotion(emotion="高兴"):
    """显示表情"""
    emo_map = {
        "愤怒": "Anger", "生气": "Anger",
        "微笑": "Smile", "高兴": "Smile", "开心": "Smile",
        "疑惑": "Doubt", "搜索": "Search", "呼吸": "Breathe",
        "眨眼": "Blink", "抵抗": "Resist", "爱心": "Love",
        "骄傲": "Proud", "挠痒痒": "Ticklish", "虚弱": "Weakness",
        "睡眠": "Sleep", "苏醒": "WakeUp"
    }
    emo_name = emo_map.get(emotion, "Smile")
    print(f"显示表情: {emotion}")
    try:
        u.screen_display_emotion(emo_name, loop=False)
        time.sleep(2)
        return True
    except Exception as e:
        print(f" 表情显示失败: {e}")
        return False

def imitate_sound(sound=""):
    """模仿音效"""
    sound_map = {
        "熊": "bear", "鸟": "bird", "鸡": "chicken", "牛": "cow",
        "狗": "dog", "大象": "elephant", "长颈鹿": "giraffe", "马": "horse",
        "狮子": "lion", "猴子": "monkey", "猪": "pig", "老虎": "tiger",
        "救护车": "ambulance", "警车": "police_car_1", "汽车喇叭": "carhorn",
        "门铃": "doorbell", "机器人": "robot", "引擎": "engine", "激光": "laser",
        "高兴": "happy", "生气": "angry", "惊讶": "surprise", "疑问": "doubt",
        "完成": "complete", "收到": "received", "遵命": "yes"
    }
    snd_name = sound_map.get(sound, "monkey")
    print(f" 播放音效: {sound}")
    try:
        u.play_sound(snd_name, wait=True)
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f" 音效播放失败: {e}")
        try:
            u.play_audio_tts(f"模仿{sound}的声音", 0, True)
            return True
        except:
            return False

def set_breathing_light(color="红色", duration=3):
    """设置呼吸灯"""
    color_map = {
        "红色": [255, 0, 0], "蓝色": [0, 0, 255], "绿色": [0, 255, 0]
    }
    rgb = color_map.get(color, [255, 0, 0])
    d = safe_int(duration, 3)
    print(f" {color}呼吸灯 {d}秒")
    try:
        u.show_light_rgb_effect(rgb[0], rgb[1], rgb[2], 2)
        time.sleep(d)
        u.show_light_rgb_effect(0, 0, 0, 1)
        return True
    except Exception as e:
        print(f"呼吸灯失败: {e}")
        return False

def display_text_on_screen(text, color=0):
    """屏幕显示文字"""
    txt = safe_str(text)
    print(f"屏幕显示: {txt}")
    try:
        u.screen_clear()
        u.screen_display_background(1)
        u.screen_print_text(txt, color)
        time.sleep(2)
        return True
    except Exception as e:
        print(f"文字显示失败: {e}")
        return False

def speak_and_display(text):
    """语音播报并屏幕显示文字"""
    txt = safe_str(text)
    print(f"播报并显示: {txt}")
    try:
        u.screen_clear()
        u.screen_display_background(1)
        u.screen_print_text(txt, 0)
        time.sleep(0.3)
        u.play_audio_tts(txt, 0, True)
        time.sleep(1)
        return True
    except Exception as e:
        print(f"播报失败: {e}")
        return False

def clear_screen():
    """清屏"""
    try:
        u.screen_clear()
        return True
    except:
        return False


# ========== 视觉识别函数 ==========
def _recognize_single(recognize_func, name, max_attempts=30):
    """通用单类别识别+旋转寻找框架"""
    for i in range(max_attempts):
        try:
            result = recognize_func()
            if result:
                speak_and_display(f"识别到{name}：{result}")
                return result
        except:
            pass
        turn_right(15)
        time.sleep(0.3)
    speak_and_display(f"未识别到{name}")
    return None

def recognize_traffic_sign():
    """识别交通标志"""
    def _detect():
        info = u.get_traffic_total_info()
        return info[0][0] if info else None
    return _recognize_single(_detect, "交通标志")

def recognize_gesture():
    """识别手势"""
    gesture_map = {
        "stone": "石头", "scissors": "剪刀", "cloth": "布",
        "ok": "OK手势", "like": "点赞"
    }
    def _detect():
        res = u.get_gesture_result()
        return gesture_map.get(res, res) if res else None
    return _recognize_single(_detect, "手势")

def recognize_words():
    """识别文字"""
    def _detect():
        res = u.get_words_result()
        return res.strip() if res and res.strip() else None
    return _recognize_single(_detect, "文字")

def recognize_apriltag():
    """识别AprilTag编号"""
    def _detect():
        info = u.get_apriltag_total_info()
        return str(info[0][0]) + "号" if info else None
    return _recognize_single(_detect, "标签")


# ========== AprilTag 对准（带捷径分支） ==========
# 配置参数（标准值，不额外修改）
APRIL_FRAME_WIDTH = 640
APRIL_FRAME_CENTER = APRIL_FRAME_WIDTH // 2
APRIL_COARSE_THRESHOLD = 50
APRIL_MID_THRESHOLD = 25
APRIL_FINE_THRESHOLD = 10
APRIL_STABLE_FRAMES = 3
APRIL_SEARCH_SPEED = 6
APRIL_COARSE_SPEED = 12
APRIL_MID_SPEED = 8
APRIL_FINE_MIN_SPEED = 5
APRIL_FINE_KP = 0.35
APRIL_TAG_SIZE = '10'

APRIL_STATE_SEARCH = 0
APRIL_STATE_COARSE_ALIGN = 1
APRIL_STATE_MID_ALIGN = 2
APRIL_STATE_FINE_ALIGN = 3
APRIL_STATE_STABLE_CHECK = 4


def _april_parse_tag(raw_data):
    """解析 AprilTag 原始返回数据"""
    if not isinstance(raw_data, list) or len(raw_data) < 13:
        return None
    tag_id = raw_data[1]
    if tag_id == -1:
        return None
    dist_idx = {'5': 7, '7': 8, '10': 9}[APRIL_TAG_SIZE]
    return {
        "id": int(tag_id),
        "center_x": raw_data[2],
        "center_y": raw_data[3],
        "area": raw_data[6],
        "distance": raw_data[dist_idx] * 100,
        "yaw_deg": math.degrees(raw_data[12])
    }


def align_apriltag(target_id=None):
    """
    AprilTag 五阶段对准，搜索阶段带捷径分支：
      - 偏差 ≤ 10px → 停转，直接进稳定确认（连续3帧≤10px即成功），跳过粗/中/精对准
      - 偏差 > 10px → 正常走粗对准 → 中速对准 → 精对准 → 稳定确认

    参数:
        target_id: 目标标签ID，None表示对准任意识别到的标签
    返回:
        (success, tag_info)
    """
    import math
    current_state = APRIL_STATE_SEARCH
    stable_count = 0
    last_direction = 0

    try:
        while True:
            raw_data = u.get_qrcode_apriltag_total_info()
            tag = _april_parse_tag(raw_data)

            # 阶段1：搜索标签（带捷径分支）
            if current_state == APRIL_STATE_SEARCH:
                if tag is None:
                    print("[AprilTag搜索] 未识别到标签，自转搜索中...")
                    u.mecanum_move_xyz(0, 0, APRIL_SEARCH_SPEED)
                    time.sleep(0.1)
                    continue

                if target_id is not None and tag["id"] != target_id:
                    print(f"[AprilTag搜索] 识别到ID={tag['id']}，非目标ID={target_id}，继续搜索")
                    u.mecanum_move_xyz(0, 0, APRIL_SEARCH_SPEED)
                    time.sleep(0.1)
                    continue

                # 【新增捷径分支】偏差已在精对准范围内，直接跳过所有调整阶段
                offset = tag["center_x"] - APRIL_FRAME_CENTER
                if abs(offset) <= APRIL_FINE_THRESHOLD:
                    print(f"[AprilTag搜索] 识别到目标标签 ID={tag['id']}，偏差 {offset:+.0f}px ≤ {APRIL_FINE_THRESHOLD}px，停转后直接进入稳定确认（跳过粗/中/精对准）")
                    u.mecanum_move_xyz(0, 0, 0)
                    current_state = APRIL_STATE_STABLE_CHECK
                    stable_count = 0
                else:
                    print(f"[AprilTag搜索] 识别到目标标签 ID={tag['id']}，偏差 {offset:+.0f}px，进入粗对准")
                    current_state = APRIL_STATE_COARSE_ALIGN
                    stable_count = 0

            # 阶段2：粗对准
            elif current_state == APRIL_STATE_COARSE_ALIGN:
                if tag is None:
                    print("[AprilTag粗对准] 标签丢失，返回搜索阶段")
                    current_state = APRIL_STATE_SEARCH
                    continue
                offset = tag["center_x"] - APRIL_FRAME_CENTER
                print(f"[AprilTag粗对准] ID:{tag['id']} | 偏移:{offset:+.0f}px | 阈值:±{APRIL_COARSE_THRESHOLD}px")
                if abs(offset) <= APRIL_COARSE_THRESHOLD:
                    print("[AprilTag粗对准] 进入中速对准阶段")
                    current_state = APRIL_STATE_MID_ALIGN
                    continue
                direction = 1 if offset > 0 else -1
                u.mecanum_move_xyz(0, 0, -direction * APRIL_COARSE_SPEED)
                last_direction = direction
                time.sleep(0.05)

            # 阶段3：中速对准
            elif current_state == APRIL_STATE_MID_ALIGN:
                if tag is None:
                    print("[AprilTag中速对准] 标签丢失，返回粗对准")
                    current_state = APRIL_STATE_COARSE_ALIGN
                    continue
                offset = tag["center_x"] - APRIL_FRAME_CENTER
                print(f"[AprilTag中速对准] ID:{tag['id']} | 偏移:{offset:+.0f}px | 阈值:±{APRIL_MID_THRESHOLD}px")
                if abs(offset) <= APRIL_MID_THRESHOLD:
                    print("[AprilTag中速对准] 进入精对准阶段")
                    current_state = APRIL_STATE_FINE_ALIGN
                    continue
                direction = 1 if offset > 0 else -1
                u.mecanum_move_xyz(0, 0, -direction * APRIL_MID_SPEED)
                last_direction = direction
                time.sleep(0.05)

            # 阶段4：精对准
            elif current_state == APRIL_STATE_FINE_ALIGN:
                if tag is None:
                    print("[AprilTag精对准] 标签丢失，返回中速对准")
                    current_state = APRIL_STATE_MID_ALIGN
                    continue
                offset = tag["center_x"] - APRIL_FRAME_CENTER
                print(f"[AprilTag精对准] ID:{tag['id']} | 偏移:{offset:+.0f}px | 阈值:±{APRIL_FINE_THRESHOLD}px")
                if abs(offset) <= APRIL_FINE_THRESHOLD:
                    print("[AprilTag精对准] 偏移达标，进入稳定确认阶段")
                    current_state = APRIL_STATE_STABLE_CHECK
                    stable_count = 0
                    u.mecanum_move_xyz(0, 0, 0)
                    continue
                direction = 1 if offset > 0 else -1
                speed = APRIL_FINE_KP * abs(offset)
                speed = max(APRIL_FINE_MIN_SPEED, min(speed, APRIL_MID_SPEED))
                speed = int(speed)
                u.mecanum_move_xyz(0, 0, -direction * speed)
                last_direction = direction
                time.sleep(0.05)

            # 阶段5：稳定确认
            elif current_state == APRIL_STATE_STABLE_CHECK:
                if tag is None:
                    print("[AprilTag稳定确认] 标签丢失，返回精对准")
                    current_state = APRIL_STATE_FINE_ALIGN
                    continue
                offset = tag["center_x"] - APRIL_FRAME_CENTER
                if abs(offset) <= APRIL_FINE_THRESHOLD:
                    stable_count += 1
                    print(f"[AprilTag稳定确认] 居中保持中... {stable_count}/{APRIL_STABLE_FRAMES}")
                    u.mecanum_move_xyz(0, 0, 0)
                    if stable_count >= APRIL_STABLE_FRAMES:
                        print(f"\n AprilTag对准完成！ID={tag['id']} 距离={tag['distance']:.1f}cm 偏航={tag['yaw_deg']:.1f}°\n")
                        u.mecanum_stop()
                        return True, tag
                else:
                    print("[AprilTag稳定确认] 偏移超出阈值，退回精对准")
                    stable_count = 0
                    current_state = APRIL_STATE_FINE_ALIGN
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n手动中断AprilTag对准")
        u.mecanum_stop()
        return False, None
    except Exception as e:
        print(f"AprilTag对准异常: {e}")
        u.mecanum_stop()
        return False, None

def recognize_color():
    """识别色块颜色"""
    def _detect():
        info = u.get_color_total_info()
        if not info:
            return None
        c = info[0]
        color_zh = {"red": "红色", "blue": "蓝色", "green": "绿色"}
        return color_zh.get(c, c)
    return _recognize_single(_detect, "色块")


# ========== 陪伴对话模式（大模型可能拆成 chat_mode + 表情 + 音效 + 灯） ==========

def chat_mode(times=2, repeat=False):
    """
    陪伴对话模式：和用户进行 times 次对话
    大模型可能把陪伴任务拆成 chat_mode + show_emotion + imitate_sound + set_breathing_light
    参数：
      times: 对话次数（整数）
      repeat: 是否重复回答（True/False）
    """
    import random

    # 延迟导入大模型（ugot环境有qianfan）
    try:
        import ssl
        try:
            import certifi
            def _patched_ctx(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ctx.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
                ctx.check_hostname = True
                ctx.verify_mode = ssl.CERT_REQUIRED
                ctx.load_verify_locations(cafile=certifi.where())
                return ctx
            ssl.create_default_context = _patched_ctx
        except ImportError:
            pass

        import qianfan
        from api_config import QIANFAN_ACCESS_KEY, QIANFAN_SECRET_KEY
        import os
        os.environ["QIANFAN_ACCESS_KEY"] = QIANFAN_ACCESS_KEY
        os.environ["QIANFAN_SECRET_KEY"] = QIANFAN_SECRET_KEY
        _chat_comp = qianfan.ChatCompletion(model="DeepSeek-V4-Flash")
        LLM_READY = True
    except Exception as e:
        print(f"  大模型初始化失败: {e}")
        LLM_READY = False

    SYSTEM_PROMPT = """你是一个陪伴机器人，回答必须遵守：
1. 简短，1-2句话，不超过40个字
2. 口语化，适合语音播报
3. 不要解释、不要列举、不要说"作为AI"
4. 用户讲笑话就回一个短笑话
5. 用户自我介绍就做一句简短自我介绍
"""

    FALLBACK_ANSWERS = [
        "这个问题有点难，不过开心最重要啦！",
        "哈哈你说的真有趣，我们换个话题吧？",
        "我正在思考人生呢，你觉得今天天气怎么样？",
        "这个我得想想，要不我给你讲个笑话吧！",
        "嗯嗯我听到了，你继续说，我在认真听呢。",
    ]

    chat_history = []
    t = safe_int(times, 2)
    rep = repeat if isinstance(repeat, bool) else str(repeat).lower() in ('true', '1', 'yes')

    print(f"\n  陪伴对话模式：{t}次对话，重复回答={rep}")
    # 开场不说话（main.py已播报大模型response），直接进入对话

    for i in range(t):
        print(f"\n  --- 第 {i+1}/{t} 次对话 ---")

        # 提示用户说话
        try:
            u.play_audio_tts(f"第{i+1}次对话，你说", 0, True)
        except:
            pass

        # 循环识别用户说话（15秒超时）
        user_text = None
        start_time = time.time()
        while time.time() - start_time < 15:
            try:
                result = u.start_audio_asr()
                if result and result.strip():
                    user_text = result.strip()
                    print(f"  用户说：{user_text}")
                    break
            except:
                pass
            time.sleep(0.3)

        if not user_text:
            print("  未识别到用户说话，跳过本轮")
            try:
                u.play_audio_tts("没有听到你说话，我们继续吧", 0, True)
            except:
                pass
            continue

        # 屏幕显示用户说的话
        display_text_on_screen(user_text)

        # 大模型生成回答
        if LLM_READY:
            try:
                # 千帆模型不支持system角色作为第一条消息，把系统提示拼到user消息里
                user_prompt = f"{SYSTEM_PROMPT}\n\n用户说：{user_text}"
                messages = list(chat_history)
                messages.append({"role": "user", "content": user_prompt})
                resp = _chat_comp.do(
                    messages=messages,
                    top_p=0.8,
                    temperature=0.7,
                    penalty_score=1.0,
                    request_timeout=8,
                )
                answer = resp["result"].strip().strip('"').strip("'")
                if len(answer) > 50:
                    answer = answer[:50] + "。"
            except Exception as e:
                print(f"  大模型调用失败: {e}，使用兜底回答")
                answer = random.choice(FALLBACK_ANSWERS)
        else:
            answer = random.choice(FALLBACK_ANSWERS)

        print(f"  机器人回答：{answer}")

        # 语音播报回答
        try:
            u.play_audio_tts(answer, 0, True)
        except Exception as e:
            print(f"  语音播报失败: {e}")

        # 屏幕显示回答
        display_text_on_screen(answer)

        # 重复回答
        if rep:
            time.sleep(0.5)
            try:
                u.play_audio_tts(answer, 0, True)
            except:
                pass

        # 保存对话历史（只保留最近2轮）
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > 4:
            chat_history = chat_history[-4:]

        time.sleep(0.5)

    try:
        u.play_audio_tts("聊天结束，谢谢你和我说话", 0, True)
    except:
        pass
    return True


def chat_once(question="你还好吗？需要帮助吗？"):
    """
    场景化单次对话：机器人主动问一个问题，用户回答，机器人回应。
    适用于"有人摔倒了询问情况"等场景，只问一次，符合场景。
    参数：
      question: 机器人主动问的问题（由大模型根据场景生成）
    返回：机器人的回应内容
    """
    import random

    # 延迟导入大模型（和 chat_mode 一样的 SSL 补丁）
    try:
        import ssl
        try:
            import certifi
            def _patched_ctx2(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ctx.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
                ctx.check_hostname = True
                ctx.verify_mode = ssl.CERT_REQUIRED
                ctx.load_verify_locations(cafile=certifi.where())
                return ctx
            ssl.create_default_context = _patched_ctx2
        except ImportError:
            pass

        import qianfan
        from api_config import QIANFAN_ACCESS_KEY, QIANFAN_SECRET_KEY
        import os
        os.environ["QIANFAN_ACCESS_KEY"] = QIANFAN_ACCESS_KEY
        os.environ["QIANFAN_SECRET_KEY"] = QIANFAN_SECRET_KEY
        _chat_comp2 = qianfan.ChatCompletion(model="DeepSeek-V4-Flash")
        LLM_READY2 = True
    except Exception as e:
        print(f"  大模型初始化失败: {e}")
        LLM_READY2 = False

    SYSTEM_PROMPT2 = """你是一个陪伴机器人，正在询问对方情况。
对方回答后，你要用一句话回应，表达关心或提供帮助。
规则：
1. 回答必须简短（1句话，不超过30字）
2. 口语化、温暖、有人情味
3. 对方说没事/还好 → 回应"那就好，注意休息，有需要随时叫我"
4. 对方说疼/不舒服 → 回应"好的，我帮你叫人，你先别动"
5. 对方说需要帮助 → 回应"好的，我马上帮你，你坚持一下"
6. 不要说"作为AI"，不要解释，直接回应
"""

    FALLBACK2 = [
        "好的，我明白了，你先休息一下，有需要随时叫我。",
        "嗯嗯，我在这里，我帮你看看周围情况。",
        "好的，你注意安全，我马上帮你叫人来。",
    ]

    q = safe_str(question, "你还好吗？需要帮助吗？")

    print(f"\n  场景化单次对话：问「{q}」")

    # 1. 播报问题 + 屏幕显示问题
    try:
        u.play_audio_tts(q, 0, True)
    except:
        pass
    display_text_on_screen(q)

    # 2. 等待用户回答（15秒超时）
    user_answer = None
    start_time = time.time()
    while time.time() - start_time < 15:
        try:
            result = u.start_audio_asr()
            if result and result.strip():
                user_answer = result.strip()
                print(f"  用户回答：{user_answer}")
                break
        except:
            pass
        time.sleep(0.3)

    if not user_answer:
        print("  未听到用户回答，使用默认回应")
        user_answer = "（对方没有回答）"

    # 3. 大模型根据用户回答生成回应
    if LLM_READY2:
        try:
            user_prompt = f"{SYSTEM_PROMPT2}\n\n我问：{q}\n对方回答：{user_answer}\n请用一句话回应。"
            resp = _chat_comp2.do(
                messages=[{"role": "user", "content": user_prompt}],
                top_p=0.8,
                temperature=0.7,
                penalty_score=1.0,
                request_timeout=8,
            )
            answer = resp["result"].strip().strip('"').strip("'")
            if len(answer) > 40:
                answer = answer[:40] + "。"
        except Exception as e:
            print(f"  大模型调用失败: {e}，使用兜底回答")
            answer = random.choice(FALLBACK2)
    else:
        answer = random.choice(FALLBACK2)

    print(f"  机器人回应：{answer}")

    # 4. 播报回应 + 屏幕显示回应（得分点：显示屏显示回答的文字）
    try:
        u.play_audio_tts(answer, 0, True)
    except:
        pass
    display_text_on_screen(answer)

    return answer


# ========== 四大场景任务（调度器：调用外部脚本） ==========

def _run_script(script_name, args):
    """
    通用脚本调用函数
    :param script_name: 脚本文件名（如 smart_warehouse.py）
    :param args: 命令行参数列表
    :return: True=成功, False=失败
    """
    import subprocess
    import os
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[警告] 脚本不存在: {script_path}")
        print(f"[提示] 请先创建 {script_name} 脚本")
        return False
    cmd = [TF_PYTHON, script_path] + [str(a) for a in args]
    print(f"执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"{script_name} 执行成功")
            return True
        else:
            print(f"{script_name} 执行失败，返回码: {result.returncode}")
            return False
    except Exception as e:
        print(f"{script_name} 执行出错: {e}")
        return False


# ---- 1. 智能仓储 → smart_warehouse.py ----
def smart_warehouse_task(pre_rotate, forward_dist, target_color, target_pos, backward_dist, rotate_angle=0, need_return=1):
    """
    智能仓储完整任务（调用 smart_warehouse.py）
    参数：
    - pre_rotate: 前置转圈角度（正数右转，负数左转，0不转），如360=先原地转一圈
    - forward_dist: 前进距离（cm）
    - target_color: 目标颜色（红色/绿色/蓝色）
    - target_pos: 目标存储区编号（1-6）
    - backward_dist: 后退距离（cm）
    - rotate_angle: 完成后转向角度（度），正=左转，负=右转
    - need_return: 1=7步(回到取货区), 0=6步(不返回)
    """
    print("\n" + "=" * 50)
    print(" 智能仓储任务启动（调用 smart_warehouse.py）")
    print(f" 前置转圈{pre_rotate}° | 前进{forward_dist}cm | 抓取{target_color} | 放置{target_pos}号 | 后退{backward_dist}cm | 转向{rotate_angle}度 | {'7步(返回)' if need_return else '6步(不返回)'}")
    print("=" * 50)

    # smart_warehouse.py smart_warehouse 模式参数：smart_warehouse 前置转圈 前进cm 颜色 放置号 后退cm 转向角 是否返回
    return _run_script("smart_warehouse.py", ["smart_warehouse", str(pre_rotate), str(forward_dist), target_color, str(target_pos), str(backward_dist), str(rotate_angle), str(need_return)])


# ---- 2. 智能陪伴 → smart_companion.py ----
def smart_companion_task(chat_times, emotion, sound, light_color, light_duration=3, repeat=False):
    """
    智能陪伴完整任务（调用 smart_companion.py，待创建）
    参数：
    - chat_times: 对话次数
    - emotion: 表情
    - sound: 音效
    - light_color: 呼吸灯颜色
    - light_duration: 呼吸灯时长（秒）
    - repeat: 是否重复回答
    """
    print("\n" + "=" * 50)
    print(" 智能陪伴任务启动（调用 smart_companion.py）")
    print(f" 对话{chat_times}次 | 表情{emotion} | 音效{sound} | {light_color}呼吸灯{light_duration}秒 | 重复:{repeat}")
    print("=" * 50)

    # smart_companion.py 参数：chat_times emotion sound light_color light_duration repeat
    return _run_script("smart_companion.py", [chat_times, emotion, sound, light_color, light_duration, repeat])


# ---- 3. 智能侦察 → smart_recon.py ----
def smart_detective_task(sequence=None):
    """
    智能侦察任务（调用 smart_recon.py，待创建）
    参数：
    - sequence: 识别顺序列表，如 ['文字', '手势', '交通标志', '标签', '色块']
    """
    if sequence is None:
        sequence = ["交通标志", "手势", "文字", "标签", "色块"]

    print("\n" + "=" * 50)
    print(" 智能侦察任务启动（调用 smart_recon.py）")
    print(f" 识别顺序: {' → '.join(sequence)}")
    print("=" * 50)

    # smart_recon.py 参数：识别顺序（用逗号分隔的字符串）
    seq_str = ",".join(sequence)
    return _run_script("smart_recon.py", [seq_str])


# ---- 4. 智能解算 → smart_geometry.py ----
def geometry_task(action_seq):
    """
    智能解算任务（调用 smart_geometry.py）
    参数：动作序列字符串，动作之间用分号分隔
    每个动作格式为 "类型:参数1,参数2,..."
    支持动作类型：
      polygon:<边数>,<边长>,<cw|ccw>[,<forward|backward>]  走多边形（边3-8，边长≤50）
      move:<距离>,<left|right|forward|backward>              平移
      turn:<角度>,<left|right>                                原地转向（≤360°）
      rotate:<角度>                                            旋转（用于转圈）
      arm:<时长秒>                                              机械臂动作（≥3秒）
    示例：
      geometry_task("polygon:4,50,cw;turn:90,left;polygon:4,50,cw,backward;move:30,left;arm:3")
      geometry_task("polygon:4,30,cw;move:30,right;polygon:5,30,ccw;arm:3;rotate:360")
    """
    actions = [a.strip() for a in action_seq.split(";") if a.strip()]

    print("\n" + "=" * 50)
    print(" 智能解算任务启动（调用 smart_geometry.py）")
    print(f" 动作序列（共{len(actions)}个）:")
    for i, a in enumerate(actions):
        print(f"   {i + 1}. {a}")
    print("=" * 50)

    return _run_script("smart_geometry.py", [action_seq])


# ========== 动作执行引擎 ==========
def execute_action(action_str):
    """执行单条动作字符串，安全解析调用"""
    try:
        action_str = action_str.strip()
        if '(' not in action_str:
            return False

        func_name = action_str.split('(')[0].strip()
        params_str = action_str.split('(', 1)[1].rsplit(')', 1)[0].strip()

        # 解析参数：按逗号分割，但保护方括号（列表）和引号（字符串）内的逗号
        params = []
        if params_str:
            parts = []
            depth = 0
            in_quote = False
            quote_char = ''
            cur = ''
            for ch in params_str:
                # 引号状态跟踪（双引号或单引号）
                if ch in ('"', "'") and not in_quote:
                    in_quote = True
                    quote_char = ch
                elif ch == quote_char and in_quote:
                    in_quote = False
                    quote_char = ''
                # 方括号深度跟踪（只在引号外生效）
                elif ch == '[' and not in_quote:
                    depth += 1
                elif ch == ']' and not in_quote:
                    depth -= 1
                # 逗号分割（只在方括号外且引号外生效）
                if ch == ',' and depth == 0 and not in_quote:
                    parts.append(cur)
                    cur = ''
                else:
                    cur += ch
            if cur.strip():
                parts.append(cur)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if p.startswith('[') and p.endswith(']'):
                    try:
                        params.append(ast.literal_eval(p))
                    except Exception:
                        params.append(_parse_param(p))
                else:
                    params.append(_parse_param(p))

        # 调用函数
        if func_name in globals():
            func = globals()[func_name]
            func(*params)
            return True
        else:
            print(f" 未知函数: {func_name}")
            return False
    except Exception as e:
        print(f" 执行动作失败 [{action_str}]: {e}")
        return False

def _parse_param(p):
    """解析单个参数"""
    p = p.strip()
    if p in ("True", "False"):
        return p == "True"
    if p == "None":
        return None
    if (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
        return p[1:-1]
    try:
        return int(p)
    except:
        try:
            return float(p)
        except:
            return p

def execute_action_list(action_list):
    """批量执行动作列表"""
    total = len(action_list)
    success = 0
    for i, action in enumerate(action_list):
        print(f"\n  执行第 {i + 1}/{total} 个动作: {action}")
        if execute_action(action):
            success += 1
        time.sleep(0.5)
    print(f"\n 动作序列执行完成：{success}/{total} 成功")
    return success


# ========== 初始化 ==========
def initialize_robot():
    """机器人初始化"""
    print(" 机器人初始化中...")
    arm_reset()
    clear_screen()
    display_text_on_screen("机器人就绪")
    print(" 机器人初始化完成")
