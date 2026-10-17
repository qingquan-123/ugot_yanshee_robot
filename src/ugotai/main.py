# -*- coding: utf-8 -*-
import time
import os
import ssl
import certifi


def patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(cafile=certifi.where())
    return ctx


# 必须在导入 qianfan（依赖 aiohttp）之前打补丁：
# aiohttp 在 import 阶段就会调用 ssl.create_default_context()，
# 若此时补丁未生效，它会去读 Windows 证书存储，触发
# ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]
ssl.create_default_context = patched_create_default_context

from ugot import ugot
from llm_planner import agent_plan_qianfan
from action_scheduler import execute_action_list, clear_screen, initialize_robot


# 全局只初始化一次
u = ugot.UGOT()
u.initialize("10.233.104.242")


def agent_play():
    print("\n  等待语音指令...")
    u.play_audio_tts("你好，我是UGOT，需要我做什么呢", 0, True)

    # 语音识别
    while True:
        speech_result = u.start_audio_asr()
        if speech_result:
            print(f" 识别到指令：{speech_result}")
            os.makedirs('temp', exist_ok=True)
            with open('temp/speech_recognition.txt', 'w', encoding='utf-8') as f:
                f.write(speech_result)
            break
        else:
            u.play_audio_tts("我没有听清楚，请再说一遍", 0, True)

    # 大模型解析
    agent_plan_list, ai_response = agent_plan_qianfan(speech_result)

    # 保存动作计划
    with open('temp/agent_plan.txt', 'w', encoding='utf-8') as f:
        f.write(str(agent_plan_list))

    # 不复述任务，直接执行动作
    print("\n 开始执行动作...")
    execute_action_list(agent_plan_list)
    clear_screen()
    print("\n 任务执行完毕")


if __name__ == '__main__':
    initialize_robot()
    try:
        agent_play()
    finally:
        # 运行完一次任务后清理资源并退出
        print("\n 任务完成，正在清理资源...")
        try:
            u.mecanum_stop()
        except Exception:
            pass
        try:
            clear_screen()
        except Exception:
            pass
        try:
            u.close_camera()
        except Exception:
            pass
        print(" 已退出")
