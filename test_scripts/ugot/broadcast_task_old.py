# -*- coding: utf-8 -*-
"""
ugot_gb.py - 语音识别获取任务并广播给yanshee
直接运行测试，不需要巡线/抓取
"""
import time
import socket
import json
import re
from ugot import ugot

BROADCAST_PORT = 9527
BROADCAST_ADDR = '10.233.104.217'

COLORS = ["红色", "蓝色", "绿色", "黄色", "白色", "黑色"]
AREA_LETTERS = ['A', 'B']


def extract_task_info(task_message):
    extracted_colors = []
    for color in COLORS:
        start = 0
        while True:
            pos = task_message.find(color, start)
            if pos == -1:
                break
            extracted_colors.append((color, pos))
            start = pos + len(color)
    extracted_colors.sort(key=lambda x: x[1])
    colors_ordered = [c for c, _ in extracted_colors]

    extracted_stores = []
    pattern = r'([A-Ba-b0-9])\s*号\s*(?:存储区|库区|仓库|存放区)'
    for m in re.finditer(pattern, task_message):
        char = m.group(1).upper()
        extracted_stores.append((char, m.start()))

    if not extracted_stores:
        has_store_keyword = any(k in task_message for k in
                                ["存储", "放置", "区", "库区", "仓库", "存放"])
        if has_store_keyword:
            for char in AREA_LETTERS:
                if char in task_message:
                    extracted_stores.append((char, task_message.find(char)))

    extracted_stores.sort(key=lambda x: x[1])
    stores_ordered = [s for s, _ in extracted_stores]

    task_list = []
    pair_count = min(len(colors_ordered), len(stores_ordered))
    for i in range(pair_count):
        task_list.append((colors_ordered[i], stores_ordered[i]))

    return task_list


def broadcast(task_list):
    if len(task_list) < 2:
        print("任务数量不足，跳过广播")
        return False

    task_data = {
        "first_task": {
            "color": task_list[0][0],
            "storage_area": task_list[0][1]
        },
        "second_task": {
            "color": task_list[1][0],
            "storage_area": task_list[1][1]
        }
    }
    json_str = json.dumps(task_data, ensure_ascii=False).encode('utf-8')

    try:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        for i in range(20):
            udp_socket.sendto(json_str, (BROADCAST_ADDR, BROADCAST_PORT))
            print(f"任务数据已广播 ({i+1}/20)：{task_data}")
            time.sleep(0.1)

        udp_socket.close()
        return True
    except Exception as e:
        print(f"广播发送失败：{e}")
        return False


def main():
    print("=== 语音识别广播测试 ===")
    u = ugot.UGOT()
    u.initialize('10.233.104.242')
    print("机器人连接成功")

    while True:
        print("\n语音识别已启动，请说出指令...")
        task_msg = u.start_audio_asr()
        print(f"识别内容：{task_msg}")

        task_list = extract_task_info(task_msg)
        print(f"解析任务：{task_list}")

        if len(task_list) >= 2:
            (c1, s1), (c2, s2) = task_list[0], task_list[1]
            res = f"好的，第一次搬运{c1}放{s1}号，第二次搬运{c2}放{s2}号"
            print(res)
            u.play_audio_tts(res, 0, False)
            time.sleep(0.5)
            broadcast(task_list)
            print("广播完成，程序结束")
            break
        else:
            print("识别失败，请重试（需包含两次颜色+存储区）")
            u.play_audio_tts("请重试", 0, True)


if __name__ == "__main__":
    main()
