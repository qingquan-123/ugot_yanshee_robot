# -*- coding: utf-8 -*-
"""
test_model.py - 测试标签识别模型
循环识别摄像头画面中的标签，识别到就播报是几号
"""
import sys
import os
import subprocess

TF_PYTHON = r"C:\Users\16231\anaconda3\envs\tf-od\python.exe"
if os.path.abspath(sys.executable) != os.path.abspath(TF_PYTHON):
    ret = subprocess.call([TF_PYTHON] + sys.argv)
    sys.exit(ret)

import time
import re
import cv2
import numpy as np
from ugot import ugot

try:
    import tensorflow as tf
    tflite = tf.lite
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[错误] tensorflow 未安装，请用 tf-od 环境运行")
    sys.exit(1)

MODEL_PATH = r'C:\Users\16231\Desktop\ros\pb_tflite\ssd_mobilenet_v2_fpn_320.tflite'
LABEL_MAP_PATH = r'C:\Users\16231\Desktop\ros\pb_tflite\label_map.pbtxt'

interpreter = None
input_details = None
output_details = None
is_uint8 = False
label_map = {}


def load_label_map(path):
    global label_map
    label_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    items = re.findall(r'item\s*\{([^}]+)\}', content)
    for item in items:
        id_match = re.search(r'id:\s*(\d+)', item)
        name_match = re.search(r"name:\s*'([^']+)'", item)
        if id_match and name_match:
            label_map[int(id_match.group(1))] = name_match.group(1)
    print(f"标签映射: {dict(sorted(label_map.items()))}")


def detect(frame, min_conf=0.4):
    global interpreter, input_details, output_details, is_uint8
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (320, 320))
    if is_uint8:
        tensor = np.expand_dims(small, 0)
    else:
        tensor = np.expand_dims(small.astype(np.float32) / 255.0, 0)
    interpreter.set_tensor(input_details['index'], tensor)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]
    results = []
    h, w = frame.shape[:2]
    for i in range(len(scores)):
        if scores[i] < min_conf:
            continue
        cls_id = int(classes[i])
        cls_name = label_map.get(cls_id, f"未知({cls_id})")
        ymin, xmin, ymax, xmax = boxes[i]
        cx = int((xmin + xmax) / 2 * w)
        cy = int((ymin + ymax) / 2 * h)
        results.append({
            'id': cls_id,
            'name': cls_name,
            'center': (cx, cy),
            'score': float(scores[i])
        })
    return results


def main():
    global interpreter, input_details, output_details, is_uint8

    print("连接机器人...")
    u = ugot.UGOT()
    u.initialize('10.233.104.242')
    print("连接成功")

    print("打开摄像头...")
    u.open_camera()
    time.sleep(0.5)

    print("加载模型...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    is_uint8 = input_details['dtype'] == np.uint8
    load_label_map(LABEL_MAP_PATH)
    print("模型加载完成，开始循环识别")

    last_spoken = None
    last_spoken_time = 0

    try:
        while True:
            raw = u.read_camera_data()
            if raw is None:
                time.sleep(0.05)
                continue
            nparr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                time.sleep(0.05)
                continue

            results = detect(frame, min_conf=0.4)
            if results:
                for r in results:
                    now = time.time()
                    print(f"识别到: {r['name']}号标签 (置信度: {r['score']:.2f}) 位置: {r['center']}")
                    if r['id'] != last_spoken or now - last_spoken_time > 3:
                        u.play_audio_tts(f"{r['name']}号", 0, True)
                        last_spoken = r['id']
                        last_spoken_time = now
            else:
                print("未识别到标签")

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n手动退出")
    finally:
        try:
            u.release_models([])
        except Exception:
            pass
        print("程序结束")


if __name__ == "__main__":
    main()
