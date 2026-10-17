# -*- coding: utf-8 -*-
"""
区域标签识别测试脚本
功能：直接循环识别摄像头画面中的区域标签（one~six），识别到就播报
运行环境：tf-od（需要 tensorflow）
  C:/Users/16231/anaconda3/envs/tf-od/python.exe tf_test.py
"""

import time
import os
import re
import cv2
import numpy as np
from ugot import ugot

ROBOT_IP = '10.233.104.242'
MODEL_PATH = r'C:\Users\16231\Desktop\ros\pb_tflite\ssd_mobilenet_v2_fpn_320.tflite'
LABEL_MAP_PATH = r'C:\Users\16231\Desktop\ros\pb_tflite\label_map.pbtxt'

MIN_CONFIDENCE = 0.5
BROADCAST_COOLDOWN = 5
FRAME_INTERVAL = 0.1

LABEL_ZH = {
    'one': '一号区', 'two': '二号区', 'three': '三号区',
    'four': '四号区', 'five': '五号区', 'six': '六号区'
}

_interpreter = None
_input_details = None
_output_details = None
_is_uint8 = False
_label_map = {}
_model_loaded = False


def load_label_map(pbtxt_path):
    """加载 label_map.pbtxt，返回 {id: name}"""
    label_map = {}
    if not os.path.exists(pbtxt_path):
        print(f"[错误] label_map 文件不存在: {pbtxt_path}")
        return label_map
    try:
        with open(pbtxt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        items = re.findall(r'item\s*\{([^}]+)\}', content)
        for item in items:
            id_match = re.search(r'id:\s*(\d+)', item)
            name_match = re.search(r"name:\s*'([^']+)'", item)
            if id_match and name_match:
                label_map[int(id_match.group(1))] = name_match.group(1)
        print(f"[模型] label_map 加载成功: {label_map}")
    except Exception as e:
        print(f"[错误] 加载 label_map 失败: {e}")
    return label_map


def init_tflite_model():
    """初始化 TFLite 模型"""
    global _interpreter, _input_details, _output_details
    global _is_uint8, _label_map, _model_loaded

    if _model_loaded:
        return True

    try:
        import tensorflow as tf
        tflite = tf.lite
    except ImportError:
        print("[错误] 当前环境没有 tensorflow，请用 tf-od 环境运行")
        print("  命令: C:/Users/16231/anaconda3/envs/tf-od/python.exe tf_test.py")
        return False

    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        return False

    try:
        print("[模型] 正在加载 TFLite 模型（CPU多线程模式）...")
        _interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()[0]
        _output_details = _interpreter.get_output_details()
        _is_uint8 = _input_details['dtype'] == np.uint8
        _label_map = load_label_map(LABEL_MAP_PATH)
        _model_loaded = True
        print(f"[模型] TFLite 模型加载完成，输入类型: {'uint8' if _is_uint8 else 'float32'}")
        return True
    except Exception as e:
        print(f"[错误] 加载 TFLite 模型失败: {e}")
        return False


def detect_labels(frame, min_conf=MIN_CONFIDENCE):
    """
    用 TFLite 模型检测画面中的区域标签
    返回：检测到的目标列表 [{label_id, label_name, score, center_x, center_y, area}]
    """
    if not _model_loaded or _interpreter is None:
        return []

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (320, 320))

        if _is_uint8:
            tensor = np.expand_dims(small, 0)
        else:
            tensor = np.expand_dims(small.astype(np.float32) / 255.0, 0)

        _interpreter.set_tensor(_input_details['index'], tensor)
        _interpreter.invoke()

        boxes = _interpreter.get_tensor(_output_details[0]['index'])[0]
        scores = _interpreter.get_tensor(_output_details[1]['index'])[0]
        count = int(_interpreter.get_tensor(_output_details[2]['index'])[0])
        classes = _interpreter.get_tensor(_output_details[3]['index'])[0]

        h, w = frame.shape[:2]
        results = []
        for i in range(count):
            if scores[i] < min_conf:
                continue
            label_id = int(classes[i]) + 1
            label_name = _label_map.get(label_id, f"ID{label_id}")
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            results.append({
                'label_id': label_id,
                'label_name': label_name,
                'score': float(scores[i]),
                'center_x': center_x,
                'center_y': center_y,
                'area': area,
                'bbox': (x1, y1, x2, y2)
            })
        return results
    except Exception as e:
        print(f"[识别异常] {e}")
        return []


def main():
    print("=" * 50)
    print("  区域标签识别测试脚本")
    print(f"  机器人IP: {ROBOT_IP}")
    print(f"  模型: {os.path.basename(MODEL_PATH)}")
    print(f"  最低置信度: {MIN_CONFIDENCE}")
    print(f"  播报冷却: {BROADCAST_COOLDOWN}秒")
    print("=" * 50)

    print("\n[1/3] 连接机器人...")
    got = ugot.UGOT()
    got.initialize(ROBOT_IP)
    print("      连接成功")

    print("\n[2/3] 打开摄像头...")
    got.open_camera()
    time.sleep(0.5)
    print("      摄像头已开启")

    print("\n[3/3] 加载TFLite模型...")
    if not init_tflite_model():
        print("模型加载失败，程序退出")
        got.mecanum_stop()
        return
    print("      模型加载完成")

    got.play_audio_tts("标签识别已就绪", 0, True)
    print("\n开始循环识别（按 Ctrl+C 退出）...\n")

    last_broadcast = {}
    frame_count = 0

    try:
        while True:
            frame_data = got.read_camera_data()
            if frame_data is None:
                time.sleep(0.1)
                continue

            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                time.sleep(0.1)
                continue

            frame_count += 1

            results = detect_labels(frame)

            if results:
                best = max(results, key=lambda x: x['score'])
                label_zh = LABEL_ZH.get(best['label_name'], best['label_name'])

                print(f"[帧{frame_count}] 识别到: {label_zh} "
                      f"(置信度: {best['score']:.1%}, "
                      f"中心: ({best['center_x']}, {best['center_y']}), "
                      f"面积: {best['area']}px²)")

                now = time.time()
                last_time = last_broadcast.get(best['label_name'], 0)
                if now - last_time >= BROADCAST_COOLDOWN:
                    broadcast_text = f"识别到{label_zh}"
                    print(f"  >>> 播报: {broadcast_text}")
                    got.play_audio_tts(broadcast_text, 0, False)
                    last_broadcast[best['label_name']] = now
            else:
                if frame_count % 10 == 0:
                    print(f"[帧{frame_count}] 未识别到区域标签")

            time.sleep(FRAME_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n手动中断，程序退出")
    finally:
        print("清理资源...")
        got.mecanum_stop()
        try:
            got.close_camera()
        except Exception:
            pass
        print("程序结束")


if __name__ == "__main__":
    main()
