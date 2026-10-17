# -*- coding: utf-8 -*-
"""
test1.py - 单独测试标签面积识别和放置逻辑
运行环境：tf-od
用法：python test1.py
"""
import os
import time
import numpy as np
import cv2

try:
    import tensorflow as tf
    tflite = tf.lite
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[错误] tensorflow 未安装，请用 tf-od 环境运行")
    exit(1)

from ugot import ugot

AREA_1 = 9000
AREA_2 = 11000
AREA_3 = 15000
AREA_4 = 17000
AREA_5 = 21000

FORWARD_SPEED = 5
SETTLE_DELAY = 0.3

EXTRA_1 = 5
EXTRA_2 = 4
EXTRA_3 = 3
EXTRA_4 = 2
EXTRA_5 = 1

BACKWARD_DISTANCE = 6


ROBOT_IP = '10.233.104.242'
MODEL_PATH = r'C:\Users\16231\Desktop\ros\pb_tflite\ssd_mobilenet_v2_fpn_320.tflite'
LABEL_MAP_PATH = r'C:\Users\16231\Desktop\ros\pb_tflite\label_map.pbtxt'

_interpreter = None
_input_details = None
_output_details = None
_is_uint8 = False
_label_map = {}


def load_label_map(pbtxt_path):
    label_map = {}
    if not os.path.exists(pbtxt_path):
        print(f"[错误] label_map 文件不存在: {pbtxt_path}")
        return label_map
    import re
    with open(pbtxt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    for item in content.split('item {'):
        id_match = re.search(r'id:\s*(\d+)', item)
        name_match = re.search(r'name:\s*[\'"]([^\'"]+)[\'"]', item)
        if id_match and name_match:
            label_map[int(id_match.group(1))] = name_match.group(1)
    print(f"[模型] 标签映射: {label_map}")
    return label_map


def init_model():
    global _interpreter, _input_details, _output_details, _is_uint8, _label_map
    print("[模型] 加载 TFLite 模型...")
    _interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    _interpreter.allocate_tensors()
    _input_details = _interpreter.get_input_details()[0]
    _output_details = _interpreter.get_output_details()
    _is_uint8 = _input_details['dtype'] == np.uint8
    _label_map = load_label_map(LABEL_MAP_PATH)
    print("[模型] 加载完成")


def detect_labels(frame, min_conf=0.8):
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
    try:
        classes = _interpreter.get_tensor(_output_details[2]['index'])[0]
    except IndexError:
        return []

    frame_h, frame_w = frame.shape[:2]
    results = []
    for i in range(len(scores)):
        if scores[i] < min_conf:
            continue
        cls_id = int(classes[i])
        ymin, xmin, ymax, xmax = boxes[i]
        bx = int(xmin * frame_w)
        by = int(ymin * frame_h)
        bw = int((xmax - xmin) * frame_w)
        bh = int((ymax - ymin) * frame_h)
        area = bw * bh
        results.append({
            'label_id': cls_id,
            'area': area,
            'score': float(scores[i])
        })
    return results


def main():
    got = ugot.UGOT()
    got.initialize(ROBOT_IP)
    print("机器人连接成功")

    print("机器复位...")
    got.mechanical_clamp_release()
    time.sleep(1.0)
    got.mechanical_single_joint_control(3, 60, 350)
    time.sleep(1.0)
    got.mechanical_single_joint_control(2, 60, 350)
    time.sleep(1.0)
    got.mechanical_single_joint_control(1, 0, 350)
    time.sleep(1.0)

    print("等待3秒...")
    time.sleep(3)

    print("爪子夹紧")
    got.mechanical_clamp_close()
    time.sleep(1.5)

    init_model()

    print("打开摄像头...")
    got.open_camera()
    time.sleep(1)

    print("\n========== 开始面积识别测试 ==========")
    print(f"档位: <{AREA_1}前进{EXTRA_1} | {AREA_1}~{AREA_2}前{EXTRA_2} | {AREA_2}~{AREA_3}前{EXTRA_3} | {AREA_3}~{AREA_4}前{EXTRA_4} | {AREA_4}~{AREA_5}前{EXTRA_5} | >{AREA_5}退{BACKWARD_DISTANCE}")
    print(f"速度: {FORWARD_SPEED}")
    
    print("========================================\n")

    try:
        while True:
            frame = got.read_camera_data()
            if frame is None:
                time.sleep(0.5)
                continue

            nparr = np.frombuffer(frame, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if data is None:
                continue

            labels = detect_labels(data)
            if not labels:
                print("未识别到标签，前进...")
                got.mecanum_move_speed_times(0, FORWARD_SPEED, EXTRA_1, 1)
                time.sleep(SETTLE_DELAY)
                continue

            target = max(labels, key=lambda x: x['area'])
            area = target['area']
            label_id = target['label_id']

            if area < AREA_1:
                print(f"标签{label_id} 面积: {area} < {AREA_1}，前进{EXTRA_1}cm")
                got.mecanum_move_speed_times(0, FORWARD_SPEED, EXTRA_1, 1)
                time.sleep(SETTLE_DELAY)

            elif area <= AREA_5:
                print(f"标签{label_id} 面积: {area}，停止前进")
                got.mecanum_stop()

                extra = 0
                if area <= AREA_2:
                    extra = EXTRA_2
                    print(f"面积{area}在{AREA_1}~{AREA_2}之间，先前进{EXTRA_2}cm...")
                elif area <= AREA_3:
                    extra = EXTRA_3
                    print(f"面积{area}在{AREA_2}~{AREA_3}之间，先前进{EXTRA_3}cm...")
                elif area <= AREA_4:
                    extra = EXTRA_4
                    print(f"面积{area}在{AREA_3}~{AREA_4}之间，先前进{EXTRA_4}cm...")
                else:
                    extra = EXTRA_5
                    print(f"面积{area}在{AREA_4}~{AREA_5}之间，先前进{EXTRA_5}cm...")

                if extra > 0:
                    got.mecanum_move_speed_times(0, FORWARD_SPEED, extra, 1)
                    time.sleep(0.5)
                    got.mecanum_stop()

                print("释放物块...")
                got.mechanical_single_joint_control(3, 16, 150)
                time.sleep(0.5)
                got.mechanical_single_joint_control(2, -80, 150)
                time.sleep(0.5)
                got.mechanical_clamp_release()
                time.sleep(1)
                got.mechanical_single_joint_control(3, 45, 150)
                time.sleep(0.5)
                got.mechanical_single_joint_control(2, 0, 150)
                time.sleep(0.5)

                print("完成！")
                break

            else:
                print(f"标签{label_id} 面积: {area} > {AREA_5}，后退{BACKWARD_DISTANCE}cm")
                got.mecanum_stop()
                got.mecanum_move_speed_times(1, FORWARD_SPEED, BACKWARD_DISTANCE, 1)
                time.sleep(0.5)
                got.mecanum_stop()
                break

    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        got.mecanum_stop()
        try:
            got.close_camera()
        except Exception:
            pass
        print("测试结束")


if __name__ == "__main__":
    main()
