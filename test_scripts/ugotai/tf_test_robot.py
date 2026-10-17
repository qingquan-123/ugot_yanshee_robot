# -*- coding: utf-8 -*-
"""
区域标签识别测试脚本（机器人内部端侧运行版）
运行环境：机器人内部 Linux 系统（python3 + tflite-runtime + Pillow + ugot SDK）
模型路径：/root/models/ssd_mobilenet_v2_fpn_320.tflite
上传方式：通过 MobaXterm 把本文件传到 /root/ 目录
运行方式：ssh 登录机器人后执行 python3 /root/tf_test_robot.py
"""

import time
import os
import re
import sys
import io

try:
    import numpy as np
except ImportError as e:
    print(f"[错误] 缺少 numpy: {e}")
    sys.exit(1)

try:
    from PIL import Image
except ImportError as e:
    print(f"[错误] 缺少 Pillow: {e}")
    print("请安装: 下载 Pillow aarch64 wheel 并手动解压到 site-packages")
    sys.exit(1)

try:
    import tflite_runtime.interpreter as tflite
except ImportError as e:
    print(f"[错误] 缺少 tflite-runtime: {e}")
    sys.exit(1)

try:
    from ugot import ugot
except ImportError as e:
    print(f"[错误] 无法导入 ugot SDK: {e}")
    sys.exit(1)


MODEL_PATH = '/root/models/ssd_mobilenet_v2_fpn_320.tflite'
LABEL_MAP_PATH = '/root/models/label_map.pbtxt'

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
    """初始化 TFLite 模型（用 tflite-runtime）"""
    global _interpreter, _input_details, _output_details
    global _is_uint8, _label_map, _model_loaded

    if _model_loaded:
        return True

    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        return False

    try:
        print(f"[模型] 正在加载本地 TFLite 模型: {MODEL_PATH}")
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


def decode_frame(frame_data):
    """
    用 Pillow 解码摄像头帧数据（JPEG 字节）
    返回：RGB numpy 数组 (height, width, 3)
    """
    img = Image.open(io.BytesIO(frame_data))
    return np.array(img)


def preprocess_for_tflite(frame_rgb, target_size=(320, 320)):
    """
    图像预处理：resize 到目标尺寸，构造模型输入张量
    参数：frame_rgb - RGB numpy 数组
    返回：模型输入张量 (1, 320, 320, 3)
    """
    img = Image.fromarray(frame_rgb)
    img_resized = img.resize(target_size, Image.BILINEAR)
    small = np.array(img_resized)

    if _is_uint8:
        tensor = np.expand_dims(small, 0)
    else:
        tensor = np.expand_dims(small.astype(np.float32) / 255.0, 0)
    return tensor


def detect_labels(frame_rgb, min_conf=MIN_CONFIDENCE):
    """
    用 TFLite 模型检测画面中的区域标签
    参数：frame_rgb - RGB numpy 数组
    返回：检测到的目标列表 [{label_id, label_name, score, center_x, center_y, area}]
    """
    if not _model_loaded or _interpreter is None:
        return []

    try:
        tensor = preprocess_for_tflite(frame_rgb)

        _interpreter.set_tensor(_input_details['index'], tensor)
        _interpreter.invoke()

        boxes = _interpreter.get_tensor(_output_details[0]['index'])[0]
        scores = _interpreter.get_tensor(_output_details[1]['index'])[0]
        count = int(_interpreter.get_tensor(_output_details[2]['index'])[0])
        classes = _interpreter.get_tensor(_output_details[3]['index'])[0]

        h, w = frame_rgb.shape[:2]
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
    print("  区域标签识别测试（机器人端侧运行版）")
    print(f"  模型: {MODEL_PATH}")
    print(f"  最低置信度: {MIN_CONFIDENCE}")
    print(f"  播报冷却: {BROADCAST_COOLDOWN}秒")
    print("=" * 50)

    print("\n[1/3] 初始化机器人...")
    got = ugot.UGOT()
    try:
        got.initialize('127.0.0.1')
        print("      机器人本地连接成功")
    except Exception as e:
        print(f"      本地连接失败({e})，尝试不initialize直接使用...")

    print("\n[2/3] 打开摄像头...")
    try:
        got.open_camera()
        time.sleep(0.5)
        print("      摄像头已开启")
    except Exception as e:
        print(f"      摄像头开启失败: {e}")
        return

    print("\n[3/3] 加载本地TFLite模型...")
    if not init_tflite_model():
        print("模型加载失败，程序退出")
        return
    print("      模型加载完成")

    try:
        got.play_audio_tts("标签识别已就绪", 0, True)
    except Exception as e:
        print(f"播报失败（不影响识别）: {e}")

    print("\n开始循环识别（按 Ctrl+C 退出）...\n")

    last_broadcast = {}
    frame_count = 0

    try:
        while True:
            try:
                frame_data = got.read_camera_data()
            except Exception as e:
                print(f"读取摄像头失败: {e}")
                time.sleep(0.5)
                continue

            if frame_data is None:
                time.sleep(0.1)
                continue

            try:
                frame_rgb = decode_frame(frame_data)
            except Exception as e:
                print(f"解码帧失败: {e}")
                time.sleep(0.1)
                continue

            frame_count += 1

            results = detect_labels(frame_rgb)

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
                    try:
                        got.play_audio_tts(broadcast_text, 0, False)
                    except Exception as e:
                        print(f"  播报失败: {e}")
                    last_broadcast[best['label_name']] = now
            else:
                if frame_count % 10 == 0:
                    print(f"[帧{frame_count}] 未识别到区域标签")

            time.sleep(FRAME_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n手动中断，程序退出")
    finally:
        print("清理资源...")
        try:
            got.mecanum_stop()
        except Exception:
            pass
        try:
            got.close_camera()
        except Exception:
            pass
        print("程序结束")


if __name__ == "__main__":
    main()
