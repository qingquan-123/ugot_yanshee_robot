# 修复Windows控制台中文闪退/无输出
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import time
import os
import tensorflow as tf
from typing import Optional, List, Dict
from apriltag_opt import wttag

tflite = tf.lite
from ugot import ugot

# ===================== 全局变量 =====================
grasp_flag: int = 0
got: Optional[ugot.UGOT] = None

# 自定义模型相关全局变量
interp: Optional[tflite.Interpreter] = None
inp: Optional[dict] = None
out: Optional[list] = None
is_uint8: bool = False
label_map: Dict[int, str] = {}

# 视觉模式状态标记：解决模型覆盖、巡线失效问题
_line_model_loaded = False
_current_vision_mode = None  # 可选值: None / "raw" / "line" / "apriltag"

# 文件路径
MODEL_PATH = r"C:\Users\16231\Desktop\ros\pb_tflite\ssd_mobilenet_v2_fpn_320.tflite"
LABEL_MAP_PATH = r"C:\Users\16231\Desktop\ros\pb_tflite\label_map.pbtxt"
IMG_SIZE = 320
CONF_TH = 0.5  # 标签置信度阈值
LABEL_AREA_THRESHOLD = 14000  # 标签面积到位阈值

# ========== 运动参数 ==========
FIVE_TURN_SPEED = 113
FIVE_TURN_TIME = 186
FIVE_TRANS_SPEED = 90
FIVE_BACK_SPEED = 150

# ========== 视觉搜索通用参数 ==========
_MORPH_KERNEL = np.ones((5, 5), np.uint8)
ALIGN_THRESHOLD = 18    # 对齐阈值，单位像素
ALIGN_SPEED = 3         # 横移对齐速度
ALIGN_DURATION = 1      # 横移对齐时长
INVALID_EDGE_RATIO = 0.2  # 左右无效区域比例
STRICT_MIN_AREA = 2000    # 目标最小面积阈值
LOST_FRAME_THRESHOLD = 30 # 连续丢帧阈值
MAX_SEARCH_TIME_THREE = 3 # 3区最大搜索次数
MAX_INIT_FRAMES = 5       # 初始化帧数
MAX_CAMERA_CHECKS = 5     # 摄像头重试次数


def _get_color_ranges() -> dict:
    lower_red1 = np.array([0, 167, 132])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([161, 167, 132])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([33, 113, 62])
    upper_green = np.array([76, 255, 255])
    lower_blue = np.array([100, 167, 95])
    upper_blue = np.array([143, 255, 255])
    return {
        '红色': [(lower_red1, upper_red1), (lower_red2, upper_red2)],
        '绿色': [(lower_green, upper_green)],
        '蓝色': [(lower_blue, upper_blue)]
    }

# 颜色范围全局常量
COLOR_RANGES = _get_color_ranges()


def extract_task_info(task_message: str):
    """
    从语音识别文本中提取「目标颜色」和「存储区编号」
    支持：阿拉伯数字（1-6）、汉字数字（一-六）
    """
    color_map = {
        "红色": ["红色", "红颜色", "红的", "红"],
        "绿色": ["绿色", "绿颜色", "绿的", "绿"],
        "蓝色": ["蓝色", "蓝颜色", "蓝的", "蓝"]
    }
    target_color = None
    for color_name, keywords in color_map.items():
        for kw in keywords:
            if kw in task_message:
                target_color = color_name
                break
        if target_color:
            break

    chinese_num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
    target_store = None

    # 匹配阿拉伯数字
    for num in range(1, 7):
        if f"{num}号" in task_message:
            target_store = num
            break

    # 匹配汉字数字
    if target_store is None:
        for cn_char, num in chinese_num.items():
            if f"{cn_char}号" in task_message:
                target_store = num
                break

    return target_color, target_store


def load_label_map(pbtxt_path: str) -> Dict[int, str]:
    """解析 label_map.pbtxt -> {id: name}"""
    result_map = {}
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        block = {}
        for line in f:
            line = line.strip()
            if line.startswith("item {"):
                block = {}
            elif line.startswith("}"):
                if "id" in block and "name" in block:
                    result_map[int(block["id"])] = block["name"].strip('"').strip("'")
            elif ":" in line:
                key, val = line.split(":", 1)
                block[key.strip()] = val.strip()
    return result_map


def _read_camera_frame():
    """统一封装摄像头帧读取"""
    global got
    frame = got.read_camera_data()
    if frame is None:
        return None
    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return data


def detect_color_objects(frame, min_area=500) -> List[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results = []
    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                results.append({
                    'color': color_name,
                    'center': (center_x, center_y),
                    'area': area,
                    'rect': (x, y, w, h)
                })
    return results


def detect_label_objects(frame, min_conf=CONF_TH) -> List[dict]:
    global interp, inp, out, is_uint8, label_map
    results = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    if is_uint8:
        tensor = np.expand_dims(small, 0)
    else:
        tensor = np.expand_dims(small.astype(np.float32) / 255.0, 0)

    interp.set_tensor(inp['index'], tensor)
    interp.invoke()

    boxes = interp.get_tensor(out[0]['index'])[0]
    scores = interp.get_tensor(out[1]['index'])[0]
    try:
        classes = interp.get_tensor(out[2]['index'])[0]
    except IndexError:
        classes = None
    if classes is None:
        return results

    frame_h, frame_w = frame.shape[:2]
    for i in range(len(scores)):
        if scores[i] < min_conf:
            continue
        cls_id = int(classes[i])
        cls_name = label_map.get(cls_id, f"未知({cls_id})")
        ymin, xmin, ymax, xmax = boxes[i]
        bx = int(xmin * frame_w)
        by = int(ymin * frame_h)
        bw = int((xmax - xmin) * frame_w)
        bh = int((ymax - ymin) * frame_h)
        area = bw * bh
        center_x = int(bx + bw / 2)
        center_y = int(by + bh / 2)
        results.append({
            'label_id': cls_id,
            'label_name': cls_name,
            'center': (center_x, center_y),
            'area': area,
            'rect': (bx, by, bw, bh),
            'score': scores[i]
        })
    return results


def calculate_color_ratio_in_bottom_region(frame, target_color, region_ratio=1 / 4) -> float:
    height, width = frame.shape[:2]
    bottom_start = int(height * (1 - region_ratio))
    bottom_region = frame[bottom_start:height, 0:width]
    hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)
    target_ranges = COLOR_RANGES.get(target_color, [])
    if not target_ranges:
        return 0.0
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in target_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    total_pixels = mask.shape[0] * mask.shape[1]
    if total_pixels == 0:
        return 0.0
    return cv2.countNonZero(mask) / total_pixels


def go_forward(target_color, speed=5):
    """视觉引导向前靠近目标（颜色物块抓取用）"""
    global got
    while True:
        data = _read_camera_frame()
        if data is None:
            print("未获取到摄像头数据，继续尝试...", flush=True)
            time.sleep(0.1)
            continue

        height, width = data.shape[:2]
        detected_objects = detect_color_objects(data)
        target_objects = [obj for obj in detected_objects if obj['color'] == target_color]

        if not target_objects:
            got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.1)
            continue

        target_obj = max(target_objects, key=lambda x: x['area'])
        x, y, w, h = target_obj['rect']
        bottom_y = y + h
        area = target_obj['area']
        stop_y_threshold = height * 0.9
        stop_area_threshold = height * width * 0.09
        color_ratio = calculate_color_ratio_in_bottom_region(data, target_color, 1 / 4)

        print(f"目标底部Y: {bottom_y:.0f}/{height}, 面积占比: {area / (height * width):.2%}, 底部占比: {color_ratio:.2%}", flush=True)

        if (bottom_y >= stop_y_threshold and area >= stop_area_threshold) or color_ratio >= 0.39:
            print("到达抓取位置，停止前进", flush=True)
            got.mecanum_stop()
            break
        else:
            if bottom_y >= height * 0.7 or area >= stop_area_threshold * 0.6:
                got.mecanum_move_speed_times(0, 3, 1, 0)
            else:
                got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.1)


def go_forward_label(target_id, speed=5):
    """标签版前进到位（存储区放货用，根据标签面积判断距离）"""
    global got
    while True:
        data = _read_camera_frame()
        if data is None:
            time.sleep(0.1)
            continue

        label_list = detect_label_objects(data)
        target_list = [obj for obj in label_list if obj['label_id'] == target_id]
        if not target_list:
            print("前进中丢失标签，缓慢前进...", flush=True)
            got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.3)
            continue

        target = max(target_list, key=lambda x: x['area'])
        print(f"前进中 - 标签面积: {target['area']:.0f}", flush=True)
        if target['area'] >= LABEL_AREA_THRESHOLD:
            print("标签面积达标，停止前进", flush=True)
            got.mecanum_stop()
            break
        else:
            got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.3)


def grasp(got, flag):
    """抓取动作"""
    print(f"执行抓取动作，当前区域 flag={flag}", flush=True)
    got.mechanical_single_joint_control(3, -45, 100)
    got.mechanical_single_joint_control(2, -25, 100)
    got.mechanical_clamp_release()
    time.sleep(0.1)

    got.mecanum_move_speed_times(1, 20,13, 1)

    got.mechanical_single_joint_control(3, 15, 100)
    got.mechanical_single_joint_control(2, -90, 100)
    time.sleep(0.6)

    got.mecanum_move_speed_times(0, 8,10, 1)
    time.sleep(0.1)

    got.mecanum_turn_speed_times(2, 11, 1, 2)
    got.mecanum_move_speed_times(0, 5, 3, 1)
    time.sleep(0.1)

    got.mechanical_clamp_close()
    time.sleep(0.1)
    got.mechanical_clamp_close()
    time.sleep(0.1)
    got.mecanum_turn_speed_times(3, 11, 3, 2)

    got.mecanum_move_speed_times(1, 10, 2, 1)
    time.sleep(0.1)
    got.mechanical_single_joint_control(3, 45, 100)
    time.sleep(0.1)
    got.mecanum_move_speed_times(1, 10, 10, 1)
    time.sleep(0.1)

    got.mechanical_single_joint_control(2, 0, 100)
    got.mechanical_single_joint_control(2, 40, 100)
    got.mechanical_single_joint_control(2, 90, 100)
    got.mechanical_single_joint_control(3, -90, 100)
    print("抓取完成", flush=True)


def grasp3(got, flag):
    """3号区专属抓取动作"""
    print(f"执行3号区专属抓取动作，当前区域 flag={flag}", flush=True)
    got.mechanical_single_joint_control(3, -70, 100)
    got.mechanical_single_joint_control(2, -25, 100)
    got.mechanical_clamp_release()
    time.sleep(0.1)

    got.mecanum_move_speed_times(1, 20, 11, 1)
    got.mecanum_move_speed_times(1, 20, 1, 1)

    got.mechanical_single_joint_control(3, 15, 100)
    got.mechanical_single_joint_control(2, -90, 100)
    time.sleep(0.6)

    got.mecanum_move_speed_times(0, 10, 10, 1)
    time.sleep(0.1)
    got.mecanum_move_speed_times(0, 5, 7, 1)
    time.sleep(0.1)

    got.mecanum_turn_speed_times(2, 11, 3, 2)
    got.mecanum_move_speed_times(0, 5, 2, 1)
    time.sleep(0.1)

    got.mechanical_clamp_close()
    time.sleep(0.1)
    got.mechanical_clamp_close()
    time.sleep(0.1)
    got.mecanum_turn_speed_times(3, 11, 5, 2)

    got.mecanum_move_speed_times(1, 10, 2, 1)
    time.sleep(0.1)
    got.mechanical_single_joint_control(3, 45, 100)
    time.sleep(0.1)
    got.mecanum_move_speed_times(1, 10, 10, 1)
    time.sleep(0.1)

    got.mechanical_single_joint_control(2, 0, 100)
    got.mechanical_single_joint_control(2, 40, 100)
    got.mechanical_single_joint_control(2, 90, 100)
    got.mechanical_single_joint_control(3, -90, 100)
    print("3号区抓取完成", flush=True)


def release_block():
    """机械臂放下物块"""
    global got
    print("正在放下物块...", flush=True)

    got.mechanical_single_joint_control(2, -100, 50)
    got.mechanical_single_joint_control(3, 16, 50)
    time.sleep(0.6)

    got.mechanical_clamp_release()
    time.sleep(0.2)

    got.mechanical_single_joint_control(2, -25, 100)
    got.mechanical_single_joint_control(3, -70, 100)
    time.sleep(0.5)

    got.mechanical_single_joint_control(2, 80, 100)
    got.mechanical_single_joint_control(3, -80, 100)
    print("物块已放下", flush=True)


def five_store(target_store, num, search_time, flag, task, flag_diction):
    """抓取完成后返回中转区域（支持1-6号区回位，保留原接口兼容）"""
    global got
    print("flag:", flag, flush=True)

    if flag == 6:
        print("6号区抓取完成，返回中转区", flush=True)
        got.mecanum_translate_speed_times(-90, 20, 13, 1)
    elif flag == 5:
        print("5号区抓取完成，返回中转区", flush=True)
        got.mecanum_translate_speed_times(-90, 50, 43, 1)
    elif flag == 4:
        print("4号区抓取完成，返回中转区", flush=True)
        got.mecanum_translate_speed_times(-90, 50, 68, 1)
    elif flag == 2:
        print("2号区抓取完成，返回中转区", flush=True)
        got.mecanum_move_speed_times(1, FIVE_BACK_SPEED, 59, 1)
        got.mecanum_turn_speed_times(2, 150, 85, 2)
    elif flag == 1:
        print("1号区抓取完成，返回中转区", flush=True)
        got.mecanum_move_speed_times(1, FIVE_BACK_SPEED, 56, 1)
        got.mecanum_turn_speed_times(2, 150, 86, 2)
        got.mecanum_move_speed_times(0, 30, 15, 1)
    elif flag == 3:
        print("已抓取3区物块，返回中转区", flush=True)
        got.mecanum_move_speed_times(1, FIVE_BACK_SPEED, 30, 1)
        got.mecanum_turn_speed_times(2, 150, 43, 2)
        time.sleep(0.5)
        got.mecanum_translate_speed_times(90, 100, 55, 1)

    print("返回初始区域完成", flush=True)


# ===================== 第一次搜索：颜色物块搜索抓取 =====================
def get_bbox(target_color):
    """
    颜色物块搜索+抓取（新版分区搜索框架：6→5→4→3→2→1区依次搜索）
    返回：detected_objects, search_time, flag, flag_diction
    """
    global got, _current_vision_mode
    _current_vision_mode = "raw"  # 标记当前为原始帧模式

    state = "INITIALIZING"
    search_time = 0
    flag_diction = None
    init_frames = 0
    camera_check_count = 0
    flag = 6  # 从6号区开始搜索
    detected_objects = []
    current_zone_id = None
    lost_frame_count = 0

    print("=== 开始搜索颜色物块 ===", flush=True)
    try:
        while True:
            data = _read_camera_frame()
            if data is None:
                camera_check_count += 1
                print(f"未获取到摄像头数据，重试 {camera_check_count}/{MAX_CAMERA_CHECKS}", flush=True)
                if camera_check_count >= MAX_CAMERA_CHECKS:
                    print("摄像头数据获取失败，退出颜色检测", flush=True)
                    break
                time.sleep(0.1)
                continue
            camera_check_count = 0

            detected_objects = detect_color_objects(data)
            target_objects = [obj for obj in detected_objects if obj['color'] == target_color]
            height, width = data.shape[:2]
            center_x = width // 2

            # 非1/6号区启用有效区域过滤+面积过滤
            if flag not in (1, 6):
                valid_left = width * INVALID_EDGE_RATIO
                valid_right = width * (1 - INVALID_EDGE_RATIO)
                strict_area = 500 if flag == 3 else STRICT_MIN_AREA
                target_objects = [
                    obj for obj in target_objects
                    if valid_left <= obj['center'][0] <= valid_right
                       and obj['area'] >= strict_area
                ]

            # 状态1：初始化
            if state == "INITIALIZING":
                init_frames += 1
                print(f"摄像头初始化中... ({init_frames}/{MAX_INIT_FRAMES})", flush=True)
                if init_frames >= MAX_INIT_FRAMES:
                    print("摄像头初始化完成，开始搜索 6 号区目标", flush=True)
                    state = "FINDING_TARGET"

            # 状态2：搜索目标
            elif state == "FINDING_TARGET":
                if flag == 6:
                    if target_objects:
                        current_zone_id = 6
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"6号区检测到目标，偏差: {deviation_x}", flush=True)
                        got.mecanum_translate_speed_times(90, 10, 15, 1)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("6号区无目标，向右平移搜索5号区", flush=True)
                        got.mecanum_translate_speed_times(90, 10, 40, 1)
                        time.sleep(0.1)
                        flag = 5

                elif flag == 5:
                    if target_objects:
                        current_zone_id = 5
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"5号区检测到有效目标，偏差: {deviation_x}", flush=True)
                        got.mecanum_translate_speed_times(-90, 20, 2, 1)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("5号区无有效目标，向右平移搜索4号区", flush=True)
                        got.mecanum_translate_speed_times(90, 10, 25, 1)
                        time.sleep(0.1)
                        flag = 4

                elif flag == 4:
                    if target_objects:
                        current_zone_id = 4
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"4号区检测到有效目标，偏差: {deviation_x}", flush=True)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("4号区无有效目标，转向搜索3号区", flush=True)
                        got.mecanum_turn_speed_times(3, 90, 34, 2)
                        flag = 3
                        search_time = 0

                elif flag == 3:
                    if target_objects:
                        current_zone_id = 3
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"3号区检测到有效目标，偏差: {deviation_x}", flush=True)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        search_time += 1
                        print(f"3区寻找目标... (搜索次数: {search_time})", flush=True)
                        got.mecanum_stop()

                        if search_time > MAX_SEARCH_TIME_THREE:
                            print("3区域没有有效目标色块，切换至2区搜索", flush=True)
                            search_time = 0
                            flag = 2
                            got.mecanum_turn_speed_times(3, 90, 51, 2)
                            time.sleep(0.05)
                            print("旋转后检测目标", flush=True)

                            for step in range(1):
                                data = _read_camera_frame()
                                if data is None:
                                    continue
                                detected_objects = detect_color_objects(data)
                                target_objects = [obj for obj in detected_objects if obj['color'] == target_color]
                                valid_left = width * INVALID_EDGE_RATIO
                                valid_right = width * (1 - INVALID_EDGE_RATIO)
                                target_objects = [
                                    obj for obj in target_objects
                                    if valid_left <= obj['center'][0] <= valid_right
                                       and obj['area'] >= STRICT_MIN_AREA
                                ]
                                if target_objects:
                                    print(f"旋转过程中检测到有效目标！", flush=True)
                                    current_zone_id = 2
                                    target_obj = max(target_objects, key=lambda x: x['area'])
                                    deviation_x = target_obj['center'][0] - center_x
                                    state = "ADJUSTING_LATERAL"
                                    got.mecanum_stop()
                                    lost_frame_count = 0
                                    break

                elif flag == 2:
                    if target_objects:
                        current_zone_id = 2
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"2号区检测到有效目标，偏差: {deviation_x}", flush=True)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("2号区无有效目标，向右平移搜索1号区", flush=True)
                        search_time += 1
                        got.mecanum_translate_speed_times(90, 15, 19, 1)
                        time.sleep(0.1)
                        flag = 1

                elif flag == 1:
                    if target_objects:
                        current_zone_id = 1
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"1号区检测到有效目标，偏差: {deviation_x}", flush=True)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("1号区当前位置无目标，启动阶梯式平移搜索", flush=True)
                        found_target = False
                        step_speed = 5
                        step_duration = 2

                        # 向右搜索3次
                        for step in range(3):
                            print(f"1号区右移搜索 第{step+1}/3次", flush=True)
                            got.mecanum_translate_speed_times(90, step_speed, step_duration, 0)
                            time.sleep(0.15)

                            data = _read_camera_frame()
                            if data is None:
                                continue
                            new_detected = detect_color_objects(data)
                            new_targets = [obj for obj in new_detected if obj['color'] == target_color]

                            if new_targets:
                                current_zone_id = 1
                                target_obj = max(new_targets, key=lambda x: x['area'])
                                deviation_x = target_obj['center'][0] - center_x
                                print(f"右移第{step+1}次检测到目标，偏差: {deviation_x}", flush=True)
                                state = "ADJUSTING_LATERAL"
                                got.mecanum_stop()
                                lost_frame_count = 0
                                found_target = True
                                break
                        if found_target:
                            continue

                        # 向左搜索5次
                        print("右移3次未找到目标，切换向左平移搜索，共5次", flush=True)
                        for step in range(5):
                            print(f"1号区左移搜索 第{step+1}/5次", flush=True)
                            got.mecanum_translate_speed_times(-90, step_speed, step_duration, 0)
                            time.sleep(0.15)

                            data = _read_camera_frame()
                            if data is None:
                                continue
                            new_detected = detect_color_objects(data)
                            new_targets = [obj for obj in new_detected if obj['color'] == target_color]

                            if new_targets:
                                current_zone_id = 1
                                target_obj = max(new_targets, key=lambda x: x['area'])
                                deviation_x = target_obj['center'][0] - center_x
                                print(f"左移第{step+1}次检测到目标，偏差: {deviation_x}", flush=True)
                                state = "ADJUSTING_LATERAL"
                                got.mecanum_stop()
                                lost_frame_count = 0
                                found_target = True
                                break
                        if found_target:
                            continue

                        print("1号区左右平移搜索完毕，未找到目标，停止搜索", flush=True)
                        got.mecanum_stop()
                        state = "COMPLETED"
                        break

            # 状态3：左右对齐
            elif state == "ADJUSTING_LATERAL":
                if not target_objects:
                    lost_frame_count += 1
                    print(f"当前帧未检测到目标，连续丢帧：{lost_frame_count}/{LOST_FRAME_THRESHOLD}", flush=True)
                    if lost_frame_count >= LOST_FRAME_THRESHOLD:
                        print("连续多帧未检测到，返回搜索态", flush=True)
                        state = "FINDING_TARGET"
                        lost_frame_count = 0
                    time.sleep(0.05)
                    continue

                lost_frame_count = 0
                target_obj = max(target_objects, key=lambda x: x['area'])
                target_center_x = target_obj['center'][0]
                target_center_y = target_obj['center'][1]
                deviation_x = target_center_x - center_x

                print(f"调整中 - 目标位置: ({target_center_x}, {target_center_y}), 中心位置: {center_x}, 偏差: {deviation_x}", flush=True)

                if abs(deviation_x) > ALIGN_THRESHOLD:
                    if deviation_x > ALIGN_THRESHOLD:
                        print("向右横移一点", flush=True)
                        got.mecanum_translate_speed_times(90, 4, 1, 1)
                    else:
                        print("向左横移一点", flush=True)
                        got.mecanum_translate_speed_times(-90, 4, 1, 1)
                else:
                    print("左右位置已调整完成！目标已对齐", flush=True)
                    flag_diction = "right" if deviation_x > 0 else "left"
                    got.mecanum_stop()

                    # 机械臂预备姿态
                    got.mechanical_single_joint_control(3, -70, 100)
                    got.mechanical_single_joint_control(2, -25, 100)
                    time.sleep(0.05)
                    got.mechanical_clamp_release()
                    time.sleep(0.05)

                    # 3号区额外前进补偿
                    if current_zone_id == 3:
                        got.mecanum_move_speed_times(0, 30, 12, 1)

                    # 前进到位并抓取
                    go_forward(target_color, speed=5)
                    if current_zone_id == 3:
                        grasp3(got, current_zone_id)
                    else:
                        grasp(got, current_zone_id)

                    print("抓取完成", flush=True)
                    state = "COMPLETED"
                    break

            print(f"状态: {state}, 当前区: flag={flag}, 检测到目标数: {len(target_objects)}", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print('-----KeyboardInterrupt', flush=True)
    except Exception as e:
        print(f"颜色检测过程中发生错误: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        got.mecanum_stop()
        print("颜色检测任务结束", flush=True)

    return detected_objects, search_time, current_zone_id if current_zone_id else flag, flag_diction


# ===================== 第二次搜索：标签存储区搜索放货 =====================
def search_label_and_place(target_id):
    """
    标签存储区搜索+放货（与get_bbox同框架：6→5→4→3→2→1区依次搜索）
    返回：detected_objects, search_time, flag, flag_diction
    """
    global got, label_map, _current_vision_mode
    _current_vision_mode = "raw"
    target_name = label_map.get(target_id, f"ID{target_id}")

    state = "INITIALIZING"
    search_time = 0
    flag_diction = None
    init_frames = 0
    camera_check_count = 0
    flag = 6  # 从6号区开始搜索
    detected_objects = []
    current_zone_id = None
    lost_frame_count = 0

    print(f"\n=== 开始搜索 {target_name} 标签存储区 ===", flush=True)
    try:
        while True:
            data = _read_camera_frame()
            if data is None:
                camera_check_count += 1
                print(f"未获取到摄像头数据，重试 {camera_check_count}/{MAX_CAMERA_CHECKS}", flush=True)
                if camera_check_count >= MAX_CAMERA_CHECKS:
                    print("摄像头数据获取失败，退出标签检测", flush=True)
                    break
                time.sleep(0.1)
                continue
            camera_check_count = 0

            detected_objects = detect_label_objects(data)
            target_objects = [obj for obj in detected_objects if obj['label_id'] == target_id]
            height, width = data.shape[:2]
            center_x = width // 2

            # 非1/6号区启用有效区域过滤+面积过滤
            if flag not in (1, 6):
                valid_left = width * INVALID_EDGE_RATIO
                valid_right = width * (1 - INVALID_EDGE_RATIO)
                strict_area = 500 if flag == 3 else STRICT_MIN_AREA
                target_objects = [
                    obj for obj in target_objects
                    if valid_left <= obj['center'][0] <= valid_right
                       and obj['area'] >= strict_area
                ]

            # 状态1：初始化
            if state == "INITIALIZING":
                init_frames += 1
                print(f"模型检测初始化中... ({init_frames}/{MAX_INIT_FRAMES})", flush=True)
                if init_frames >= MAX_INIT_FRAMES:
                    print("初始化完成，开始搜索 6 号区标签", flush=True)
                    state = "FINDING_TARGET"

            # 状态2：搜索目标
            elif state == "FINDING_TARGET":
                if flag == 6:
                    if target_objects:
                        current_zone_id = 6
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"6号区检测到{target_name}标签，偏差: {deviation_x}", flush=True)
                        got.mecanum_translate_speed_times(90, 10, 15, 1)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("6号区无目标标签，向右平移搜索5号区", flush=True)
                        got.mecanum_translate_speed_times(90, 10, 40, 1)
                        time.sleep(0.1)
                        flag = 5

                elif flag == 5:
                    if target_objects:
                        current_zone_id = 5
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"5号区检测到有效{target_name}标签，偏差: {deviation_x}", flush=True)
                        got.mecanum_translate_speed_times(-90, 20, 2, 1)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("5号区无有效目标标签，向右平移搜索4号区", flush=True)
                        got.mecanum_translate_speed_times(90, 10, 25, 1)
                        time.sleep(0.1)
                        flag = 4

                elif flag == 4:
                    if target_objects:
                        current_zone_id = 4
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"4号区检测到有效{target_name}标签，偏差: {deviation_x}", flush=True)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("4号区无有效目标标签，转向搜索3号区", flush=True)
                        got.mecanum_turn_speed_times(3, 90, 34, 2)
                        flag = 3
                        search_time = 0

                elif flag == 3:
                    if target_objects:
                        current_zone_id = 3
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"3号区检测到有效{target_name}标签，偏差: {deviation_x}", flush=True)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        search_time += 1
                        print(f"3区寻找标签... (搜索次数: {search_time})", flush=True)
                        got.mecanum_stop()

                        if search_time > MAX_SEARCH_TIME_THREE:
                            print("3区域没有有效目标标签，切换至2区搜索", flush=True)
                            search_time = 0
                            flag = 2
                            got.mecanum_turn_speed_times(3, 90, 51, 2)
                            time.sleep(0.05)
                            print("旋转后检测目标标签", flush=True)

                            for step in range(1):
                                data = _read_camera_frame()
                                if data is None:
                                    continue
                                detected_objects = detect_label_objects(data)
                                target_objects = [obj for obj in detected_objects if obj['label_id'] == target_id]
                                valid_left = width * INVALID_EDGE_RATIO
                                valid_right = width * (1 - INVALID_EDGE_RATIO)
                                target_objects = [
                                    obj for obj in target_objects
                                    if valid_left <= obj['center'][0] <= valid_right
                                       and obj['area'] >= STRICT_MIN_AREA
                                ]
                                if target_objects:
                                    print(f"旋转过程中检测到有效{target_name}标签！", flush=True)
                                    current_zone_id = 2
                                    target_obj = max(target_objects, key=lambda x: x['area'])
                                    deviation_x = target_obj['center'][0] - center_x
                                    state = "ADJUSTING_LATERAL"
                                    got.mecanum_stop()
                                    lost_frame_count = 0
                                    break

                elif flag == 2:
                    if target_objects:
                        current_zone_id = 2
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"2号区检测到有效{target_name}标签，偏差: {deviation_x}", flush=True)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("2号区无有效目标标签，向右平移搜索1号区", flush=True)
                        search_time += 1
                        got.mecanum_translate_speed_times(90, 15, 19, 1)
                        time.sleep(0.1)
                        flag = 1

                elif flag == 1:
                    if target_objects:
                        current_zone_id = 1
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"1号区检测到有效{target_name}标签，偏差: {deviation_x}", flush=True)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("1号区当前位置无目标标签，启动阶梯式平移搜索", flush=True)
                        found_target = False
                        step_speed = 5
                        step_duration = 2

                        # 向右搜索3次
                        for step in range(3):
                            print(f"1号区右移搜索 第{step+1}/3次", flush=True)
                            got.mecanum_translate_speed_times(90, step_speed, step_duration, 0)
                            time.sleep(0.15)

                            data = _read_camera_frame()
                            if data is None:
                                continue
                            new_detected = detect_label_objects(data)
                            new_targets = [obj for obj in new_detected if obj['label_id'] == target_id]

                            if new_targets:
                                current_zone_id = 1
                                target_obj = max(new_targets, key=lambda x: x['area'])
                                deviation_x = target_obj['center'][0] - center_x
                                print(f"右移第{step+1}次检测到标签，偏差: {deviation_x}", flush=True)
                                state = "ADJUSTING_LATERAL"
                                got.mecanum_stop()
                                lost_frame_count = 0
                                found_target = True
                                break
                        if found_target:
                            continue

                        # 向左搜索5次
                        print("右移3次未找到标签，切换向左平移搜索，共5次", flush=True)
                        for step in range(5):
                            print(f"1号区左移搜索 第{step+1}/5次", flush=True)
                            got.mecanum_translate_speed_times(-90, step_speed, step_duration, 0)
                            time.sleep(0.15)

                            data = _read_camera_frame()
                            if data is None:
                                continue
                            new_detected = detect_label_objects(data)
                            new_targets = [obj for obj in new_detected if obj['label_id'] == target_id]

                            if new_targets:
                                current_zone_id = 1
                                target_obj = max(new_targets, key=lambda x: x['area'])
                                deviation_x = target_obj['center'][0] - center_x
                                print(f"左移第{step+1}次检测到标签，偏差: {deviation_x}", flush=True)
                                state = "ADJUSTING_LATERAL"
                                got.mecanum_stop()
                                lost_frame_count = 0
                                found_target = True
                                break
                        if found_target:
                            continue

                        print("1号区左右平移搜索完毕，未找到目标标签，停止搜索", flush=True)
                        got.mecanum_stop()
                        state = "COMPLETED"
                        break

            # 状态3：左右对齐
            elif state == "ADJUSTING_LATERAL":
                if not target_objects:
                    lost_frame_count += 1
                    print(f"当前帧未检测到标签，连续丢帧：{lost_frame_count}/{LOST_FRAME_THRESHOLD}", flush=True)
                    if lost_frame_count >= LOST_FRAME_THRESHOLD:
                        print("连续多帧未检测到，返回搜索态", flush=True)
                        state = "FINDING_TARGET"
                        lost_frame_count = 0
                    time.sleep(0.05)
                    continue

                lost_frame_count = 0
                target_obj = max(target_objects, key=lambda x: x['area'])
                target_center_x = target_obj['center'][0]
                target_center_y = target_obj['center'][1]
                deviation_x = target_center_x - center_x

                print(f"对准中 - 标签位置: ({target_center_x}, {target_center_y}), 中心位置: {center_x}, 偏差: {deviation_x}", flush=True)

                if abs(deviation_x) > ALIGN_THRESHOLD:
                    if deviation_x > ALIGN_THRESHOLD:
                        print("向右横移一点", flush=True)
                        got.mecanum_translate_speed_times(90, ALIGN_SPEED, ALIGN_DURATION, 0)
                    else:
                        print("向左横移一点", flush=True)
                        got.mecanum_translate_speed_times(-90, ALIGN_SPEED, ALIGN_DURATION, 0)
                else:
                    print("左右对准完成！", flush=True)
                    flag_diction = "right" if deviation_x > 0 else "left"
                    got.mecanum_stop()

                    # 前进到位并放货
                    print("前进靠近存储区", flush=True)
                    go_forward_label(target_id, speed=6)
                    release_block()

                    print("放货完成", flush=True)
                    state = "COMPLETED"
                    break

            print(f"状态: {state}, 当前区: flag={flag}, 检测到{target_name}标签数: {len(target_objects)}", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print('-----KeyboardInterrupt', flush=True)
    except Exception as e:
        print(f"标签检测过程中发生错误: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        got.mecanum_stop()
        print("标签搜索任务结束", flush=True)

    return detected_objects, search_time, current_zone_id if current_zone_id else flag, flag_diction


def reset_line_vision(got):
    """
    安全复位巡线视觉：智能判断是否需要重载模型，避免运动中切换
    解决AprilTag覆盖模型后巡线失效的问题
    """
    global _line_model_loaded, _current_vision_mode
    print("============================ 巡线视觉复位 ===========================", flush=True)

    # 先关闭摄像头，清空原始帧占用
    try:
        got.close_camera()
        time.sleep(0.12)
    except Exception:
        pass
    got.open_camera()
    time.sleep(0.15)

    # 只有当前不是巡线模式时，才重新加载模型
    if _current_vision_mode != "line":
        print("当前非巡线模式，重新加载巡线模型...", flush=True)
        got.load_models(["line_recognition"])
        _line_model_loaded = True
        _current_vision_mode = "line"
        time.sleep(0.2)
    else:
        print("当前已是巡线模式，直接预热", flush=True)

    got.set_track_recognition_line(0)
    time.sleep(0.08)
    print("视觉模型预热中...", flush=True)
    for _ in range(2):
        got.get_single_track_total_info()
        time.sleep(0.06)
    print("巡线视觉状态复位完成", flush=True)
    return True


def _reset_robot_camera(got):
    """
    彻底复位机器人摄像头状态，切回内置AI模式
    用于程序启动前和退出后，保证语音识别、巡线等功能正常
    """
    global _current_vision_mode
    print("复位机器人摄像头状态...", flush=True)
    try:
        got.close_camera()
        time.sleep(0.3)
        got.open_camera()
        time.sleep(0.3)
        # 加载巡线模型，触发内置AI服务启动
        got.load_models(["line_recognition"])
        _current_vision_mode = "line"
        time.sleep(0.3)
        # 最后关闭，留给后续流程自由初始化
        got.close_camera()
        time.sleep(0.2)
        print("摄像头状态复位完成", flush=True)
    except Exception as e:
        print(f"摄像头复位异常: {e}", flush=True)


def main():
    global got, label_map
    global interp, inp, out, is_uint8, _current_vision_mode

    print("=== UGOT 语音控制物块分拣系统 ===", flush=True)
    try:
        # 1. 连接机器人 + 前置复位摄像头（解决上次残留导致语音失效）
        print("连接机器人...", flush=True)
        got = ugot.UGOT()
        got.initialize('10.233.104.242')
        print("机器人连接成功", flush=True)
        got.set_volume(100)

        # 关键：启动先复位摄像头状态，保证语音识别可用
        _reset_robot_camera(got)

        # 2. 语音识别获取任务（此时摄像头已复位，语音正常）
        print("\n【开始语音识别】", flush=True)
        got.play_audio_tts("开始获取任务", 0, True)
        task_msg = got.start_audio_asr()
        print(f"语音识别内容：{task_msg}", flush=True)

        target_color, target_store = extract_task_info(task_msg)
        if target_color and target_store:
            res_text = f"好的，马上把{target_color}物块放到{target_store}号存储区"
            print(res_text, flush=True)
            got.play_audio_tts(res_text, 0, False)
        else:
            print("语音识别失败，未提取到有效颜色和存储区编号", flush=True)
            got.play_audio_tts("识别失败，请重新下达指令", 0, False)
            return

        # 3. 加载自定义TFLite模型（CPU多线程模式，避开CUDA卡死）
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")
        if not os.path.exists(LABEL_MAP_PATH):
            raise FileNotFoundError(f"找不到标签文件：{LABEL_MAP_PATH}")

        print("加载标签映射...", flush=True)
        label_map = load_label_map(LABEL_MAP_PATH)
        print(f"共 {len(label_map)} 个标签：{dict(sorted(label_map.items()))}", flush=True)

        print("加载 TFLite 模型（CPU多线程模式）...", flush=True)
        interp = tflite.Interpreter(
            model_path=MODEL_PATH,
            num_threads=4
        )
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()
        is_uint8 = inp['dtype'] == np.uint8
        print("模型加载完成", flush=True)

        # 4. 打开摄像头，切换为原始帧模式，用于自定义视觉检测
        print("打开摄像头...", flush=True)
        got.open_camera()
        time.sleep(1)
        _current_vision_mode = "raw"
        print("摄像头就绪", flush=True)

        got.mechanical_clamp_release()
        got.mechanical_single_joint_control(1, 0, 250)
        got.mechanical_single_joint_control(3, -65, 100)
        got.mechanical_single_joint_control(2, -40, 100)
        time.sleep(0.5)

        # 5. 执行完整任务流程
        print(f"\n开始执行：抓取{target_color}物块 → 放置{target_store}号存储区", flush=True)

        # 第一步：搜索并抓取物块
        detected_objs, search_time, flag, flag_diction = get_bbox(target_color)

        # 第二步：返回中转区域
        five_store(
            target_store=target_store,
            num=1,
            search_time=search_time,
            flag=flag,
            task="back_to_center",
            flag_diction=flag_diction
        )

        wttag()
        _current_vision_mode = "apriltag"
        reset_line_vision(got)

        search_label_and_place(target_store)

        # 6. 任务完成
        finish_text = f"任务完成，{target_color}物块已放到{target_store}号存储区"
        print(f"\n{finish_text}", flush=True)
        got.play_audio_tts("任务完成", 0, False)

    except KeyboardInterrupt:
        print("\n手动退出程序", flush=True)
    except Exception as e:
        print(f"程序运行异常: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            got.play_audio_tts("程序出现异常", 0, False)
        except:
            pass
    finally:
        if got:
            got.mecanum_stop()
            try:
                # 关键：退出时彻底复位摄像头，切回内置模式，保证下次语音正常
                _reset_robot_camera(got)
            except Exception as e:
                print(f"退出复位异常: {e}", flush=True)
        print("程序退出，机器人状态已复位", flush=True)


if __name__ == "__main__":
    main()


