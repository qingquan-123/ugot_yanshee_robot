# -*- coding: utf-8 -*-
"""
smart_warehouse.py - 智能仓储完整流程（完全独立，不依赖 ugot 目录下其他模块）
流程：机器复位 → 视觉复位 → 6区搜索抓取 → 抓取后返回 → AprilTag对准 → 6区搜索放置 → 放置后返回
两种模式：
  - 7步模式（需要回到取货区）：执行全部7步
  - 6步模式（不需要回到取货区）：执行前6步，去掉第7步放置后返回
"""

import sys
import os
import time
import math
import re
import cv2
import numpy as np
from ugot import ugot

# ===== TensorFlow 条件导入（ugot 环境没有，tf-od 环境才有）=====
try:
    import tensorflow as tf
    tflite = tf.lite
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[警告] tensorflow 未安装，模型识别功能不可用")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(_PROJECT_ROOT, 'models', 'ssd_mobilenet_v2_fpn_320.tflite')
LABEL_MAP_PATH = os.path.join(_PROJECT_ROOT, 'models', 'label_map.pbtxt')

_interpreter = None
_input_details = None
_output_details = None
_is_uint8 = False
_label_map = {}
_model_loaded = False

_MORPH_KERNEL = np.ones((5, 5), np.uint8)

def _get_color_ranges():
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

COLOR_RANGES = _get_color_ranges()

TARGET_TAG_ID = None
FRAME_WIDTH = 640
FRAME_CENTER = FRAME_WIDTH // 2
COARSE_THRESHOLD = 60    # 粗对准阈值，快速拉到中速区
MID_THRESHOLD = 30       # 中速对准阈值，快速靠近
FINE_THRESHOLD = 25      # 精对准/微调偏移阈值
STABLE_FRAMES = 2        # 稳定帧数，避免误判
SEARCH_SPEED = 8         # 搜索阶段自转速度
COARSE_SPEED = 4        # 粗对准阶段旋转速度
MID_SPEED = 8            # 中速对准阶段旋转速度
FINE_MIN_SPEED = 5       # 精调最低速度
FINE_KP = 0.25           # 精调比例系数，越大越快
TAG_SIZE = '10'

# 摄像头标定偏移：机器人物理正对标签时，识别偏移为+16px（而非0）
# 所有偏差判断均减去此值，修正后正对时偏差=0
CALIBRATION_OFFSET = 16

# 搜索阶段首帧稳定确认：连续2帧id相同且偏差差≤此值才认为数据稳定
STABLE_OFFSET_DIFF = 5

STATE_SEARCH = 0
STATE_COARSE_ALIGN = 1
STATE_MID_ALIGN = 2
STATE_FINE_ALIGN = 3
STATE_STABLE_CHECK = 4

LABEL_AREA_THRESHOLD = 13000
ALIGN_THRESHOLD = 18
ALIGN_SPEED = 3
ALIGN_DURATION = 1
MAX_INIT_FRAMES = 5
MAX_CAMERA_CHECKS = 5
LOST_FRAME_THRESHOLD = 10
MAX_SEARCH_TIME_THREE = 3
INVALID_EDGE_RATIO = 0.2
STRICT_MIN_AREA = 2000



def _read_camera_frame(got):
    """读取摄像头帧"""
    frame = got.read_camera_data()
    if frame is None:
        return None
    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return data


def detect_color_objects(frame, min_area=500):
    """检测颜色物体"""
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


def calculate_color_ratio_in_bottom_region(frame, target_color, region_ratio=1 / 4):
    """计算底部区域颜色占比"""
    height, width = frame.shape[:2]
    bottom_start = int(height * (1 - region_ratio))
    bottom_region = frame[bottom_start:height, 0:width]
    hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)
    target_ranges = COLOR_RANGES.get(target_color, [])
    if not target_ranges:
        return 0
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in target_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    total_pixels = mask.shape[0] * mask.shape[1]
    if total_pixels == 0:
        return 0
    return cv2.countNonZero(mask) / total_pixels


def go_forward(got, target_color, speed=5):
    """视觉引导向前靠近目标"""
    while True:
        data = _read_camera_frame(got)
        if data is None:
            print("未获取到摄像头数据，继续尝试...")
            time.sleep(0.5)
            continue
        height, width = data.shape[:2]
        detected_objects = detect_color_objects(data)
        target_objects = [obj for obj in detected_objects if obj['color'] == target_color]
        if not target_objects:
            got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.5)
            continue
        target_obj = max(target_objects, key=lambda x: x['area'])
        x, y, w, h = target_obj['rect']
        bottom_y = y + h
        area = target_obj['area']
        stop_y_threshold = height * 0.9
        stop_area_threshold = height * width * 0.09
        color_ratio = calculate_color_ratio_in_bottom_region(data, target_color, 1 / 4)
        print(f"目标底部Y: {bottom_y:.0f}/{height}, 面积占比: {area / (height * width):.2%}, 底部占比: {color_ratio:.2%}")
        if (bottom_y >= stop_y_threshold and area >= stop_area_threshold) or color_ratio >= 0.39:
            print("到达抓取位置，停止前进")
            got.mecanum_stop()
            break
        else:
            if bottom_y >= height * 0.7 or area >= stop_area_threshold * 0.6:
                got.mecanum_move_speed_times(0, 3, 1, 0)
            else:
                got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.5)


def go_forward_zone3(got, target_color, speed=5):
    """3号区专属：视觉引导向前靠近目标"""
    STOP_Y_RATIO = 0.8
    STOP_AREA_RATIO = 0.085
    BOTTOM_COLOR_RATIO = 0.2
    SLOW_TRIGGER_Y_RATIO = 0.7
    SLOW_TRIGGER_AREA_RATIO = 0.6
    SLOW_SPEED = 3
    BOTTOM_REGION_RATIO = 1 / 4
    while True:
        data = _read_camera_frame(got)
        if data is None:
            print("【3区前进】未获取到摄像头数据，继续尝试...")
            time.sleep(0.5)
            continue
        height, width = data.shape[:2]
        detected_objects = detect_color_objects(data)
        target_objects = [obj for obj in detected_objects if obj['color'] == target_color]
        if not target_objects:
            print("【3区前进】未检测到目标，低速前进维持")
            got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.5)
            continue
        target_obj = max(target_objects, key=lambda x: x['area'])
        x, y, w, h = target_obj['rect']
        bottom_y = y + h
        area = target_obj['area']
        stop_y_threshold = height * STOP_Y_RATIO
        stop_area_threshold = height * width * STOP_AREA_RATIO
        color_ratio = calculate_color_ratio_in_bottom_region(data, target_color, BOTTOM_REGION_RATIO)
        print(f"【3区前进】底部Y: {bottom_y:.0f}/{height} | 面积占比: {area / (height * width):.2%} | 底部占比: {color_ratio:.2%}")
        if (bottom_y >= stop_y_threshold and area >= stop_area_threshold) or color_ratio >= BOTTOM_COLOR_RATIO:
            print("【3区前进】到达抓取位置，停止前进")
            got.mecanum_stop()
            break
        else:
            if bottom_y >= height * SLOW_TRIGGER_Y_RATIO or area >= stop_area_threshold * SLOW_TRIGGER_AREA_RATIO:
                got.mecanum_move_speed_times(0, SLOW_SPEED, 1, 0)
            else:
                got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.5)


def grasp(got, flag):
    """抓取动作"""
    print(f"执行抓取动作，当前区域 flag={flag}")
    got.mechanical_single_joint_control(3, -45, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -25, 150)
    time.sleep(0.5)
    got.mechanical_clamp_release()
    time.sleep(1.0)
    got.mecanum_move_speed_times(1, 20, 9, 1)
    time.sleep(0.5)
    got.mechanical_single_joint_control(3, 15, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -80, 150)
    time.sleep(0.5)
    got.mecanum_turn_speed_times(2, 11, 2, 2)
    time.sleep(0.5)
    got.mecanum_move_speed_times(0, 8, 8, 1)
    time.sleep(0.5)
    got.mecanum_move_speed_times(0, 5, 2, 1)
    time.sleep(0.5)
    got.mechanical_clamp_close()
    time.sleep(1)
    got.mecanum_turn_speed_times(3, 11, 3, 2)
    time.sleep(0.5)
    got.mecanum_move_speed_times(1, 10, 2, 1)
    time.sleep(0.5)
    got.mechanical_single_joint_control(3, 45, 150)
    time.sleep(0.5)
    got.mecanum_move_speed_times(1, 10, 10, 1)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 0, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 40, 150)
    time.sleep(0.5)


    print("抓取完成")


def grasp3(got, flag):
    """3号区专属抓取动作"""
    print(f"执行3号区专属抓取动作，当前区域 flag={flag}")
    got.mechanical_single_joint_control(3, -70, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -25, 150)
    time.sleep(0.5)
    got.mechanical_clamp_release()
    time.sleep(1.0)
    got.mecanum_move_speed_times(1, 20, 11, 1)
    got.mecanum_move_speed_times(1, 20, 1, 1)
    time.sleep(0.5)
    got.mechanical_single_joint_control(3, 15, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -80, 150)
    time.sleep(0.5)
    got.mecanum_move_speed_times(0, 10, 10, 1)
    time.sleep(0.5)
    got.mecanum_move_speed_times(0, 5, 7, 1)
    time.sleep(0.5)
    got.mecanum_turn_speed_times(2, 11, 3, 2)
    time.sleep(0.5)
    got.mecanum_move_speed_times(0, 5, 2, 1)
    time.sleep(0.5)
    got.mechanical_clamp_close()
    time.sleep(1)
    got.mecanum_turn_speed_times(3, 11, 6, 2)
    time.sleep(0.5)
    got.mecanum_move_speed_times(1, 10, 2, 1)
    time.sleep(0.5)
    got.mechanical_single_joint_control(3, 45, 150)
    time.sleep(0.5)
    got.mecanum_move_speed_times(1, 10, 10, 1)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 0, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 40, 150)
    time.sleep(0.5)




def reset_line_vision(got):
    """视觉复位（关摄像头→重开→加载巡线模型→预热）"""
    print("=" * 50)
    print("【强制复位巡线视觉模式】开始")
    try:
        got.close_camera()
        time.sleep(0.3)
    except Exception as e:
        print(f"关闭摄像头提示（非致命）: {e}")
    got.open_camera()
    time.sleep(0.3)
    print("强制加载巡线模型，切换AI识别模式...")
    got.load_models(["line_recognition"])
    time.sleep(0.3)
    got.set_track_recognition_line(0)
    time.sleep(0.1)
    print("预热并校验巡线数据...")
    stable_count = 0
    success = False
    for i in range(3):
        try:
            line_info = got.get_single_track_total_info()
            offset, line_type, x, y = line_info
            print(f"  第{i + 1}帧: offset={offset}, line_type={line_type}")
            if line_type in (1, 2, 3, 4):
                stable_count += 1
                if stable_count >= 3:
                    success = True
                    print(f" 巡线模式复位成功，连续读到有效线路")
                    break
            else:
                stable_count = 0
        except Exception as e:
            print(f"  读取巡线数据异常: {e}")
            stable_count = 0
        time.sleep(0.05)
    if not success:
        print("警告：复位后未读到稳定线路，但已切换到巡线模式")
    print("=" * 50)
    return success


def get_bbox(got, target_color):
    """
    6区分段搜索抓取目标色块（6→5→4→3→2→1）
    返回：(detected_objects, search_time, flag, flag_diction, current_zone_id)
    """
    got.mechanical_clamp_release()
    time.sleep(0.2)
    state = "INITIALIZING"
    search_time = 0
    flag_diction = None
    init_frames = 0
    camera_check_count = 0
    flag = 6
    detected_objects = []
    current_zone_id = None
    lost_frame_count = 0
    print("=================================== 开始6区搜索抓取 ===================================")
    try:
        while True:
            data = _read_camera_frame(got)
            if data is None:
                camera_check_count += 1
                print(f"未获取到摄像头数据，重试 {camera_check_count}/{MAX_CAMERA_CHECKS}")
                if camera_check_count >= MAX_CAMERA_CHECKS:
                    print("摄像头数据获取失败，退出颜色检测")
                    break
                time.sleep(0.5)
                continue
            camera_check_count = 0
            detected_objects = detect_color_objects(data)
            target_objects = [obj for obj in detected_objects if obj['color'] == target_color]
            height, width = data.shape[:2]
            center_x = width // 2
            if flag not in (1, 6):
                valid_left = width * INVALID_EDGE_RATIO
                valid_right = width * (1 - INVALID_EDGE_RATIO)
                strict_area = 500 if flag == 3 else STRICT_MIN_AREA
                target_objects = [
                    obj for obj in target_objects
                    if valid_left <= obj['center'][0] <= valid_right
                    and obj['area'] >= strict_area
                ]
            if state == "INITIALIZING":
                init_frames += 1
                print(f"摄像头初始化中... ({init_frames}/{MAX_INIT_FRAMES})")
                if init_frames >= MAX_INIT_FRAMES:
                    print("摄像头初始化完成，开始搜索 6 号区目标")
                    state = "FINDING_TARGET"
            elif state == "FINDING_TARGET":
                if flag == 6:
                    if target_objects:
                        current_zone_id = 6
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"6号区检测到目标，偏差: {deviation_x}")
                        got.mecanum_translate_speed_times(90, 10, 15, 1)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("6号区无目标，向右平移搜索5号区")
                        got.mecanum_translate_speed_times(90, 8, 42, 1)
                        time.sleep(0.5)
                        flag = 5
                elif flag == 5:
                    if target_objects:
                        current_zone_id = 5
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"5号区检测到有效目标，偏差: {deviation_x}")
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("5号区无有效目标，向右平移搜索4号区")
                        got.mecanum_translate_speed_times(90, 10, 25, 1)
                        time.sleep(0.5  )
                        flag = 4
                elif flag == 4:
                    if target_objects:
                        current_zone_id = 4
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"4号区检测到有效目标，偏差: {deviation_x}")
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("4号区无有效目标，转向搜索3号区")
                        time.sleep(0.5)
                        got.mecanum_turn_speed_times(3, 90, 34, 2)
                        time.sleep(1)
                        flag = 3
                        search_time = 0
                elif flag == 3:
                    if target_objects:
                        current_zone_id = 3
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"3号区检测到有效目标，偏差: {deviation_x}")
                        got.mecanum_move_speed_times(0, 30, 12, 1)
                        got.mecanum_translate_speed_times(90, 15, 3, 1)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        search_time += 1
                        print(f"3区寻找目标... (搜索次数: {search_time})")
                        got.mecanum_stop()
                        if search_time > MAX_SEARCH_TIME_THREE:
                            print("3区域没有有效目标色块，切换至2区搜索")
                            search_time = 0
                            flag = 2
                            got.mecanum_turn_speed_times(3, 90, 51, 2)
                            time.sleep(0.05)
                            print("旋转后检测目标")
                            for step in range(1):
                                data = _read_camera_frame(got)
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
                                    print(f"旋转过程中检测到有效目标！")
                                    current_zone_id = 2
                                    target_obj = max(target_objects, key=lambda x: x['area'])
                                    target_center_x = target_obj['center'][0]
                                    deviation_x = target_center_x - center_x
                                    state = "ADJUSTING_LATERAL"
                                    got.mecanum_stop()
                                    lost_frame_count = 0
                                    break
                elif flag == 2:
                    if target_objects:
                        current_zone_id = 2
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"2号区检测到有效目标，偏差: {deviation_x}")
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("2号区无有效目标，向右平移搜索1号区")
                        search_time += 1
                        got.mecanum_translate_speed_times(90, 15, 19, 1)
                        time.sleep(0.5)
                        flag = 1
                elif flag == 1:
                    if target_objects:
                        current_zone_id = 1
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"1号区检测到有效目标，偏差: {deviation_x}")
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("1号区当前位置无目标，启动阶梯式平移搜索")
                        found_target = False
                        search_step_speed = 5
                        search_step_duration = 2
                        for step in range(3):
                            print(f"1号区右移搜索 第{step+1}/3次")
                            got.mecanum_translate_speed_times(90, search_step_speed, search_step_duration, 0)
                            time.sleep(0.15)
                            data = _read_camera_frame(got)
                            if data is None:
                                continue
                            new_detected = detect_color_objects(data)
                            new_targets = [obj for obj in new_detected if obj['color'] == target_color]
                            if new_targets:
                                current_zone_id = 1
                                target_obj = max(new_targets, key=lambda x: x['area'])
                                deviation_x = target_obj['center'][0] - center_x
                                print(f"右移第{step+1}次检测到目标，偏差: {deviation_x}")
                                state = "ADJUSTING_LATERAL"
                                got.mecanum_stop()
                                lost_frame_count = 0
                                found_target = True
                                break
                        if found_target:
                            continue
                        print("右移3次未找到目标，切换向左平移搜索，共5次")
                        for step in range(5):
                            print(f"1号区左移搜索 第{step+1}/5次")
                            got.mecanum_translate_speed_times(-90, search_step_speed, search_step_duration, 0)
                            time.sleep(0.15)
                            data = _read_camera_frame(got)
                            if data is None:
                                continue
                            new_detected = detect_color_objects(data)
                            new_targets = [obj for obj in new_detected if obj['color'] == target_color]
                            if new_targets:
                                current_zone_id = 1
                                target_obj = max(new_targets, key=lambda x: x['area'])
                                deviation_x = target_obj['center'][0] - center_x
                                print(f"左移第{step+1}次检测到目标，偏差: {deviation_x}")
                                state = "ADJUSTING_LATERAL"
                                got.mecanum_stop()
                                lost_frame_count = 0
                                found_target = True
                                break
                        if found_target:
                            continue
                        print("1号区左右平移搜索完毕，未找到目标，停止搜索")
                        got.mecanum_stop()
                        state = "COMPLETED"
                        break
            elif state == "ADJUSTING_LATERAL":
                if not target_objects:
                    lost_frame_count += 1
                    print(f"当前帧未检测到目标，连续丢帧：{lost_frame_count}/{LOST_FRAME_THRESHOLD}")
                    if lost_frame_count >= LOST_FRAME_THRESHOLD:
                        print("连续多帧未检测到，返回搜索态")
                        state = "FINDING_TARGET"
                        lost_frame_count = 0
                    time.sleep(0.05)
                    continue
                lost_frame_count = 0
                target_obj = max(target_objects, key=lambda x: x['area'])
                target_center_x = target_obj['center'][0]
                target_center_y = target_obj['center'][1]
                deviation_x = target_center_x - center_x
                print(f"调整中 - 目标位置: ({target_center_x}, {target_center_y}), 中心: {center_x}, 偏差: {deviation_x}")
                if abs(deviation_x) > ALIGN_THRESHOLD:
                    if deviation_x > ALIGN_THRESHOLD:
                        print("向右横移一点")
                        got.mecanum_translate_speed_times(90, ALIGN_SPEED, ALIGN_DURATION, 0)
                    elif deviation_x < -ALIGN_THRESHOLD:
                        print("向左横移一点")
                        got.mecanum_translate_speed_times(-90, ALIGN_SPEED, ALIGN_DURATION, 0)
                else:
                    print("左右位置已调整完成！目标已对齐")
                    flag_diction = "right" if deviation_x > 0 else "left"
                    got.mecanum_stop()
                    got.mechanical_single_joint_control(3, -70, 150)
                    got.mechanical_single_joint_control(2, -25, 150)
                    time.sleep(0.05)
                    got.mechanical_clamp_release()
                    time.sleep(0.05)
                    if current_zone_id == 3:
                        go_forward_zone3(got, target_color, speed=5)
                    else:
                        go_forward(got, target_color, speed=5)
                    time.sleep(0.5)
                    if current_zone_id == 3:
                        grasp3(got, current_zone_id)
                    else:
                        grasp(got, current_zone_id)
                    print("抓取完成")
                    state = "COMPLETED"
                    break
            print(f"状态: {state}, 当前区: flag={flag}, 检测到目标数: {len(target_objects)}")
            time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print('-----KeyboardInterrupt')
    except Exception as e:
        print(f"颜色检测过程中发生错误: {e}")
    finally:
        got.mecanum_stop()
        try:
            got.close_camera()
            time.sleep(0.2)
        except Exception:
            pass
        print("颜色检测任务结束，摄像头已释放")
    return detected_objects, search_time, current_zone_id, flag_diction, current_zone_id



def parse_tag_data(raw_data):
    """解析 AprilTag 原始返回数据"""
    if not isinstance(raw_data, list) or len(raw_data) < 13:
        return None
    tag_id = raw_data[1]
    if tag_id == -1:
        return None
    dist_idx = {'5': 7, '7': 8, '10': 9}[TAG_SIZE]
    return {
        "id": int(tag_id),
        "center_x": raw_data[2],
        "center_y": raw_data[3],
        "area": raw_data[6],
        "distance": raw_data[dist_idx] * 100,
        "yaw_deg": math.degrees(raw_data[12])
    }


def align_apriltag(got):
    current_state = STATE_SEARCH
    stable_count = 0
    pending_tag = None
    ROTATE_DONE_THRESHOLD = 25
    TRANSLATE_SPEED = 4
    TRANSLATE_DURATION = 5
    FINE_ROTATE_SPEED = 4
    FINE_TRANSLATE_SPEED = 3
    FINE_TRANSLATE_DURATION = 5
    translate_count = 0
    try:
        while True:
            raw_data = got.get_qrcode_apriltag_total_info()
            tag = parse_tag_data(raw_data)
            if current_state == STATE_SEARCH:
                if tag is None:
                    print("[搜索] 未识别到标签，自转搜索中...")
                    got.mecanum_move_xyz(0, 0, SEARCH_SPEED)
                    time.sleep(0.5)
                    pending_tag = None
                    continue
                if TARGET_TAG_ID is not None and tag["id"] != TARGET_TAG_ID:
                    print(f"[搜索] 识别到ID={tag['id']}，非目标，继续搜索")
                    got.mecanum_move_xyz(0, 0, SEARCH_SPEED)
                    time.sleep(0.5)
                    pending_tag = None
                    continue
                if pending_tag is None:
                    pending_tag = tag
                    offset1 = tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                    print(f"[搜索] 第1帧 ID={tag['id']} 偏差{offset1:+.0f}px，等待第2帧确认...")
                    got.mecanum_move_xyz(0, 0, 0)
                    time.sleep(0.3)
                    continue
                offset1 = pending_tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                offset2 = tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                id_same = (tag["id"] == pending_tag["id"])
                offset_close = (abs(offset2 - offset1) <= STABLE_OFFSET_DIFF)
                if id_same and offset_close:
                    print(f"[搜索] 连续2帧稳定！ID={tag['id']} 偏差{offset2:+.0f}px")
                    if abs(offset2) <= FINE_THRESHOLD:
                        print(f"[搜索] 偏差{offset2:+.0f}px <= {FINE_THRESHOLD}px，直接进入稳定确认")
                        got.mecanum_move_xyz(0, 0, 0)
                        current_state = STATE_STABLE_CHECK
                        stable_count = 0
                    else:
                        print(f"[搜索] 偏差{offset2:+.0f}px，进入旋转对准方向")
                        current_state = STATE_COARSE_ALIGN
                        stable_count = 0
                    pending_tag = None
                else:
                    reason = "id不同" if not id_same else f"偏差差{abs(offset2-offset1):.0f}px超阈值{STABLE_OFFSET_DIFF}px"
                    print(f"[搜索] 数据不稳定（{reason}），重新确认")
                    pending_tag = tag
                    got.mecanum_move_xyz(0, 0, 0)
                    time.sleep(0.3)
            elif current_state == STATE_COARSE_ALIGN:
                if tag is None:
                    print("[旋转对准] 标签丢失，返回搜索")
                    current_state = STATE_SEARCH
                    continue
                offset = tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                print(f"[旋转对准] ID:{tag['id']} 偏移:{offset:+.0f}px 阈值:±{ROTATE_DONE_THRESHOLD}px")
                if abs(offset) <= ROTATE_DONE_THRESHOLD:
                    print("[旋转对准] 方向已摆正，进入横移对准位置")
                    got.mecanum_stop()
                    time.sleep(0.2)
                    current_state = STATE_MID_ALIGN
                    continue
                direction = 1 if offset > 0 else -1
                got.mecanum_move_xyz(0, 0, -direction * COARSE_SPEED)
                time.sleep(0.3)
            elif current_state == STATE_MID_ALIGN:
                if tag is None:
                    print("[横移对准] 标签丢失，返回旋转对准")
                    current_state = STATE_COARSE_ALIGN
                    continue
                offset = tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                print(f"[横移对准] ID:{tag['id']} 偏移:{offset:+.0f}px 阈值:±{FINE_THRESHOLD}px")
                if abs(offset) <= FINE_THRESHOLD:
                    print("[横移对准] 位置已对准，进入微调")
                    got.mecanum_stop()
                    time.sleep(0.2)
                    current_state = STATE_FINE_ALIGN
                    continue
                translate_count += 1
                cur_speed = TRANSLATE_SPEED
                if translate_count > 3:
                    cur_speed = max(2, TRANSLATE_SPEED - 1)
                if translate_count >= 5:
                    print("[横移对准] 已平移5次，进入微调")
                    got.mecanum_stop()
                    time.sleep(0.2)
                    current_state = STATE_FINE_ALIGN
                    continue
                if offset > FINE_THRESHOLD:
                    print(f"向右横移 (第{translate_count}次, 速度{cur_speed})")
                    got.mecanum_translate_speed_times(90, cur_speed, TRANSLATE_DURATION, 1)
                elif offset < -FINE_THRESHOLD:
                    print(f"向左横移 (第{translate_count}次, 速度{cur_speed})")
                    got.mecanum_translate_speed_times(-90, cur_speed, TRANSLATE_DURATION, 1)
                time.sleep(0.3)
            elif current_state == STATE_FINE_ALIGN:
                if tag is None:
                    print("[微调] 标签丢失，返回横移对准")
                    current_state = STATE_MID_ALIGN
                    continue
                offset = tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                print(f"[微调] ID:{tag['id']} 偏移:{offset:+.0f}px 阈值:±{FINE_THRESHOLD}px")
                if abs(offset) <= FINE_THRESHOLD:
                    print("[微调] 偏移达标，进入稳定确认")
                    current_state = STATE_STABLE_CHECK
                    stable_count = 0
                    got.mecanum_move_xyz(0, 0, 0)
                    continue
                direction = 1 if offset > 0 else -1
                got.mecanum_move_xyz(0, 0, -direction * FINE_ROTATE_SPEED)
                time.sleep(0.3)
            elif current_state == STATE_STABLE_CHECK:
                if tag is None:
                    print("[稳定确认] 标签丢失，返回微调")
                    current_state = STATE_FINE_ALIGN
                    continue
                offset = tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                if abs(offset) <= FINE_THRESHOLD:
                    stable_count += 1
                    print(f"[稳定确认] 居中保持中... {stable_count}/{STABLE_FRAMES}")
                    got.mecanum_move_xyz(0, 0, 0)
                    if stable_count >= STABLE_FRAMES:
                        print(f"\n 对准完成！已正对 ID={tag['id']} 的AprilTag")
                        print(f"   距离：{tag['distance']:.1f}cm | 偏航角：{tag['yaw_deg']:.1f}°")
                        got.mecanum_stop()
                        return True, tag
                else:
                    print("[稳定确认] 偏移超出阈值，退回微调")
                    stable_count = 0
                    current_state = STATE_FINE_ALIGN
                time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n手动中断对准")
        got.mecanum_stop()
        return False, None



def load_label_map(pbtxt_path):
    """加载 label_map.pbtxt，返回 {id: name} 字典"""
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
        print(f"[模型] 加载 label_map 成功: {label_map}")
    except Exception as e:
        print(f"[错误] 加载 label_map 失败: {e}")
    return label_map


def init_tflite_model():
    """初始化 TFLite 模型（只加载一次）"""
    global _interpreter, _input_details, _output_details, _is_uint8, _label_map, _model_loaded
    if _model_loaded:
        return True
    if not TF_AVAILABLE:
        print("[错误] tensorflow 不可用，无法加载模型")
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
        print(f"[模型] 标签映射: {dict(sorted(_label_map.items()))}")
        return True
    except Exception as e:
        print(f"[错误] 加载 TFLite 模型失败: {e}")
        return False


def detect_label_objects(frame, min_conf=0.8):
    """
    TFLite 模型检测标签（参考用户提供的标准实现）
    修正：1.先转RGB 2.判断输入类型 3.输出顺序boxes/scores/classes 4.label_id不加1
    """
    if not _model_loaded or _interpreter is None:
        return []
    try:
        # 1. 图像预处理：BGR→RGB→resize到320
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (320, 320))

        # 2. 根据模型输入类型构造张量
        if _is_uint8:
            tensor = np.expand_dims(small, 0)
        else:
            tensor = np.expand_dims(small.astype(np.float32) / 255.0, 0)

        # 3. 推理
        _interpreter.set_tensor(_input_details['index'], tensor)
        _interpreter.invoke()

        # 4. 输出顺序：boxes=out[0], scores=out[1], classes=out[2]
        boxes = _interpreter.get_tensor(_output_details[0]['index'])[0]
        scores = _interpreter.get_tensor(_output_details[1]['index'])[0]
        try:
            classes = _interpreter.get_tensor(_output_details[2]['index'])[0]
        except IndexError:
            classes = None
        if classes is None:
            return []

        # 5. 解析结果
        frame_h, frame_w = frame.shape[:2]
        results = []
        for i in range(len(scores)):
            if scores[i] < min_conf:
                continue
            cls_id = int(classes[i])
            cls_name = _label_map.get(cls_id, f"未知({cls_id})")
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
                'score': float(scores[i])
            })
        return results
    except Exception as e:
        print(f"[错误] 标签检测失败: {e}")
        import traceback
        traceback.print_exc()
        return []



def go_forward_label(got, target_id, speed=5):
    """
    基于标签面积的前进到位（5档逻辑）
    < 9000       : 前进5cm继续找
    9000~11000   : 先前进4cm，再放下
    11000~15000  : 先前进3cm，再放下
    15000~17000  : 先前进2cm，再放下
    17000~21000  : 先前进1cm，再放下
    > 21000      : 后退6cm
    """
    target_name = _label_map.get(target_id, f"ID{target_id}")

    # 3号标签专用逻辑
    if target_id == 3:
        AREA_3_LOW = 8000
        AREA_3_HIGH = 11000
        while True:
            time.sleep(0.3)
            frame = got.read_camera_data()
            if frame is None:
                time.sleep(0.5)
                continue
            nparr = np.frombuffer(frame, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if data is None:
                continue
            label_list = detect_label_objects(data)
            target_list = [obj for obj in label_list if obj["label_id"] == target_id]
            if not target_list:
                print("[放置前进] 丢失{}标签，前进3cm...".format(target_name))
                got.mecanum_move_speed_times(0, speed, 3, 1)
                time.sleep(0.3)
                continue
            target = max(target_list, key=lambda x: x["area"])
            area = target["area"]
            print("[放置前进] {} 标签面积: {:.0f}".format(target_name, area))
            if area < AREA_3_LOW:
                print("  < {}，前进3cm".format(AREA_3_LOW))
                got.mecanum_move_speed_times(0, speed, 3, 1)
                time.sleep(0.3)
            elif area <= AREA_3_HIGH:
                print("  {}~{}，先前进4cm再放下".format(AREA_3_LOW, AREA_3_HIGH))
                got.mecanum_stop()
                got.mecanum_move_speed_times(0, speed, 4, 1)
                time.sleep(0.5)
                got.mecanum_stop()
                break
            else:
                print("  > {}，直接放下".format(AREA_3_HIGH))
                got.mecanum_stop()
                break
        return

    AREA_1 = 9000
    AREA_2 = 11000
    AREA_3 = 15000
    AREA_4 = 17000
    AREA_5 = 21000

    EXTRA_1 = 5
    EXTRA_2 = 4
    EXTRA_3 = 3
    EXTRA_4 = 2
    EXTRA_5 = 1
    BACKWARD_DISTANCE = 6
    SETTLE_DELAY = 0.3

    while True:
        frame = got.read_camera_data()
        if frame is None:
            time.sleep(0.5)
            continue
        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if data is None:
            continue
        label_list = detect_label_objects(data)
        target_list = [obj for obj in label_list if obj["label_id"] == target_id]

        if not target_list:
            print(f"[放置前进] 丢失 {target_name} 标签，前进{EXTRA_1}cm...")
            got.mecanum_move_speed_times(0, speed, EXTRA_1, 1)
            time.sleep(SETTLE_DELAY)
            continue

        target = max(target_list, key=lambda x: x["area"])
        area = target["area"]
        print(f"[放置前进] {target_name} 标签面积: {area:.0f}")

        if area < AREA_1:
            print(f"  < {AREA_1}，前进{EXTRA_1}cm")
            got.mecanum_move_speed_times(0, speed, EXTRA_1, 1)
            time.sleep(SETTLE_DELAY)

        elif area <= AREA_5:
            got.mecanum_stop()
            extra = 0
            if area <= AREA_2:
                extra = EXTRA_2
                print(f"  {AREA_1}~{AREA_2}，先前进{EXTRA_2}cm再放下")
            elif area <= AREA_3:
                extra = EXTRA_3
                print(f"  {AREA_2}~{AREA_3}，先前进{EXTRA_3}cm再放下")
            elif area <= AREA_4:
                if target_id == 5:
                    extra = 0
                    print(f"  {AREA_3}~{AREA_4}，标签五直接放下")
                else:
                    extra = EXTRA_4
                    print(f"  {AREA_3}~{AREA_4}，先前进{EXTRA_4}cm再放下")
            else:
                extra = EXTRA_5
                print(f"  {AREA_4}~{AREA_5}，先前进{EXTRA_5}cm再放下")

            if extra > 0:
                got.mecanum_move_speed_times(0, speed, extra, 1)
                time.sleep(0.5)
                got.mecanum_stop()
            break

        else:
            print(f"  > {AREA_5}，后退{BACKWARD_DISTANCE}cm")
            got.mecanum_stop()
            got.mecanum_move_speed_times(1, speed, BACKWARD_DISTANCE, 1)
            time.sleep(0.5)
            got.mecanum_stop()
            break

def release_block(got, target_id=None):
    """释放物块（机械臂下降→松开夹爪→抬起）
    target_id=6时特殊处理：后退3cm→高位放臂→前进2cm→松爪，避免卡板子"""
    print("[放置] 开始释放物块...")
    
    if target_id == 6:
        print("[放置] 6号标签特殊流程：后退3cm→高位放臂→前进2cm→松爪")
        got.mecanum_stop()
        got.mecanum_move_speed_times(1, 10, 3, 1)
        time.sleep(0.5)
        got.mecanum_stop()
        # 高位放臂（比原来16度高，25度）
        got.mechanical_single_joint_control(3, 25, 150)
        time.sleep(0.5)
        got.mechanical_single_joint_control(2, -80, 150)
        time.sleep(0.5)
        # 前进2cm对准
        got.mecanum_move_speed_times(0, 10, 2, 1)
        time.sleep(0.5)
        got.mecanum_stop()
    else:
        got.mechanical_single_joint_control(3, 16, 150)
        time.sleep(0.5)
        got.mechanical_single_joint_control(2, -80, 150)
        time.sleep(0.5)

    got.mechanical_clamp_release()
    time.sleep(1.0)

    got.mecanum_move_speed_times(1, 10, 1, 1)


    got.mechanical_single_joint_control(3, 45, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 0, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 40, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 60, 150)
    time.sleep(0.5)
    got.mechanical_single_joint_control(3, -60, 200)
    time.sleep(0.5)
    print("[放置] 物块释放完成")
    print("[放置] 物块释放完成")


def search_label_and_place(got, target_id):
    """
    6区分段搜索放置目标区域（模型识别标签，6→5→4→3→2→1）
    返回：(detected_objects, search_time, flag, flag_diction)
    """
    if not init_tflite_model():
        print("[错误] 模型初始化失败，无法执行放置任务")
        return [], 0, None, None
    target_name = _label_map.get(target_id, f"ID{target_id}")

    print(f"\n===== 开始搜索放置 {target_name} 号存储区 =====")
    state = "INITIALIZING"
    search_time = 0
    flag_diction = "right"
    init_frames = 0
    camera_check_count = 0
    flag = 6
    detected_objects = []
    current_zone_id = None
    lost_frame_count = 0
    retry_count = 0
    try:
        while True:
            data = _read_camera_frame(got)
            if data is None:
                camera_check_count += 1
                print(f"未获取到摄像头数据，重试 {camera_check_count}/{MAX_CAMERA_CHECKS}")
                if camera_check_count >= MAX_CAMERA_CHECKS:
                    print("摄像头数据获取失败，退出标签检测")
                    break
                time.sleep(0.5)
                continue
            camera_check_count = 0
            detected_objects = detect_label_objects(data)
            target_objects = [obj for obj in detected_objects if obj['label_id'] == target_id]
            height, width = data.shape[:2]
            center_x = width // 2
            if flag not in (1, 6):
                valid_left = width * INVALID_EDGE_RATIO
                valid_right = width * (1 - INVALID_EDGE_RATIO)
                strict_area = 500 if flag == 3 else STRICT_MIN_AREA
                target_objects = [
                    obj for obj in target_objects
                    if valid_left <= obj['center'][0] <= valid_right
                    and obj['area'] >= strict_area
                ]
            if state == "INITIALIZING":
                init_frames += 1
                print(f"摄像头初始化中... ({init_frames}/{MAX_INIT_FRAMES})")
                if init_frames >= MAX_INIT_FRAMES:
                    print(f"摄像头初始化完成，开始搜索 6 号区标签")
                    state = "FINDING_TARGET"
            elif state == "FINDING_TARGET":
                if flag == target_id and not target_objects:
                    retry_count += 1
                    if retry_count == 1:
                        print(f"[目标区搜索] 未识别到标签，再识别一次")
                        time.sleep(0.1)
                        continue
                    elif retry_count == 2:
                        print(f"[目标区搜索] 仍未识别到，向右平移")
                        got.mecanum_translate_speed_times(90, 4, 5, 1)
                        time.sleep(0.3)
                        continue
                    elif retry_count == 3:
                        print(f"[目标区搜索] 仍未识别到，向左平移2次")
                        got.mecanum_translate_speed_times(-90, 4, 10, 1)
                        time.sleep(0.3)
                        continue
                    elif retry_count == 4:
                        print(f"[目标区搜索] 仍未识别到，再向右平移")
                        got.mecanum_translate_speed_times(90, 4, 5, 1)
                        time.sleep(0.3)
                        continue
                    else:
                        print(f"[目标区搜索] 多次尝试未识别到，切换下一个区")
                        retry_count = 0
                if flag == target_id and target_objects:
                    retry_count = 0
                if flag == 6:
                    if target_objects:
                        current_zone_id = 6
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"6号区检测到目标标签，偏差: {deviation_x}")
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("6号区无目标标签，向右平移搜索5号区")
                        got.mecanum_translate_speed_times(90, 10, 25, 1)
                        time.sleep(0.5)
                        flag = 5
                elif flag == 5:
                    if target_objects:
                        current_zone_id = 5
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"5号区检测到有效目标标签，偏差: {deviation_x}")
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("5号区无有效目标标签，向右平移搜索4号区")
                        got.mecanum_translate_speed_times(90, 10, 26, 1)
                        time.sleep(0.5)
                        flag = 4
                elif flag == 4:
                    if target_objects:
                        current_zone_id = 4
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"4号区检测到有效目标标签，偏差: {deviation_x}")
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("4号区无有效目标标签，转向搜索3号区")
                        time.sleep(0.5)
                        got.mecanum_turn_speed_times(3, 90, 34, 2)
                        got.mecanum_move_speed_times(0, 10, 5, 1)
                        got.mecanum_translate_speed_times(90, 10, 3, 1)
                        time.sleep(1)
                        flag = 3
                        search_time = 0
                elif flag == 3:
                    if target_objects:
                        current_zone_id = 3
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"3号区检测到有效目标标签，偏差: {deviation_x}")
                        got.mecanum_move_speed_times(0, 30, 4, 1)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        search_time += 1
                        print(f"3区寻找目标标签... (搜索次数: {search_time})")
                        got.mecanum_stop()
                        if search_time > MAX_SEARCH_TIME_THREE:
                            print("3区域没有有效目标标签，切换至2区搜索")
                            search_time = 0
                            flag = 2
                            got.mecanum_translate_speed_times(-90, 10, 3, 1)
                            got.mecanum_move_speed_times(1, 10, 8, 1)
                            got.mecanum_turn_speed_times(3, 90, 51, 2)
                            time.sleep(0.05)
                            print("旋转后检测目标标签")
                            for step in range(1):
                                data = _read_camera_frame(got)
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
                                    print(f"旋转过程中检测到有效目标标签！")
                                    current_zone_id = 2
                                    target_obj = max(target_objects, key=lambda x: x['area'])
                                    target_center_x = target_obj['center'][0]
                                    deviation_x = target_center_x - center_x
                                    state = "ADJUSTING_LATERAL"
                                    got.mecanum_stop()
                                    lost_frame_count = 0
                                    break
                elif flag == 2:
                    if target_objects:
                        current_zone_id = 2
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"2号区检测到有效目标标签，偏差: {deviation_x}")
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("2号区无有效目标标签，向右平移搜索1号区")
                        search_time += 1
                        got.mecanum_translate_speed_times(90, 15, 19, 1)
                        time.sleep(0.5)
                        flag = 1
                elif flag == 1:
                    if target_objects:
                        current_zone_id = 1
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"1号区检测到有效目标标签，偏差: {deviation_x}")
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("1号区当前位置无目标标签，启动阶梯式平移搜索")
                        found_target = False
                        search_step_speed = 5
                        search_step_duration = 2
                        for step in range(3):
                            print(f"1号区右移搜索 第{step+1}/3次")
                            got.mecanum_translate_speed_times(90, search_step_speed, search_step_duration, 0)
                            time.sleep(0.15)
                            data = _read_camera_frame(got)
                            if data is None:
                                continue
                            new_detected = detect_label_objects(data)
                            new_targets = [obj for obj in new_detected if obj['label_id'] == target_id]
                            if new_targets:
                                current_zone_id = 1
                                target_obj = max(new_targets, key=lambda x: x['area'])
                                deviation_x = target_obj['center'][0] - center_x
                                print(f"右移第{step+1}次检测到目标标签，偏差: {deviation_x}")
                                state = "ADJUSTING_LATERAL"
                                got.mecanum_stop()
                                lost_frame_count = 0
                                found_target = True
                                break
                        if found_target:
                            continue
                        print("右移3次未找到目标，切换向左平移搜索，共5次")
                        for step in range(5):
                            print(f"1号区左移搜索 第{step+1}/5次")
                            got.mecanum_translate_speed_times(-90, search_step_speed, search_step_duration, 0)
                            time.sleep(0.15)
                            data = _read_camera_frame(got)
                            if data is None:
                                continue
                            new_detected = detect_label_objects(data)
                            new_targets = [obj for obj in new_detected if obj['label_id'] == target_id]
                            if new_targets:
                                current_zone_id = 1
                                target_obj = max(new_targets, key=lambda x: x['area'])
                                deviation_x = target_obj['center'][0] - center_x
                                print(f"左移第{step+1}次检测到目标标签，偏差: {deviation_x}")
                                state = "ADJUSTING_LATERAL"
                                got.mecanum_stop()
                                lost_frame_count = 0
                                found_target = True
                                break
                        if found_target:
                            continue
                        print("1号区左右平移搜索完毕，未找到目标标签，停止搜索")
                        got.mecanum_stop()
                        state = "COMPLETED"
                        break
            elif state == "ADJUSTING_LATERAL":
                if not target_objects:
                    lost_frame_count += 1
                    print(f"当前帧未检测到目标标签，连续丢帧：{lost_frame_count}/{LOST_FRAME_THRESHOLD}")
                    if lost_frame_count >= LOST_FRAME_THRESHOLD:
                        print("连续多帧未检测到，返回搜索态")
                        state = "FINDING_TARGET"
                        lost_frame_count = 0
                    time.sleep(0.05)
                    continue
                lost_frame_count = 0
                target_obj = max(target_objects, key=lambda x: x['area'])
                target_center_x = target_obj['center'][0]
                target_center_y = target_obj['center'][1]
                deviation_x = target_center_x - center_x
                print(f"调整中 - 目标位置: ({target_center_x}, {target_center_y}), 中心: {center_x}, 偏差: {deviation_x}")
                if abs(deviation_x) > ALIGN_THRESHOLD:
                    if deviation_x > ALIGN_THRESHOLD:
                        print("向右横移一点")
                        got.mecanum_translate_speed_times(90, ALIGN_SPEED, ALIGN_DURATION, 0)
                        time.sleep(0.5)
                    elif deviation_x < -ALIGN_THRESHOLD:
                        print("向左横移一点")
                        got.mecanum_translate_speed_times(-90, ALIGN_SPEED, ALIGN_DURATION, 0)
                        time.sleep(0.5)
                else:
                    print("左右位置已调整完成！目标标签已对齐")
                    flag_diction = "right" if deviation_x > 0 else "left"
                    got.mecanum_stop()
                    go_forward_label(got, target_id, speed=5)
                    release_block(got, target_id)
                    print("放置完成")
                    state = "COMPLETED"
                    break
            print(f"状态: {state}, 当前区: flag={flag}, 检测到目标标签数: {len(target_objects)}")
            time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print('-----KeyboardInterrupt')
    except Exception as e:
        print(f"标签检测过程中发生错误: {e}")
    finally:
        got.mecanum_stop()
        try:
            got.close_camera()
            time.sleep(0.2)
        except Exception:
            pass
        print("标签检测任务结束，摄像头已释放")
    return detected_objects, search_time, current_zone_id, flag_diction



def five_store_1(got, flag):
    """
    抓取后返回中转区
    flag=6：左转2° + 右移13cm
    flag=5：左转5° + 右移43cm
    flag=4：左转6° + 右移68cm
    flag=3：前进20cm + 左转40° + 右移65cm
    flag=2：左转5° + 后退57cm + 左转85°
    flag=1：后退56cm + 左转86°
    """
    print(f"\n===== 抓取后返回中转区（flag={flag}）=====")
    if flag == 6:
        print("6号存储区抓取完成，返回中转区1")
        got.mecanum_turn_speed_times(2, 113, 2, 2)
        got.mecanum_translate_speed_times(-90, 20, 13, 1)
    elif flag == 5:
        print("5号存储区抓取完成，返回中转区1")
        got.mecanum_turn_speed_times(2, 113, 5, 2)
        got.mecanum_translate_speed_times(-90, 50, 43, 1)
    elif flag == 4:
        print("4号存储区抓取完成，返回中转区1")
        got.mecanum_turn_speed_times(2, 113, 6, 2)
        got.mecanum_translate_speed_times(-90, 50, 68, 1)
        got.mecanum_move_speed_times(0, 50, 4, 1)
    elif flag == 3:
        print("已抓取3区物块，返回中转区1")
        got.mecanum_move_speed_times(1, 60, 20, 1)
        got.mecanum_turn_speed_times(2, 50, 41, 2)
        time.sleep(0.5)
        got.mecanum_translate_speed_times(-90, 50, 60, 1)
    elif flag == 2:
        print("2号存储区抓取完成，返回中转区1")
        got.mecanum_turn_speed_times(2, 30, 5, 2)
        got.mecanum_move_speed_times(1, 50, 57, 1)
        got.mecanum_turn_speed_times(2, 50, 85, 2)
        got.mecanum_move_speed_times(1, 10, 6, 1)
    elif flag == 1:
        print("1号存储区抓取完成，返回中转区1")
        got.mecanum_move_speed_times(1, 60, 10, 1)
        got.mecanum_turn_speed_times(2, 60, 90, 2)
        got.mecanum_move_speed_times(0, 60, 12, 1)
        got.mecanum_translate_speed_times(-90, 50, 53, 1)
    else:
        print(f"未知 flag={flag}，不执行返回动作")
    time.sleep(0.3)
    print("抓取后返回中转区完成")


def five_store_2(got, flag):
    print(f"\n===== 放置后返回中转区（flag={flag}）=====")
    got.mecanum_move_speed_times(1, 50, 10, 1)
    got.mecanum_turn_speed_times(3, 90, 90, 2)
    time.sleep(0.3)
    if flag == 6:
        print("6号存储区放置完成，返回中转区")
        got.mecanum_move_speed_times(1, 50, 10, 1)
    elif flag == 5:
        print("5号存储区放置完成，返回中转区")
        got.mecanum_move_speed_times(1, 50, 33, 1)
    elif flag == 4:
        print("4号存储区放置完成，返回中转区")
        got.mecanum_move_speed_times(1, 50, 65, 1)
    elif flag == 3:
        print("3号存储区放置完成，返回中转区")
        got.mecanum_turn_speed_times(3, 50, 45, 2)
        time.sleep(0.3)
        got.mecanum_translate_speed_times(90, 50, 66, 1)
    elif flag == 2:
        print("2号存储区放置完成，返回中转区")
        got.mecanum_translate_speed_times(90, 50, 65, 1)
    elif flag == 1:
        print("1号存储区放置完成，返回中转区")
        got.mecanum_translate_speed_times(90, 50, 65, 1)
    else:
        print(f"未知 flag={flag}，不执行额外返回动作")
    time.sleep(0.3)
    print("放置后返回中转区完成")




def arm_reset(got):
    print("\n===== 机器复位 =====")
    got.mechanical_clamp_release()
    time.sleep(1.0)
    got.mechanical_single_joint_control(3, 60, 350)
    time.sleep(1.0)
    got.mechanical_single_joint_control(2, 60, 350)
    time.sleep(1.0)
    got.mechanical_single_joint_control(1, 0, 350)
    time.sleep(1.0)
    got.mechanical_clamp_release()
    time.sleep(0.3)
    print("机器复位完成")


def action_forward(got, distance_cm, speed=15):
    """前进指定距离"""
    print(f"  前进 {distance_cm}cm")
    got.mecanum_move_speed_times(0, speed, distance_cm, 1)
    time.sleep(0.5)


def action_backward(got, distance_cm, speed=15):
    """后退指定距离"""
    print(f"  后退 {distance_cm}cm")
    got.mecanum_move_speed_times(1, speed, distance_cm, 1)
    time.sleep(0.5)


def action_turn(got, angle, speed=60):
    """原地转向，正数左转，负数右转"""
    direction = "左转" if angle > 0 else "右转"
    abs_angle = abs(angle)
    print(f"  {direction} {abs_angle}°")
    if angle > 0:
        got.mecanum_turn_speed_times(2, speed, abs_angle, 2)
    else:
        got.mecanum_turn_speed_times(3, speed, abs_angle, 2)
    time.sleep(0.5)



def action_rotate(got, angle, speed=80):
    """原地旋转（转圈用），正数右转(顺时针)，负数左转(逆时针)，和 znjs 的 rotate:360 对应"""
    direction = "右转" if angle > 0 else "左转"
    abs_angle = abs(angle)
    print(f"  旋转 {abs_angle}° ({direction})")
    if angle > 0:
        got.mecanum_turn_speed_times(3, speed, abs_angle, 2)
    else:
        got.mecanum_turn_speed_times(2, speed, abs_angle, 2)
    time.sleep(0.5)

def align_with_apriltag(got):
    """AprilTag对准（加载模型→五阶段对准→释放模型）"""
    print("\n===== AprilTag 对准 =====")
    got.open_camera()
    time.sleep(0.5)
    got.load_models(['apriltag_qrcode'])
    time.sleep(1)
    success, tag_info = align_apriltag(got)
    try:
        got.release_models(['apriltag_qrcode'])
    except Exception:
        pass
    if success:
        print(f"AprilTag 对准成功：ID={tag_info.get('id')}, 距离={tag_info.get('distance', 0):.1f}cm")
    else:
        print("AprilTag 对准失败")
    return success, tag_info



def warehouse_full_7step(got, target_color, target_store_id):
    """
    7步完整流程（需要回到取货区）
    1. AprilTag对准（先识别tag码）
    2. 机器复位（再复位机械臂）
    3. 视觉复位
    4. 6区搜索抓取
    5. 抓取后返回中转区
    6. 6区搜索放置
    7. 放置后返回中转区
    返回：(success, grab_flag, place_flag)
    """
    print("\n" + "=" * 60)
    print(f"智能仓储7步完整流程：先Tag对准 → 抓取{target_color} → 放置到{target_store_id}号 → 返回")
    print("=" * 60)
    try:
        print("\n===== 第1步：AprilTag 对准 =====")
        align_with_apriltag(got)

        arm_reset(got)

        print("\n===== 第3步：视觉复位 =====")
        reset_line_vision(got)

        print(f"\n===== 第4步：搜索抓取 {target_color} 色块 =====")
        _, _, grab_flag, _, _ = get_bbox(got, target_color)
        if grab_flag is None:
            print("未找到目标色块，任务终止")
            return False, None, None
        print(f"在 {grab_flag} 号区抓取到 {target_color} 色块")

        if grab_flag == target_store_id:
            print(f"抓取区({grab_flag}号)与目标放置区({target_store_id}号)相同，冲突！改放6号")
            target_store_id = 6

        five_store_1(got, grab_flag)

        align_with_apriltag(got)

        got.mecanum_translate_speed_times(90, 10, 14, 1)
        time.sleep(0.5)
        got.mecanum_move_speed_times(0, 10, 5, 1)

        print(f"\n===== 第6步：搜索放置到 {target_store_id} 号存储区 =====")
        _, _, place_flag, _ = search_label_and_place(got, target_store_id)
        if place_flag is None:
            print("未找到目标存储区，任务终止")
            return False, grab_flag, None
        print(f"在 {place_flag} 号区放置完成")

        five_store_2(got, place_flag)

        print("\n" + "=" * 60)
        print(f"7步流程完成！抓取区={grab_flag}, 放置区={place_flag}")
        print("=" * 60)
        return True, grab_flag, place_flag
    except Exception as e:
        print(f"7步流程执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None
    finally:
        got.mecanum_stop()


def warehouse_full_6step(got, target_color, target_store_id):
    """
    6步流程（不需要回到取货区，去掉第7步放置后返回）
    1. AprilTag对准（先识别tag码）
    2. 机器复位（再复位机械臂）
    3. 视觉复位
    4. 6区搜索抓取
    5. 抓取后返回中转区
    6. 6区搜索放置
    返回：(success, grab_flag, place_flag)
    """
    print("\n" + "=" * 60)
    print(f"智能仓储6步流程：先Tag对准 → 抓取{target_color} → 放置到{target_store_id}号（不返回）")
    print("=" * 60)
    try:
        # 第1步：先识别tag码
        print("\n===== 第1步：AprilTag 对准 =====")
        align_with_apriltag(got)

        # 第2步：再机器复位
        arm_reset(got)

        # 第3步：视觉复位
        print("\n===== 第3步：视觉复位 =====")
        reset_line_vision(got)

        # 第4步：搜索抓取
        print(f"\n===== 第4步：搜索抓取 {target_color} 色块 =====")
        _, _, grab_flag, _, _ = get_bbox(got, target_color)
        if grab_flag is None:
            print("未找到目标色块，任务终止")
            return False, None, None
        print(f"在 {grab_flag} 号区抓取到 {target_color} 色块")

        # 冲突判断：抓取区与目标放置区相同，改放6号
        if grab_flag == target_store_id:
            print(f" 抓取区({grab_flag}号)与目标放置区({target_store_id}号)相同，冲突！改放6号")
            target_store_id = 6

        # 第5步：抓取后返回
        five_store_1(got, grab_flag)

        # 第5.5步：返回中转区后再次AprilTag对准，修正返回来的车身朝向
        align_with_apriltag(got)

        # 第5.6步：对准后向右平移一点点
        got.mecanum_translate_speed_times(90, 10, 15, 1)

        # 第6步：搜索放置
        print(f"\n===== 第6步：搜索放置到 {target_store_id} 号存储区 =====")
        _, _, place_flag, _ = search_label_and_place(got, target_store_id)
        if place_flag is None:
            print("未找到目标存储区，任务终止")
            return False, grab_flag, None
        print(f"在 {place_flag} 号区放置完成")

        print("\n" + "=" * 60)
        print(f"6步流程完成！抓取区={grab_flag}, 放置区={place_flag}")
        print("=" * 60)
        return True, grab_flag, place_flag
    except Exception as e:
        print(f"6步流程执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None
    finally:
        got.mecanum_stop()



if __name__ == "__main__":
    print(f"=== smart_warehouse.py 启动 ===")

    # 智能仓储完整流程：python smart_warehouse.py smart_warehouse <前置转圈> <前进cm> <颜色> <放置号> <后退cm> <转向角> <是否返回1/0>
    if len(sys.argv) >= 2 and sys.argv[1] == "smart_warehouse":
        pre_rotate = int(sys.argv[2])       # 前置转圈角度（正数右转，负数左转，0不转）
        forward_cm = int(sys.argv[3])      # 前进距离cm
        target_color = sys.argv[4]          # 颜色
        target_store_id = int(sys.argv[5])  # 放置位置号
        backward_cm = int(sys.argv[6])      # 后退距离cm
        turn_angle = int(sys.argv[7])       # 完成后转向角度（正=左转，负=右转）
        need_return = int(sys.argv[8]) if len(sys.argv) > 8 else 1  # 1=7步(回到取货区), 0=6步(不返回)

        print(f"模式: 智能仓储完整流程")
        print(f"  前置转圈: {pre_rotate}°")
        print(f"  前进: {forward_cm}cm")
        print(f"  抓取颜色: {target_color}")
        print(f"  放置位置: {target_store_id}号")
        print(f"  后退: {backward_cm}cm")
        print(f"  转向: {turn_angle}°")
        print(f"  流程: {'7步(回到取货区)' if need_return else '6步(不返回)'}")

        got = ugot.UGOT()
        got.initialize('10.233.104.242')
        print("机器人连接成功")

        # 0. 前置转圈
        if pre_rotate != 0:
            action_rotate(got, pre_rotate)

        # 1. 前进 Zcm
        action_forward(got, forward_cm)

        # 2. 抓取并放置（7步或6步）
        if need_return:
            success, grab_flag, place_flag = warehouse_full_7step(got, target_color, target_store_id)
        else:
            success, grab_flag, place_flag = warehouse_full_6step(got, target_color, target_store_id)
        if not success:
            print("仓储流程失败")
            sys.exit(1)

        # 3. 后退 Lcm
        action_backward(got, backward_cm)

        # 4. 转向
        if turn_angle != 0:
            action_turn(got, turn_angle)

        print(f"\n智能仓储任务完成: grab_flag={grab_flag}, place_flag={place_flag}")
        sys.exit(0)

    # 原模式：python smart_warehouse.py <target_color> <target_store_id> <need_return>
    else:
        if len(sys.argv) < 4:
            print("用法:")
            print("  智能仓储: python smart_warehouse.py smart_warehouse <前进cm> <颜色> <放置号> <后退cm> <转向角>")
            print("  基础模式: python smart_warehouse.py <target_color> <target_store_id> <need_return>")
            print("示例:")
            print("  python smart_warehouse.py smart_warehouse 0 绿色 6 30 90")
            print("  python smart_warehouse.py 绿色 5 1")
            sys.exit(1)
        target_color = sys.argv[1]
        target_store_id = int(sys.argv[2])
        need_return = int(sys.argv[3])
        print(f"模式: 基础仓储流程")
        print(f"目标颜色: {target_color}")
        print(f"目标存储区: {target_store_id}号")
        print(f"是否需要返回: {'是(7步)' if need_return else '否(6步)'}")
        got = ugot.UGOT()
        got.initialize('10.233.104.242')
        print("机器人连接成功")
        if need_return:
            success, grab_flag, place_flag = warehouse_full_7step(got, target_color, target_store_id)
        else:
            success, grab_flag, place_flag = warehouse_full_6step(got, target_color, target_store_id)
        print(f"\n执行结果: success={success}, grab_flag={grab_flag}, place_flag={place_flag}")
        sys.exit(0 if success else 1)
