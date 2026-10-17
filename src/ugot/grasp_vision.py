# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
from pid_controller import quick_align_after_turn

_MORPH_KERNEL = np.ones((5, 5), np.uint8)

max_rotate_speed = 35
base_forward_speed = 28
max_forward_speed = 33
cross_forward_speed = 35
lost_line_low_speed = 8
lost_line_rotate_speed = 12
_line_model_loaded = False


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


def _read_camera_frame(got):
    """统一封装摄像头帧读取"""
    frame = got.read_camera_data()
    if frame is None:
        return None
    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return data


def detect_color_objects(frame, min_area=500):
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

        print(
            f"目标底部Y: {bottom_y:.0f}/{height}, 面积占比: {area / (height * width):.2%}, 底部占比: {color_ratio:.2%}")

        if (bottom_y >= stop_y_threshold and area >= stop_area_threshold) or color_ratio >= 0.39:
            print("到达抓取位置，停止前进")
            got.mecanum_stop()
            break
        else:
            if bottom_y >= height * 0.7 or area >= stop_area_threshold * 0.6:
                print("1-1-1-1-1-1-1-1-1-1-1--1-1--1--1-1--1--1--1-1--1-1-1--1-1-")
                got.mecanum_move_speed_times(0, 3, 1, 0)
            else:
                print("2-2-2-2-2-2-2--2--2-2--2--2-2-2-2--2--2-2--2-2--2-2-2-2-2-")
                got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.1)

def go_forward_zone3(got, target_color, speed=5):
    """
    3号区专属：视觉引导向前靠近目标
    参数已完全对齐原版 go_forward，所有可调项集中在此处
    """
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
            print("【3区】未获取到摄像头数据，继续尝试...")
            time.sleep(0.1)
            continue

        height, width = data.shape[:2]
        detected_objects = detect_color_objects(data)
        target_objects = [obj for obj in detected_objects if obj['color'] == target_color]

        if not target_objects:
            print("【3区】未检测到目标，低速前进维持")
            got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.1)
            continue

        target_obj = max(target_objects, key=lambda x: x['area'])
        x, y, w, h = target_obj['rect']
        bottom_y = y + h
        area = target_obj['area']

        stop_y_threshold = height * STOP_Y_RATIO
        stop_area_threshold = height * width * STOP_AREA_RATIO
        color_ratio = calculate_color_ratio_in_bottom_region(data, target_color, BOTTOM_REGION_RATIO)

        print(
            f"【3区】底部Y: {bottom_y:.0f}/{height} | "
            f"面积占比: {area / (height * width):.2%} | "
            f"底部占比: {color_ratio:.2%}")

        if (bottom_y >= stop_y_threshold and area >= stop_area_threshold) or color_ratio >= BOTTOM_COLOR_RATIO:
            print("【3区】到达抓取位置，停止前进")
            got.mecanum_stop()
            break
        else:
            if bottom_y >= height * SLOW_TRIGGER_Y_RATIO or area >= stop_area_threshold * SLOW_TRIGGER_AREA_RATIO:
                got.mecanum_move_speed_times(0, SLOW_SPEED, 1, 0)
            else:
                got.mecanum_move_speed_times(0, speed, 1, 0)
            time.sleep(0.1)


def grasp(got, flag):
    """抓取动作"""
    print(f"执行抓取动作，当前区域 flag={flag}")
    got.mechanical_single_joint_control(3, -45, 200)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -25, 200)
    time.sleep(0.5)
    got.mechanical_clamp_release()
    time.sleep(0.5)

    got.mecanum_move_speed_times(1, 20, 9, 1)

    got.mechanical_single_joint_control(3, 15, 200)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -80, 200)
    time.sleep(0.5)

    got.mecanum_move_speed_times(0, 8, 7, 1)
    time.sleep(0.5)

    got.mecanum_turn_speed_times(2, 11, 3, 2)
    got.mecanum_move_speed_times(0, 5, 5, 1)
    time.sleep(0.1)

    got.mechanical_clamp_close()
    time.sleep(1)
    got.mecanum_turn_speed_times(3, 11, 5, 2)

    got.mecanum_move_speed_times(1, 10, 2, 1)
    time.sleep(0.5)
    got.mechanical_single_joint_control(3, 45, 200)
    time.sleep(0.5)
    got.mecanum_move_speed_times(1, 10, 10, 1)
    time.sleep(0.5)

    got.mechanical_single_joint_control(2, 0, 200)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 40, 200)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 90, 200)
    time.sleep(0.5)
    print("抓取完成")






def grasp3(got, flag):
    """3号区专属抓取动作"""
    print(f"执行3号区专属抓取动作，当前区域 flag={flag}")
    got.mechanical_single_joint_control(3, -70, 200)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -25, 200)
    time.sleep(0.5)
    got.mechanical_clamp_release()
    time.sleep(0.5)

    got.mecanum_move_speed_times(1, 20, 11, 1)
    got.mecanum_move_speed_times(1, 20, 1, 1)

    got.mechanical_single_joint_control(3, 15, 200)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -80, 200)
    time.sleep(0.5)

    got.mecanum_move_speed_times(0, 10, 10, 1)
    time.sleep(0.1)
    got.mecanum_move_speed_times(0, 5, 7, 1)
    time.sleep(0.1)

    got.mecanum_turn_speed_times(2, 11, 3, 2)
    got.mecanum_move_speed_times(0, 5, 3, 1)
    time.sleep(0.1)

    got.mechanical_clamp_close()
    time.sleep(1)
    got.mecanum_turn_speed_times(3, 11, 5, 2)

    got.mecanum_move_speed_times(1, 10, 2, 1)
    time.sleep(0.5)
    got.mechanical_single_joint_control(3, 45, 200)
    time.sleep(0.5)
    got.mecanum_move_speed_times(1, 10, 10, 1)
    time.sleep(0.5)

    got.mechanical_single_joint_control(2, 0, 200)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 40, 200)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, 90, 200)
    time.sleep(0.5)
    print("抓取完成")

def reset_line_vision(got):
    print("=" * 50)
    print("【强制复位巡线视觉模式】开始")

    got.mecanum_stop()
    time.sleep(0.3)

    try:
        got.close_camera()
        time.sleep(0.3)
    except Exception as e:
        print(f"关闭摄像头提示: {e}")

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
    for i in range(5):
        try:
            line_info = got.get_single_track_total_info()
            offset, line_type, x, y = line_info
            print(f"  第{i + 1}帧: offset={offset}, line_type={line_type}")

            if line_type in (1, 2, 3, 4):
                stable_count += 1
                if stable_count >= 3:
                    success = True
                    print(f"巡线复位成功，读到连续有效线路")
                    break
            else:
                stable_count = 0
        except Exception as e:
            print(f"读取巡线数据异常: {e}")
            stable_count = 0
        time.sleep(0.05)

    if not success:
        print("复位后未读到稳定线路，开始巡线")

    print("=" * 50)
    return success



def _get_rotate_speed(u, pid_ctrl, max_speed=max_rotate_speed):
    line_info = u.get_single_track_total_info()
    offset, line_type, x, y = line_info
    if line_type == 0:
        return 0, line_type, x, y
    rotate_speed = round(pid_ctrl.update(offset))
    return max(min(rotate_speed, max_speed), -max_speed), line_type, x, y


def pid_find_line_store(u, target_store, pid_ctrl):
    """PID巡线"""
    LOST_LINE_THRESHOLD = 8
    RECOVER_FRAMES = 15
    lost_line_count = 0
    recover_stable_count = 0
    last_rotate_speed = 0
    turn_count = 0
    crossroad_confirm = 0
    last_line_type = 0
    start_frames = 0
    START_BUFFER_FRAMES = 15

    finding_line = False
    find_direction = 1
    find_step_count = 0

    if target_store == "A":
        print("前往A号存储区")
    else:
        print("前往B号存储区")

    while True:
        try:
            line_info = u.get_single_track_total_info()
            offset, line_type, x, y = line_info
        except Exception as e:
            print(f"巡线识别出错: {e}，停车重试")
            u.mecanum_stop()
            time.sleep(0.2)
            continue

        if line_type != last_line_type:
            if line_type == 3:
                print("检测到十字路口")
            elif line_type == 4:
                print("检测到T字路口")
            last_line_type = line_type

        if line_type in (1, 2, 4):
            lost_line_count = 0
            crossroad_confirm = 0
            finding_line = False
            find_step_count = 0
            start_frames += 1

            if recover_stable_count < RECOVER_FRAMES:
                recover_stable_count += 1
                speed_limit = 15
            else:
                speed_limit = max_forward_speed

            rotate_speed = round(pid_ctrl.update(offset))
            rotate_speed = max(min(rotate_speed, max_rotate_speed), -max_rotate_speed)
            smooth_rotate = int(last_rotate_speed * 0.4 + rotate_speed * 0.6)
            last_rotate_speed = smooth_rotate

            is_curve = line_type == 2
            if is_curve:
                dynamic_speed = int(base_forward_speed * 0.6 - min(abs(smooth_rotate) / 4, 5))
            else:
                dynamic_speed = int(base_forward_speed - min(abs(smooth_rotate) / 4, 6))

            if start_frames < START_BUFFER_FRAMES:
                ramp_speed = 12 + int((dynamic_speed - 12) * (start_frames / START_BUFFER_FRAMES))
                dynamic_speed = min(dynamic_speed, ramp_speed)

            dynamic_speed = max(min(dynamic_speed, speed_limit), 5)

            u.mecanum_move_xyz(0, dynamic_speed, -smooth_rotate)
            print(f"巡线运行 - PID偏移:{rotate_speed}, 平滑转向:{smooth_rotate}, 当前车速:{dynamic_speed}")

        elif line_type == 3:
            crossroad_confirm += 1
            if crossroad_confirm < 1:
                u.mecanum_move_xyz(0, 8, 0)
                time.sleep(0.05)
                continue

            u.mecanum_stop()
            time.sleep(0.3)
            turn_count += 1
            recover_stable_count = 0
            crossroad_confirm = 0
            start_frames = 0
            print(f"确认第{turn_count}个十字路口，执行转向")

            if target_store == "A":
                if turn_count == 1:

                    print("第1个路口")

                    u.mecanum_move_speed_times(0, 35, 21, 1)
                    u.mecanum_turn_speed_times(3, 80, 90, 2)
                    time.sleep(0.5)
                elif turn_count == 2:
                    print("第2个路口")
                    u.play_audio_tts("", 0, False)
                    u.mecanum_move_speed_times(0, 40, 18, 1)
                    u.mecanum_turn_speed_times(3, 80, 89, 2)
                    time.sleep(0.5)
                elif turn_count == 3:
                    print("第3个路口")
                    u.mecanum_move_speed_times(0, 35, 14, 1)
                    u.mecanum_translate_speed_times(90, 20, 3, 1)
                    time.sleep(0.2)
                    u.mechanical_single_joint_control(2, 80, 500)
                    time.sleep(0.5)
                    u.mechanical_single_joint_control(3, -90, 500)
                    time.sleep(0.2)
                    u.mecanum_move_speed_times(0, 20, 3, 1)
                    time.sleep(0.5)
                    u.mechanical_clamp_release()
                    break
            else:
                if turn_count == 1:
                    print("第1个路口")
                    u.play_audio_tts("", 0, False)
                    u.mecanum_move_speed_times(0, 35, 20, 1)
                    u.mecanum_turn_speed_times(3, 120, 90, 2)
                    u.mecanum_move_speed_times(0, 35, 7, 1)
                    time.sleep(0.5)
                elif turn_count == 2:
                    print("第2个路口")
                    u.mecanum_move_speed_times(0, 40, 51, 1)
                    u.mecanum_turn_speed_times(3, 120, 85, 2)
                    time.sleep(0.5)
                elif turn_count == 3:
                    u.mecanum_move_speed_times(0, 35, 13, 1)
                    u.mecanum_translate_speed_times(90, 20, 2, 1)
                    time.sleep(0.2)
                    u.mechanical_single_joint_control(2, 80, 500)
                    time.sleep(0.5)
                    u.mechanical_single_joint_control(3, -90, 500)
                    time.sleep(0.5)
                    u.mecanum_move_speed_times(0, 20, 3, 1)
                    time.sleep(0.5)
                    u.mechanical_clamp_release()
                    print("已到达B库区")
                    break


            time.sleep(0.4)
            u.mecanum_stop()
            time.sleep(0.1)


        else:
            recover_stable_count = 0
            start_frames = 0
            lost_line_count += 1

            if lost_line_count < LOST_LINE_THRESHOLD:

                u.mecanum_move_xyz(0, 2 , -int(last_rotate_speed * 0.2))
                print(f"临时丢线，低速维持找线，连续丢帧: {lost_line_count}")
            else:

                if not finding_line:
                    finding_line = True
                    find_direction = 1
                    find_step_count = 0
                    print("持续丢线，启动主动旋转找线")

                u.mecanum_move_xyz(0, 0, 10 * find_direction)
                find_step_count += 1

                if find_step_count >= 10:
                    find_direction *= -1
                    find_step_count = 0

        time.sleep(0.1)


def complete_task(u, target_store, num, task):
    print("启动巡线导航至目标库区...")
    u.set_track_recognition_line(0)
    pid_store = u.create_pid_controller()
    pid_store.set_pid(0.12, 0.0015, 0.15)
    pid_find_line_store(u, target_store, pid_store)
    u.mecanum_move_speed_times(1, 7, 13, 1)
    time.sleep(0.2)
    print("任务一完成！")


def five_store(got, target_store, num, search_time, flag, task, flag_diction, zone_id=None):
    print("flag:", flag)
    print("当前区域:", f"{zone_id}号区" if zone_id else "非4/5/6区")

    if flag == 6:
        print("6号存储区抓取完成，返回中转区")
        got.mecanum_turn_speed_times(2, 113, 182, 2)
        got.mecanum_translate_speed_times(90, 20, 13, 1)

    elif flag == 5:
        print("5号存储区抓取完成，返回中转区")
        got.mecanum_turn_speed_times(2, 113, 188, 2)
        got.mecanum_translate_speed_times(90, 50, 43, 1)

    elif flag == 4:
        print("4号存储区抓取完成，返回中转区")
        got.mecanum_turn_speed_times(2, 113, 186, 2)
        got.mecanum_translate_speed_times(90, 50, 68, 1)

    elif flag == 2:
        print("2号存储区抓取完成，返回中转区")
        got.mecanum_turn_speed_times(2, 30, 5, 2)
        got.mecanum_move_speed_times(1, 50, 57, 1)
        got.mecanum_turn_speed_times(3, 50, 85, 2)

    elif flag == 1:
        print("1号存储区抓取完成，返回中转区")
        got.mecanum_move_speed_times(1, 60, 56, 1)
        got.mecanum_turn_speed_times(3, 60, 85, 2)
        got.mecanum_move_speed_times(1, 30, 8, 1)

    elif flag == 3:
        print("已抓取3区物块")
        got.mecanum_move_speed_times(1, 60, 20, 1)
        got.mecanum_turn_speed_times(3, 50, 131, 2)
        time.sleep(0.5)
        got.mecanum_translate_speed_times(90, 50, 60, 1)

    time.sleep(0.3)
    reset_line_vision(got)

    print(f"准备前往{target_store}仓库")
    while True:
        offset, line_type, x, y = got.get_single_track_total_info()
        print(f"线路检测 - 类型: {line_type}, 偏移: {offset}")
        if line_type == 3 or line_type == 1:
            got.mecanum_stop()
            time.sleep(0.15)
            print("==================429=====================")
            got.mecanum_translate_speed_times(90, 6, 1, 0)

            for i in range(1):
                print(f"第{i}次调整")
                quick_align_after_turn(got, enable_forward_slide=False)
            complete_task(got, target_store, num, task)
            break

        elif line_type == 0:
            print("==================439=====================")
            got.mecanum_translate_speed_times(90, 4, 1, 0)
        else:
            print("==================442=====================")
            got.mecanum_move_speed_times(0, 10, 10, 1)
        time.sleep(0.05)


def get_bbox(got, target_color):

    max_search_time_three = 3
    max_init_frames = 5
    max_camera_checks = 5
    LOST_FRAME_THRESHOLD = 30

    ALIGN_THRESHOLD = 18
    ALIGN_SPEED = 3
    ALIGN_DURATION = 1

    INVALID_EDGE_RATIO = 0.2
    STRICT_MIN_AREA = 2000

    state = "INITIALIZING"
    search_time = 0
    flag_diction = None
    init_frames = 0
    camera_check_count = 0
    flag = 6
    detected_objects = []
    current_zone_id = None

    lost_frame_count = 0

    print("===================================471======================================")
    try:
        while True:
            data = _read_camera_frame(got)
            if data is None:
                camera_check_count += 1
                print(f"未获取到摄像头数据，重试 {camera_check_count}/{max_camera_checks}")
                if camera_check_count >= max_camera_checks:
                    print("摄像头数据获取失败，退出颜色检测")
                    break
                time.sleep(0.1)
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
                print(f"摄像头初始化中... ({init_frames}/{max_init_frames})")
                if init_frames >= max_init_frames:
                    print("摄像头初始化完成，开始搜索 6 号区目标")
                    state = "FINDING_TARGET"

            elif state == "FINDING_TARGET":

                if flag == 6:
                    if target_objects:
                        current_zone_id = 6
                        target_obj = max(target_objects, key=lambda x: x['area'])
                        deviation_x = target_obj['center'][0] - center_x
                        print(f"6号区检测到目标，偏差: {deviation_x}")
                        got.mecanum_translate_speed_times(90, 10, 13, 1)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        print("6号区无目标，向右平移搜索5号区")
                        got.mecanum_translate_speed_times(90, 10, 40, 1)
                        time.sleep(0.1)
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
                        got.mecanum_translate_speed_times(90,10, 25, 1)
                        time.sleep(0.1)
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
                        got.mecanum_translate_speed_times(90, 20, 2, 1)
                        got.mecanum_move_speed_times(0, 30, 12, 1)
                        state = "ADJUSTING_LATERAL"
                        got.mecanum_stop()
                        lost_frame_count = 0
                    else:
                        search_time += 1
                        print(f"3区寻找目标... (搜索次数: {search_time})")
                        got.mecanum_stop()

                        if search_time > max_search_time_three:
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
                        time.sleep(0.1)
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

                print(
                    f"调整中 - 目标位置: ({target_center_x}, {target_center_y}), 中心位置: {center_x}, 偏差: {deviation_x}")

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

                    got.mechanical_single_joint_control(3, -70, 100)
                    time.sleep(0.5)
                    got.mechanical_single_joint_control(2, -25, 100)
                    time.sleep(0.5)
                    got.mechanical_clamp_release()
                    time.sleep(0.5)

                    if current_zone_id == 3:
                        go_forward_zone3(got, target_color, speed=5)
                    else:
                        go_forward(got, target_color, speed=5)


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