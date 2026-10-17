from ugot import ugot
import cv2
import numpy as np
import time


# 颜色范围（HSV）
def get_color_ranges():
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


color_ranges = get_color_ranges()


def detect_color_objects(frame, target_color=None, min_area=500):
    """检测画面中指定颜色的色块，返回列表"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results = []
    check_colors = [target_color] if target_color else color_ranges.keys()

    for color_name in check_colors:
        ranges = color_ranges.get(color_name, [])
        if not ranges:
            continue
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                results.append({
                    'color': color_name,
                    'center': (int(x + w / 2), int(y + h / 2)),
                    'area': area,
                    'rect': (x, y, w, h)
                })
    return results


def calc_bottom_color_ratio(frame, target_color, region_ratio=1 / 4):
    """计算画面底部区域目标颜色占比，用于前进到位判断"""
    height, width = frame.shape[:2]
    bottom = frame[int(height * (1 - region_ratio)):height, :]
    hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
    ranges = color_ranges.get(target_color, [])
    if not ranges:
        return 0
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])


def go_forward_to_color(got, target_color, speed=10):
    """前进直到底部颜色占比达标"""
    while True:
        frame = got.read_camera_data()
        if frame is None:
            time.sleep(0.1)
            continue
        data = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if data is None:
            continue
        ratio = calc_bottom_color_ratio(data, target_color)
        if ratio >= 0.5:
            got.mecanum_stop()
            break
        got.mecanum_move_speed_times(0, speed, 1, 0)
        time.sleep(0.1)


def grasp_block(got, flag):
    """机械臂抓取动作（调试好的固定时序）"""
    # 预张开夹爪
    got.mechanical_single_joint_control(3, -70, 100)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -25, 100)
    time.sleep(0.5)
    got.mechanical_clamp_release()
    time.sleep(0.5)

    # 闭合夹爪抓取
    got.mechanical_clamp_close()
    time.sleep(0.5)
    got.mechanical_single_joint_control(3, -75, 100)
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -35, 100)
    time.sleep(0.5)

    # 二次确认夹紧
    got.mechanical_single_joint_control(3, -70, 100)
    time.sleep(0.5)
    got.mechanical_clamp_release()
    time.sleep(0.5)
    got.mechanical_single_joint_control(2, -50, 100)
    time.sleep(0.5)
    got.mechanical_clamp_close()
    time.sleep(0.5)

    # 抬臂
    got.mechanical_single_joint_control(2, -10, 100)
    time.sleep(0.3)
    print(" 抓取动作完成")


def grab_color_block_full(got, target_color):
    """
    完整色块抓取流程：搜索→对准→前进→抓取→回退
    返回：(是否成功, flag, flag_diction, search_time)
    """
    state = "FINDING_TARGET"
    search_time = 0
    flag = 1  # 1=右侧区域 2=左侧区域
    flag_diction = "right"
    max_search_time = 5
    align_threshold = 35

    print(f" 开始搜索并抓取{target_color}色块")
    try:
        while True:
            frame = got.read_camera_data()
            if frame is None:
                time.sleep(0.1)
                continue
            data = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            if data is None:
                continue

            targets = detect_color_objects(data, target_color)
            height, width = data.shape[:2]
            center_x = width // 2

            if state == "FINDING_TARGET":
                if targets:
                    state = "ADJUSTING_LATERAL"
                    got.mecanum_stop()
                    print("发现目标，开始横向对准")
                else:
                    search_time += 1
                    if flag == 1:
                        got.mecanum_translate_speed_times(90, 15, 1, 0)
                    else:
                        got.mecanum_translate_speed_times(-90, 15, 1, 0)

                    if search_time > max_search_time:
                        if flag == 1:
                            flag = 2
                            flag_diction = "left"
                            search_time = 0
                            got.mecanum_turn_speed_times(3, 130, 45, 2)
                            got.mecanum_translate_speed_times(70, 10, 1, 0)
                        else:
                            print("未找到目标色块")
                            return False, flag, flag_diction, search_time

            elif state == "ADJUSTING_LATERAL":
                if targets:
                    target = max(targets, key=lambda x: x['area'])
                    dev = target['center'][0] - center_x
                    if abs(dev) > align_threshold:
                        if dev > 0:
                            got.mecanum_translate_speed_times(90, 5, 1, 0)
                        else:
                            got.mecanum_translate_speed_times(-90, 5, 1, 0)
                    else:
                        print("对准完成，前进抓取")
                        got.mecanum_stop()
                        go_forward_to_color(got, target_color)
                        grasp_block(got, flag)

                        # 抓取后回退
                        if flag == 1:
                            got.mecanum_move_speed_times(1, 20, 20, 1)
                        if flag == 2 and search_time >= 1:
                            got.mecanum_translate_speed_times(90, 5, 3, 0)
                        return True, flag, flag_diction, search_time
                else:
                    state = "FINDING_TARGET"

            time.sleep(0.05)
    except Exception as e:
        print(f"抓取异常: {e}")
        return False, flag, flag_diction, search_time
    finally:
        got.mecanum_stop()