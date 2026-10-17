from ugot import ugot
import cv2
import numpy as np

got = None  # 延迟初始化，由调用方传入，避免 import 时就连接机器人

# ============距离维持控制参数============
TARGET_AREA = 30000  # 目标像素中间值
AREA_LOW = 25000  # 下限
AREA_HIGH = 35000  # 上限
Kp = 0.0008
MAX_SPEED = 5
MIN_SPEED = 5

# ============对齐稳定判定参数============
STABLE_FRAMES = 1  # 连续多少帧在区间内算对齐完成
stable_count = 0  # 当前连续稳定帧数
last_track_color = None  # 记录上一帧追踪的颜色，切换颜色时清零稳定计数


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
        'red': [(lower_red1, upper_red1), (lower_red2, upper_red2)],
        'green': [(lower_green, upper_green)],
        'blue': [(lower_blue, upper_blue)],
    }


def detect_color_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_ranges = get_color_ranges()
    color_info = {}

    for color_name, ranges in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obj_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                obj_list.append({"area": int(area), "cx": cx, "cy": cy})
        color_info[color_name] = obj_list
    return color_info


def get_largest_color_object(color_info):
    """
    从红蓝绿三色中找出面积最大的色块
    返回: {"color": 颜色名, "area": 面积, "cx": 中心x, "cy": 中心y}，没有则返回None
    """
    all_objects = []
    for color_name, obj_list in color_info.items():
        for obj in obj_list:
            all_objects.append({
                "color": color_name,
                "area": obj["area"],
                "cx": obj["cx"],
                "cy": obj["cy"]
            })

    if not all_objects:
        return None

    # 按面积降序排序，取最大的
    all_objects.sort(key=lambda x: x["area"], reverse=True)
    return all_objects[0]


def calc_move_speed(current_area):
    """
    返回：(enable_run, dir_code, spd_val)
    enable_run: True要运动 False停止
    dir_code: 0前进，1后退
    spd_val: 速度大小
    """
    error = TARGET_AREA - current_area
    speed = Kp * error

    # 死区，区间内直接停止
    if AREA_LOW <= current_area <= AREA_HIGH:
        return (False, 0, 0)

    if speed > 0:
        # 前进
        spd_val = int(speed)
        spd_val = max(MIN_SPEED, min(spd_val, MAX_SPEED))
        return (True, 0, spd_val)
    else:
        # 后退
        spd_val = int(-speed)
        spd_val = max(MIN_SPEED, min(spd_val, MAX_SPEED))
        return (True, 1, spd_val)


def wt(got=None):
    """
    自动追踪红蓝绿中最大的色块，距离稳定在目标区间后自动返回
    返回：True 对齐成功；False 中断或未检测到目标
    """
    global stable_count, last_track_color
    if got is None:
        got = globals().get('got')
    if got is None:
        raise RuntimeError('wt() 需要传入 ugot 实例，如 wt(got=your_ugot_instance)')
    stable_count = 0
    last_track_color = None

    try:
        while True:
            frame_bytes = got.read_camera_data()
            if frame_bytes is None:
                continue

            nparr = np.frombuffer(frame_bytes, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            info = detect_color_objects(data)
            target = get_largest_color_object(info)

            if target is not None:
                target_color = target["color"]
                target_area = target["area"]
                enable, direction, spd = calc_move_speed(target_area)

                # 颜色切换时，清零稳定计数
                if target_color != last_track_color:
                    stable_count = 0
                    last_track_color = target_color

                print(
                    f"追踪颜色:{target_color.upper():6s} | 像素:{target_area:6d} | 区间[{AREA_LOW},{AREA_HIGH}] | dir={direction} spd={spd} | 稳定帧:{stable_count}/{STABLE_FRAMES}")

                if enable:
                    stable_count = 0  # 还在运动，清零稳定计数
                    got.mecanum_move_speed_times(direction, spd, 1, 1)
                else:
                    stable_count += 1  # 进入目标区间，累计稳定帧
                    got.mecanum_move_speed_times(0, 0, 1, 1)

                    # 连续稳定足够帧数 → 对齐完成
                    if stable_count >= STABLE_FRAMES:
                        print(f"✅ 已对齐最大色块[{target_color.upper()}]，距离稳定，退出对准")
                        got.mecanum_move_speed_times(0, 0, 3, 1)
                        return True

            else:
                stable_count = 0
                last_track_color = None
                print("未检测到红蓝绿色块，停止运动")
                got.mecanum_move_speed_times(0, 0, 1, 1)

    except KeyboardInterrupt:
        print("\n手动中断，停止麦轮")
        got.mecanum_move_speed_times(0, 0, 3, 1)
        return False


# ============主流程：对齐 → 下一个动作============
if __name__ == "__main__":
    got = ugot.UGOT()
    got.initialize('10.233.104.242')
    got.open_camera()
    print("开始自动追踪最大色块（红/绿/蓝）并对齐距离...")
    success = wt(got)

    cv2.destroyAllWindows()