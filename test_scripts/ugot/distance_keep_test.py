from ugot import ugot
import cv2
import numpy as np

got = ugot.UGOT()
got.initialize('10.233.104.242')
got.open_camera()

# ============距离维持控制参数============
TARGET_AREA = 45000      # 目标像素中间值，区间40000‑50000
AREA_LOW = 38000         # 下限
AREA_HIGH = 50000        # 上限
Kp = 0.0008
MAX_SPEED = 8
MIN_SPEED = 3

def get_color_ranges():
    lower_red1 = np.array([0, 172, 44])
    upper_red1 = np.array([21, 255, 255])
    lower_red2 = np.array([174, 172, 44])
    upper_red2 = np.array([179, 255, 255])

    lower_green = np.array([50, 136, 84])
    upper_green = np.array([66, 255, 255])
    lower_blue = np.array([100, 117, 107])
    upper_blue = np.array([113, 255, 255])

    return {
        'red': [(lower_red1, upper_red1), (lower_red2, upper_red2)],
        'green': [(lower_green, upper_green)],
        'blue': [(lower_blue, upper_blue)],
    }

color_bgr = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'other': (0, 0, 0)
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color_name], 2)
                text = f"{color_name.upper()} A:{area}"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_name], 2)
        color_info[color_name] = obj_list
    return frame, color_info


def calc_move_speed(current_area):
    """
    返回：(enable_run, dir_code, spd_val)
    enable_run: True要运动 False停止
    dir_code: 0前进，1后退
    spd_val: 速度大小10~15
    """
    error = TARGET_AREA - current_area
    speed = Kp * error

    # 死区，40000‑50000直接停止
    if AREA_LOW <= current_area <= AREA_HIGH:
        return (False, 0, 0)

    if speed > 0:
        # 需要前进
        spd_val = int(speed)
        if spd_val > MAX_SPEED:
            spd_val = MAX_SPEED
        if spd_val < MIN_SPEED:
            spd_val = MIN_SPEED
        return (True, 0, spd_val)
    else:
        # 需要后退
        spd_val = int(-speed)
        if spd_val > MAX_SPEED:
            spd_val = MAX_SPEED
        if spd_val < MIN_SPEED:
            spd_val = MIN_SPEED
        return (True, 1, spd_val)


try:
    while True:
        frame_bytes = got.read_camera_data()
        if frame_bytes is not None:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            draw_frame, info = detect_color_objects(data)

            red_list = info["red"]
            if len(red_list) > 0:
                red_target = max(red_list, key=lambda x: x["area"])
                red_area = red_target["area"]
                enable, direction, spd = calc_move_speed(red_area)
                print(f"红色像素:{red_area:6d} | 区间[{AREA_LOW},{AREA_HIGH}] | dir={direction} spd={spd}")

                if enable:
                    # mecanum_move_speed_times(direction, speed, time, unit)
                    # unit=0:毫秒，给9999ms长时间运行
                    got.mecanum_move_speed_times(direction, spd, 1, 1)
                else:
                    got.mecanum_move_speed_times(0, 0, 1, 1)

            else:
                print("未检测红色色块，停止运动")
                got.mecanum_move_speed_times(0, 0, 1, 1)

            cv2.imshow("Area Keep 40000~50000", draw_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                got.mecanum_move_speed_times(0, 0, 1, 1)
                break

except KeyboardInterrupt:
    print("\n手动中断，停止麦轮")
    got.mecanum_move_speed_times(0, 0, 3, 1)
finally:
    cv2.destroyAllWindows()