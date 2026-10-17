# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
from ugot import ugot

ROBOT_IP = '10.233.104.242'
MIN_AREA = 500
PRINT_INTERVAL = 0.2

_MORPH_KERNEL = np.ones((5, 5), np.uint8)

def _get_color_ranges():
    """HSV颜色阈值（与原抓取逻辑完全一致）"""
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
    """读取摄像头原始帧"""
    frame = got.read_camera_data()
    if frame is None:
        return None
    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return data


def detect_color_objects(frame, min_area=MIN_AREA):
    """
    检测画面中所有颜色色块
    返回字典：{颜色名: [{"center": (x,y), "area": 面积, "rect": (x,y,w,h)}, ...]}
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = {}

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obj_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                obj_list.append({
                    "center": (center_x, center_y),
                    "area": int(area),
                    "rect": (x, y, w, h)
                })
        obj_list.sort(key=lambda x: x["area"], reverse=True)
        result[color_name] = obj_list

    return result


def main():
    print("正在连接机器人...")
    got = ugot.UGOT()
    got.initialize(ROBOT_IP)
    print("连接成功")

    print("正在打开摄像头...")
    got.open_camera()
    time.sleep(0.5)
    print("摄像头已就绪，开始颜色检测")
    print("=" * 60)
    print("提示：按 Ctrl+C 退出程序")
    print("=" * 60)

    last_print_time = 0

    try:
        while True:
            frame = _read_camera_frame(got)
            if frame is None:
                print("未获取到摄像头画面...")
                time.sleep(0.1)
                continue

            color_result = detect_color_objects(frame)

            now = time.time()
            if now - last_print_time >= PRINT_INTERVAL:
                last_print_time = now
                print(f"\n[帧时间: {time.strftime('%H:%M:%S')}]")

                total_count = 0
                for color_name, obj_list in color_result.items():
                    count = len(obj_list)
                    total_count += count
                    if count > 0:
                        largest = obj_list[0]
                        print(f"  {color_name}: 检测到 {count} 个 | 最大面积: {largest['area']}px | 中心: {largest['center']}")
                    else:
                        print(f"  {color_name}: 未检测到")

                print(f"  画面尺寸: {frame.shape[1]}x{frame.shape[0]} | 总目标数: {total_count}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n手动停止检测")
    finally:
        try:
            got.close_camera()
        except Exception:
            pass
        print("程序已退出")


if __name__ == "__main__":
    main()