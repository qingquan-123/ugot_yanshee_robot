# -*- coding: utf-8 -*-
from ugot import ugot
import time
import math

ROBOT_IP = '10.233.104.242'

# ==================== 配置参数（已优化） ====================
TARGET_TAG_ID = None

FRAME_WIDTH = 640
FRAME_CENTER = FRAME_WIDTH // 2

# 阈值优化：粗调放宽，增加中速区，精调保持精度
COARSE_THRESHOLD = 60    # 粗对准阈值，快速拉到中速区
MID_THRESHOLD = 30       # 中速对准阈值，快速靠近
FINE_THRESHOLD = 17
CALIBRATION_OFFSET = -16      # 精对准阈值，最终精度
STABLE_FRAMES = 2        # 连续2帧稳定即完成对准
MAX_STABLE_ATTEMPTS = 3  # 最多尝试3次稳定确认，第3次直接退出

# 速度优化：分三档速度
SEARCH_SPEED = 8
COARSE_SPEED = 15
MID_SPEED = 8            # 新增中速，兼顾速度和稳定
FINE_MIN_SPEED = 3       # 精调最低速度
FINE_KP = 0.25           # 精调比例系数，越大越快

TAG_SIZE = '10'
# ==========================================================

# 状态机：新增中速对准阶段
STATE_SEARCH = 0
STATE_COARSE_ALIGN = 1
STATE_MID_ALIGN = 2      # 新增：中速对准
STATE_FINE_ALIGN = 3
STATE_STABLE_CHECK = 4


def parse_tag_data(raw_data):
    """
    解析 get_qrcode_apriltag_total_info 的原始返回数据
    返回：标签信息字典，未识别到返回None
    字段对应索引：
    0:二维码文本  1:id  2:center_x  3:center_y  4:height  5:width  6:area
    7:distance5   8:distance7   9:distance10
    10:姿态角x(rad)  11:姿态角y(rad)  12:姿态角z(rad)
    """
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
        "distance": raw_data[dist_idx] * 100,  # 转成厘米
        "yaw_deg": math.degrees(raw_data[12])  # 偏航角转角度
    }


def align_apriltag(got):
    current_state = STATE_SEARCH
    stable_count = 0
    last_direction = 0
    stable_attempts = 0

    try:
        while True:
            raw_data = got.get_qrcode_apriltag_total_info()
            tag = parse_tag_data(raw_data)

            # ===================== 阶段1：搜索标签 =====================
            if current_state == STATE_SEARCH:
                if tag is None:
                    print("[搜索阶段] 未识别到标签，自转搜索中...")
                    got.mecanum_move_xyz(0, 0, SEARCH_SPEED)
                    time.sleep(0.1)
                    continue

                if TARGET_TAG_ID is not None and tag["id"] != TARGET_TAG_ID:
                    print(f"[搜索阶段] 识别到ID={tag['id']}，非目标ID={TARGET_TAG_ID}，继续搜索")
                    got.mecanum_move_xyz(0, 0, SEARCH_SPEED)
                    time.sleep(0.1)
                    continue

                print(f"[搜索阶段] 识别到目标标签 ID={tag['id']}，进入粗对准")
                current_state = STATE_COARSE_ALIGN
                stable_count = 0

            # ===================== 阶段2：粗对准 =====================
            elif current_state == STATE_COARSE_ALIGN:
                if tag is None:
                    print("[粗对准] 标签丢失，返回搜索阶段")
                    current_state = STATE_SEARCH
                    continue

                offset = tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                print(f"[粗对准] ID:{tag['id']} | 偏移:{offset:+.0f}px | 阈值:±{COARSE_THRESHOLD}px")

                if abs(offset) <= COARSE_THRESHOLD:
                    print("[粗对准] 进入中速对准阶段")
                    current_state = STATE_MID_ALIGN
                    continue

                direction = 1 if offset > 0 else -1
                got.mecanum_move_xyz(0, 0, -direction * COARSE_SPEED)
                last_direction = direction
                time.sleep(0.05)

            # ===================== 阶段3：中速对准（新增，提速关键） =====================
            elif current_state == STATE_MID_ALIGN:
                if tag is None:
                    print("[中速对准] 标签丢失，返回粗对准")
                    current_state = STATE_COARSE_ALIGN
                    continue

                offset = tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                print(f"[中速对准] ID:{tag['id']} | 偏移:{offset:+.0f}px | 阈值:±{MID_THRESHOLD}px")

                if abs(offset) <= MID_THRESHOLD:
                    print("[中速对准] 进入精对准阶段")
                    current_state = STATE_FINE_ALIGN
                    continue

                direction = 1 if offset > 0 else -1
                got.mecanum_move_xyz(0, 0, -direction * MID_SPEED)
                last_direction = direction
                time.sleep(0.05)

            # ===================== 阶段4：精对准（比例调速 + 整数修复） =====================
            elif current_state == STATE_FINE_ALIGN:
                if tag is None:
                    print("[精对准] 标签丢失，返回中速对准")
                    current_state = STATE_MID_ALIGN
                    continue

                offset = tag["center_x"] - FRAME_CENTER - CALIBRATION_OFFSET
                print(f"[精对准] ID:{tag['id']} | 偏移:{offset:+.0f}px | 阈值:±{FINE_THRESHOLD}px")

                if abs(offset) <= FINE_THRESHOLD:
                    stable_attempts += 1
                    print(f"[精对准] 偏移达标，进入稳定确认阶段（第{stable_attempts}次）")
                    current_state = STATE_STABLE_CHECK
                    stable_count = 0
                    got.mecanum_move_xyz(0, 0, 0)

                    if stable_attempts >= MAX_STABLE_ATTEMPTS:
                        print(f"已尝试{stable_attempts}次稳定确认，直接继续执行")
                        got.mecanum_stop()
                        return True, tag
                    continue

                # 比例控制：偏差越大速度越快，偏差越小速度越慢
                direction = 1 if offset > 0 else -1
                speed = FINE_KP * abs(offset)
                speed = max(FINE_MIN_SPEED, min(speed, MID_SPEED))
                speed = int(speed)

                got.mecanum_move_xyz(0, 0, -direction * speed)
                last_direction = direction
                time.sleep(0.05)

            # ===================== 阶段5：稳定确认 =====================
            elif current_state == STATE_STABLE_CHECK:
                if tag is None:
                    print("[稳定确认] 标签丢失，返回精对准")
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
                    print("[稳定确认] 偏移超出阈值，退回精对准")
                    stable_count = 0
                    current_state = STATE_FINE_ALIGN

                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n手动中断对准")
        got.mecanum_stop()
        return False, None


def wttag():
    got = None
    try:
        print("连接机器人...")
        got = ugot.UGOT()
        got.initialize(ROBOT_IP)
        print("连接成功")

        print("打开摄像头...")
        got.open_camera()
        time.sleep(0.5)

        print("加载AprilTag识别模型...")
        if not got.load_models(['apriltag_qrcode']):
            print("模型加载失败")
            return
        print("模型加载完成，等待1秒稳定...")
        time.sleep(1)

        print("\n开始AprilTag对准\n")
        success, tag_info = align_apriltag(got)

    except Exception as e:
        print(f"运行异常：{e}")
        import traceback
        traceback.print_exc()
    finally:
        if got:
            print("\n清理资源...")
            try:
                got.release_models(['apriltag_qrcode'])
                # got.close_camera()
            except Exception:
                pass
            got.mecanum_stop()
            print("程序结束")


if __name__ == "__main__":
    wttag()