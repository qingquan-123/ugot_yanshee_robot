# -*- coding: utf-8 -*-
"""
testtag.py - AprilTag 识别与对准独立测试脚本
从 smart_warehouse.py 中摘取的 AprilTag 相关函数，可直接用于 ugot 机器人测试。

功能：
  1. 连接机器人 → 打开摄像头 → 加载 apriltag_qrcode 模型
  2. 五阶段自动对准：搜索 → 粗对准 → 中速对准 → 精对准 → 稳定确认
  3. 对准完成后打印标签 ID、距离、偏航角等信息

用法：
  python testtag.py              # 自动对准视野中第一个识别到的 Tag
  python testtag.py 3            # 只对准 ID=3 的 Tag，其他 Tag 忽略
  python testtag.py --ip 10.233.104.242 3   # 指定机器人 IP 和目标 Tag

修改方式：
  - 机器人 IP：修改下方 ROBOT_IP 常量，或用 --ip 参数传入
  - 目标 Tag：修改 TARGET_TAG_ID 常量（None 表示任意 Tag），或命令行传入
  - 对准精度：调整 FINE_THRESHOLD（越小越精准，默认 10px）
  - 对准速度：调整 COARSE_SPEED / MID_SPEED / FINE_KP
  - Tag 尺寸：修改 TAG_SIZE（'5' / '7' / '10'，对应距离计算索引）
"""

import sys
import time
import math
from ugot import ugot

ROBOT_IP = '10.233.104.242'

TARGET_TAG_ID = None

FRAME_WIDTH = 640
FRAME_CENTER = FRAME_WIDTH // 2

COARSE_THRESHOLD = 50
MID_THRESHOLD = 25
FINE_THRESHOLD = 10
FINE_DEAD_ZONE = 1
STABLE_FRAMES = 3

SEARCH_SPEED = 6
COARSE_SPEED = 12
MID_SPEED = 8
FINE_MIN_SPEED = 2
FINE_KP = 0.35

TAG_SIZE = '10'

STATE_SEARCH = 0
STATE_COARSE_ALIGN = 1
STATE_MID_ALIGN = 2
STATE_FINE_ALIGN = 3
STATE_STABLE_CHECK = 4



def parse_tag_data(raw_data):
    """
    解析 get_qrcode_apriltag_total_info() 的原始返回数据。

    返回数据索引说明：
      0: 二维码文本    1: id          2: center_x    3: center_y
      4: height        5: width       6: area
      7: distance5     8: distance7   9: distance10
      10: 姿态角x(rad) 11: 姿态角y(rad) 12: 姿态角z(rad)

    参数:
        raw_data: 机器人返回的列表数据
    返回:
        dict: 包含 id, center_x, center_y, area, distance(cm), yaw_deg 的字典
        None: 未识别到标签或数据格式错误
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
        "distance": raw_data[dist_idx] * 100,
        "yaw_deg": math.degrees(raw_data[12])
    }


def align_apriltag(got):
    """
    AprilTag 五阶段自动对准。

    阶段流程：
      搜索(SEARCH) → 粗对准(COARSE) → 中速对准(MID) → 精对准(FINE) → 稳定确认(STABLE)

    任意阶段丢失标签都会回退到上一阶段重新捕捉。

    参数:
        got: ugot.UGOT() 实例，需已连接并加载 apriltag_qrcode 模型
    返回:
        (success: bool, tag_info: dict|None)
        success=True 时 tag_info 包含最终对准的标签信息
    """
    current_state = STATE_SEARCH
    stable_count = 0
    last_direction = 0

    try:
        while True:
            raw_data = got.get_qrcode_apriltag_total_info()
            tag = parse_tag_data(raw_data)

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

                offset = tag["center_x"] - FRAME_CENTER
                if abs(offset) <= FINE_THRESHOLD:
                    print(f"[搜索阶段] 识别到目标标签 ID={tag['id']}，偏差 {offset:+.0f}px 已在精对准范围(±{FINE_THRESHOLD}px)内，跳过粗/中对准，直接进入精对准慢修")
                    got.mecanum_move_xyz(0, 0, 0)
                    current_state = STATE_FINE_ALIGN
                    stable_count = 0
                else:
                    print(f"[搜索阶段] 识别到目标标签 ID={tag['id']}，偏差 {offset:+.0f}px，进入粗对准")
                    current_state = STATE_COARSE_ALIGN
                    stable_count = 0

            elif current_state == STATE_COARSE_ALIGN:
                if tag is None:
                    print("[粗对准] 标签丢失，返回搜索阶段")
                    current_state = STATE_SEARCH
                    continue

                offset = tag["center_x"] - FRAME_CENTER
                print(f"[粗对准] ID:{tag['id']} | 偏移:{offset:+.0f}px | 阈值:±{COARSE_THRESHOLD}px")

                if abs(offset) <= COARSE_THRESHOLD:
                    print("[粗对准] 进入中速对准阶段")
                    current_state = STATE_MID_ALIGN
                    continue

                direction = 1 if offset > 0 else -1
                got.mecanum_move_xyz(0, 0, -direction * COARSE_SPEED)
                last_direction = direction
                time.sleep(0.05)

            elif current_state == STATE_MID_ALIGN:
                if tag is None:
                    print("[中速对准] 标签丢失，返回粗对准")
                    current_state = STATE_COARSE_ALIGN
                    continue

                offset = tag["center_x"] - FRAME_CENTER
                print(f"[中速对准] ID:{tag['id']} | 偏移:{offset:+.0f}px | 阈值:±{MID_THRESHOLD}px")

                if abs(offset) <= MID_THRESHOLD:
                    print("[中速对准] 进入精对准阶段")
                    current_state = STATE_FINE_ALIGN
                    continue

                direction = 1 if offset > 0 else -1
                got.mecanum_move_xyz(0, 0, -direction * MID_SPEED)
                last_direction = direction
                time.sleep(0.05)

            elif current_state == STATE_FINE_ALIGN:
                if tag is None:
                    print("[精对准] 标签丢失，返回中速对准")
                    current_state = STATE_MID_ALIGN
                    continue

                offset = tag["center_x"] - FRAME_CENTER
                print(f"[精对准] ID:{tag['id']} | 偏移:{offset:+.0f}px | 死区:±{FINE_DEAD_ZONE}px")

                if abs(offset) <= FINE_DEAD_ZONE:
                    print("[精对准] 偏移进入死区，进入稳定确认阶段")
                    current_state = STATE_STABLE_CHECK
                    stable_count = 0
                    got.mecanum_move_xyz(0, 0, 0)
                    continue

                direction = 1 if offset > 0 else -1
                speed = FINE_KP * abs(offset)
                speed = max(FINE_MIN_SPEED, min(speed, MID_SPEED))
                speed = int(speed)
                if abs(offset) <= FINE_THRESHOLD:
                    speed = 1

                got.mecanum_move_xyz(0, 0, -direction * speed)
                last_direction = direction
                time.sleep(0.05)

            elif current_state == STATE_STABLE_CHECK:
                if tag is None:
                    print("[稳定确认] 标签丢失，返回精对准")
                    current_state = STATE_FINE_ALIGN
                    continue

                offset = tag["center_x"] - FRAME_CENTER
                if abs(offset) <= FINE_DEAD_ZONE:
                    stable_count += 1
                    print(f"[稳定确认] 居中保持中... {stable_count}/{STABLE_FRAMES} (偏移:{offset:+.0f}px, 死区:±{FINE_DEAD_ZONE}px)")
                    got.mecanum_move_xyz(0, 0, 0)

                    if stable_count >= STABLE_FRAMES:
                        print(f"\n===== 对准完成！ =====")
                        print(f"  标签 ID   : {tag['id']}")
                        print(f"  中心偏移  : {offset:+.0f}px")
                        print(f"  距离      : {tag['distance']:.1f}cm")
                        print(f"  偏航角    : {tag['yaw_deg']:.1f}°")
                        print(f"  标签面积  : {tag['area']:.0f}px²")
                        print("======================\n")
                        got.mecanum_stop()
                        return True, tag
                else:
                    print(f"[稳定确认] 偏移 {offset:+.0f}px 超出死区±{FINE_DEAD_ZONE}px，退回精对准")
                    stable_count = 0
                    current_state = STATE_FINE_ALIGN

                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n手动中断对准（Ctrl+C）")
        got.mecanum_stop()
        return False, None


def align_with_apriltag(got):
    """
    AprilTag 对准完整流程：加载模型 → 五阶段对准 → 释放模型。

    这是对外的主入口函数，调用前需确保 got 已连接且摄像头已打开。

    参数:
        got: ugot.UGOT() 实例
    返回:
        (success: bool, tag_info: dict|None)
    """
    print("\n===== AprilTag 对准 =====")
    print("加载 apriltag_qrcode 模型...")
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
        print("AprilTag 对准失败或被中断")

    return success, tag_info



def parse_args():
    """
    解析命令行参数。

    支持格式：
      python testtag.py
      python testtag.py <tag_id>
      python testtag.py --ip <robot_ip>
      python testtag.py --ip <robot_ip> <tag_id>

    返回:
        (robot_ip: str, target_tag_id: int|None)
    """
    robot_ip = ROBOT_IP
    target_tag_id = TARGET_TAG_ID

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--ip' and i + 1 < len(args):
            robot_ip = args[i + 1]
            i += 2
        elif args[i].isdigit():
            target_tag_id = int(args[i])
            i += 1
        else:
            i += 1

    return robot_ip, target_tag_id


def main():
    """
    主函数：连接机器人 → 打开摄像头 → 对准 Tag → 清理资源。
    """
    robot_ip, target_tag_id = parse_args()

    global TARGET_TAG_ID
    TARGET_TAG_ID = target_tag_id

    print("=" * 50)
    print("  AprilTag 识别与对准测试脚本")
    print(f"  机器人 IP : {robot_ip}")
    print(f"  目标 Tag  : {target_tag_id if target_tag_id is not None else '任意'}")
    print(f"  Tag 尺寸  : {TAG_SIZE}cm")
    print(f"  精对准阈值: ±{FINE_THRESHOLD}px")
    print("=" * 50)

    got = None
    try:
        print("\n[1/4] 连接机器人...")
        got = ugot.UGOT()
        got.initialize(robot_ip)
        print("      连接成功")

        print("\n[2/4] 打开摄像头...")
        got.open_camera()
        time.sleep(0.5)
        print("      摄像头已开启")

        print("\n[3/4] 开始 AprilTag 对准...")
        success, tag_info = align_with_apriltag(got)

        print("\n[4/4] 测试结束")
        if success:
            print(f"      结果: 对准成功 (ID={tag_info['id']}, 距离={tag_info['distance']:.1f}cm)")
        else:
            print("      结果: 对准失败或被中断")

    except Exception as e:
        print(f"\n运行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if got is not None:
            print("\n清理资源...")
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
