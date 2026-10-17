# -*- coding: utf-8 -*-
"""
smart_recon.py - 智能侦察（静止识别版）
流程：初始化 → 机器人静止，无顺序循环识别 5 类 → 按指定顺序依次播报
"""

import sys
import os
import time
import re
import cv2
import numpy as np
from ugot import ugot

ROBOT_IP = "10.233.104.242"
CHECK_INTERVAL = 0.2
MAX_LOOP_SEC   = 120
STABLE_FRAMES  = 2

REQUIRED_MODELS = [
    "color_recognition",
    "apriltag_qrcode",
    "traffic_sign",
    "gesture",
    "word_recognition",
]

VALID_SIGNS = {"左转", "右转", "红灯", "绿灯", "黄灯"}


def show_text_on_screen(got, text):
    try:
        display = text[:20] if len(text) > 20 else text
        got.screen_clear()
        got.screen_display_background(1)
        got.screen_print_text(display, 0)
    except Exception as e:
        print(f"  屏幕显示失败: {e}")


def clean_text(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、]', '', text)
    if not text:
        return None
    if len(text) > 10:
        text = text[:10]
    return text


def detect_traffic_sign(got):
    try:
        info = got.get_traffic_total_info()
        if info and len(info) > 0 and len(info[0]) > 0:
            sign = str(info[0][0])
            if sign in VALID_SIGNS:
                return sign
            else:
                print(f"  [过滤] 未知标志：{sign}")
    except Exception as e:
        print(f"  交通标志识别异常: {e}")
    return None


def detect_gesture(got):
    try:
        res = got.get_gesture_result()
        if res:
            gesture_map = {"stone": "石头", "scissors": "剪刀", "cloth": "布", "ok": "OK手势", "like": "点赞"}
            return gesture_map.get(res, res)
    except Exception as e:
        print(f"  手势识别异常: {e}")
    return None


def detect_words(got):
    try:
        res = got.get_words_result()
        if res and res.strip():
            return clean_text(res)
    except Exception as e:
        print(f"  文字识别异常: {e}")
    return None


def detect_apriltag(got):
    try:
        raw = got.get_qrcode_apriltag_total_info()
        if isinstance(raw, list) and len(raw) >= 2:
            tag_id = raw[1]
            if tag_id != -1:
                return f"{tag_id}号标签"
    except Exception as e:
        print(f"  标签识别异常: {e}")
    return None


COLOR_RANGES = {
    "红色": [(np.array([0, 167, 132]), np.array([10, 255, 255])),
             (np.array([161, 167, 132]), np.array([180, 255, 255]))],
    "绿色": [(np.array([33, 113, 62]), np.array([76, 255, 255]))],
    "蓝色": [(np.array([100, 167, 95]), np.array([143, 255, 255]))],
}
MIN_AREA = 500

def detect_color_simple(got):
    try:
        frame = got.read_camera_data()
        if frame is None:
            return None
        nparr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        best_color = None
        best_area = 0
        for color_name, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_AREA and area > best_area:
                    best_area = area
                    best_color = color_name
        if best_color:
            print(f"  [颜色调试] {best_color} (面积={best_area:.0f})")
            return best_color
    except Exception as e:
        print(f"  颜色识别异常: {e}")
    return None


RESULT_FORMAT = {
    "交通标志": "识别到交通标志为{}",
    "手势":     "识别到手势为{}",
    "文字":     "识别到文字为{}",
    "标签":     "识别到标签为{}",
    "色块":     "识别到色块为{}",
}

ALL_CATEGORIES = ["交通标志", "手势", "文字", "标签", "色块"]


def collect_all(got):
    results = {}
    pending = {}
    t0 = time.time()

    while len(results) < len(ALL_CATEGORIES) and (time.time() - t0) < MAX_LOOP_SEC:
        vals = {
            "交通标志": detect_traffic_sign(got),
            "手势":     detect_gesture(got),
            "文字":     detect_words(got),
            "标签":     detect_apriltag(got),
            "色块":     detect_color_simple(got),
        }

        for cat, v in vals.items():
            if v is None:
                pending[cat] = (None, 0)
                continue
            cand, cnt = pending.get(cat, (None, 0))
            if v == cand:
                cnt += 1
            else:
                cand, cnt = v, 1
            pending[cat] = (cand, cnt)
            need_frames = STABLE_FRAMES + 1 if cat == "色块" else STABLE_FRAMES
            if cnt < need_frames:
                continue
            if cat not in results:
                results[cat] = cand
                print(f"  认到 {cat}: {cand}  ({len(results)}/{len(ALL_CATEGORIES)})")

        time.sleep(CHECK_INTERVAL)

    print(f"识别阶段结束，共认到 {len(results)} 类：{results}")
    return results


def normalize_cat(cat):
    cat = cat.strip()
    if cat in ("颜色", "色块颜色", "色"):
        return "色块"
    return cat


def detective_full_flow(got, sequence):
    sequence = [normalize_cat(c) for c in sequence]
    seen = set()
    dedup = []
    for c in sequence:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    sequence = dedup

    print("\n" + "=" * 60)
    print(f" 智能侦察流程：播报顺序 {' → '.join(sequence)}")
    print("=" * 60)

    try:
        got.play_audio_tts("开始侦察任务", 0, True)
    except Exception:
        pass
    show_text_on_screen(got, "侦察模式")

    print("\n===== 识别阶段（静止，请依次展示道具）=====")
    show_text_on_screen(got, "请展示道具")
    results = collect_all(got)

    print("\n===== 播报阶段 =====")
    for cat in sequence:
        v = results.get(cat)
        if not v:
            print(f"  {cat} 未识别到，跳过")
            continue
        speech = RESULT_FORMAT[cat].format(v)
        print(f"  播报: {speech}")
        try:
            got.play_audio_tts(speech, 0, True)
        except Exception as e:
            print(f"  语音播报失败: {e}")
        show_text_on_screen(got, v)
        time.sleep(0.3)

    print("\n===== 流程结束 =====")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python smart_recon.py <播报顺序>")
        sys.exit(1)

    seq_str = sys.argv[1]
    sequence = [s.strip() for s in seq_str.split(",") if s.strip()]
    if not sequence:
        print("错误：顺序为空")
        sys.exit(1)

    print("=== smart_recon.py 启动（静止识别版）===")
    print(f"播报顺序: {' → '.join(sequence)}")
    print(f"确认帧数: {STABLE_FRAMES}帧")

    got = ugot.UGOT()
    got.initialize(ROBOT_IP)
    print("机器人连接成功")

    print("正在加载识别模型...")
    try:
        got.open_camera()
    except Exception as e:
        print(f"打开摄像头提示（非致命）: {e}")
    try:
        got.load_models(REQUIRED_MODELS)
    except Exception as e:
        print(f"加载模型提示: {e}")

    time.sleep(2)
    print("模型加载完成，等待稳定...")
    time.sleep(1)

    try:
        success = detective_full_flow(got, sequence)
    except Exception as e:
        print(f"流程异常: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print(f"\n执行结果: success={success}")
    sys.exit(0 if success else 1)