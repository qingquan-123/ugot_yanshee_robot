# coding=UTF-8
from time import sleep
import cv2
import os
import time
import threading
import numpy as np
import math
from multiprocessing import Process
import multiprocessing
import inspect
import ctypes
import json
import socket


import YanAPI

BROADCAST_PORT = 9527

def tts(content):
    print("播报:", content)
    try:
        YanAPI.sync_do_tts(content, True)
    except Exception as e:
        print("播报失败：", str(e))

def move_left():
    YanAPI.sync_play_motion("te_left_1_S7")

def move_right():
    YanAPI.sync_play_motion("te_right_S5")

def move_left_big():
    YanAPI.sync_play_motion("trte_L_J_big_02")

def move_right_big():
    YanAPI.sync_play_motion("trte_R_J_big_04t")

def turn_left():
    YanAPI.sync_play_motion("turn_L_S2")

def turn_right():
    YanAPI.sync_play_motion("turn_r_S1")

def quat_to_euler_deg(qx, qy, qz, qw):
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def translate(target_id=2, align_threshold=0.02, auto_break=True):
    tag_list = [{"id":i, "size":0.06} for i in range(6)]
    print("启动AprilTag识别，目标标签ID：{}".format(target_id))
    resp_start = YanAPI.start_aprilTag_recognition(tags=tag_list, enableStream=False)
    if resp_start.get("code") != 0:
        print("启动AprilTag识别失败！")
        return
    last_spoken_id = -1
    is_aligned = False
    try:
        while True:
            status_ret = YanAPI.get_aprilTag_recognition_status()
            data = status_ret.get("data", {})
            tag_array = data.get("AprilTagStatus", [])
            if len(tag_array) > 0:
                tag_info = tag_array[0]
                tag_id = tag_info["id"]
                pos_x = tag_info.get("position-x", None)
                if tag_id != target_id:
                    if tag_id != last_spoken_id:
                        print("识别到{}号标签，非目标，跳过".format(tag_id))
                        last_spoken_id = tag_id
                    is_aligned = False
                    time.sleep(0.2)
                    continue
                if pos_x is None:
                    time.sleep(0.2)
                    continue
                print("目标TagID:{}  position-x:{:.4f}".format(tag_id, pos_x))
                if tag_id != last_spoken_id:
                    print("识别到目标标签，编号{}".format(tag_id))
                    last_spoken_id = tag_id
                if pos_x > align_threshold:
                    print("标签偏右，执行右移")
                    move_right()
                    is_aligned = False
                elif pos_x < -align_threshold:
                    print("标签偏左，执行左移")
                    move_left()
                    is_aligned = False
                else:
                    if not is_aligned:
                        print("已正对目标{}号标签".format(target_id))
                        is_aligned = True
                        if auto_break:
                            break
            else:
                print("未检测到标签")
                last_spoken_id = -1
                is_aligned = False
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n停止识别")
    finally:
        YanAPI.stop_aprilTag_recognition()

def tat(target_id=3):
    tag_list = [{"id":i, "size":0.06} for i in range(6)]
    print("启动AprilTag识别")
    resp_start = YanAPI.start_aprilTag_recognition(tags=tag_list, enableStream=False)
    if resp_start.get("code") != 0:
        print("启动AprilTag识别失败！")
        return
    TARGET_TAG_ID = target_id
    pos_threshold = 0.013
    init_angle_threshold = 10
    fine_angle_threshold = 6
    init_turn_max = 2
    fine_turn_max = 3
    STATE_INIT_TURN = 0
    STATE_INIT_POS = 1
    STATE_SEARCH_TAG = 2
    STATE_TARGET_POS_1 = 3
    STATE_TARGET_ANGLE = 4
    STATE_TARGET_POS_2 = 5
    current_state = STATE_INIT_TURN
    init_turn_count = 0
    angle_align_count = 0
    try:
        while True:
            status_ret = YanAPI.get_aprilTag_recognition_status()
            data = status_ret.get("data", {})
            tag_array = data.get("AprilTagStatus", [])
            has_tag = len(tag_array) > 0
            tag_id = -1
            pos_x = 0.0
            target_angle = 0.0
            if has_tag:
                tag_info = tag_array[0]
                tag_id = tag_info["id"]
                pos_x = tag_info.get("position-x", 0.0)
                qx = tag_info.get("orientation-x")
                qy = tag_info.get("orientation-y")
                qz = tag_info.get("orientation-z")
                qw = tag_info.get("orientation-w")
                if None not in (qx, qy, qz, qw):
                    _, pitch_deg, _ = quat_to_euler_deg(qx, qy, qz, qw)
                    target_angle = pitch_deg
            if current_state == STATE_INIT_TURN:
                if not has_tag:
                    time.sleep(0.2)
                    continue
                print("阶段1[初始转向]：ID:{} 角度:{:.1f}°".format(tag_id, target_angle))
                if abs(target_angle) <= init_angle_threshold:
                    current_state = STATE_INIT_POS
                elif init_turn_count < init_turn_max:
                    if target_angle > init_angle_threshold:
                        turn_right()
                    else:
                        turn_left()
                    init_turn_count += 1
                else:
                    current_state = STATE_INIT_POS
            elif current_state == STATE_INIT_POS:
                if not has_tag:
                    current_state = STATE_INIT_TURN
                    continue
                print("阶段2[初始平移]：ID:{} X:{:.4f}".format(tag_id, pos_x))
                if pos_x > pos_threshold:
                    move_right()
                elif pos_x < -pos_threshold:
                    move_left()
                else:
                    current_state = STATE_SEARCH_TAG
            elif current_state == STATE_SEARCH_TAG:
                if not has_tag:
                    time.sleep(0.2)
                    continue
                print("阶段3[搜索]：当前ID:{} 目标ID:{}".format(tag_id, TARGET_TAG_ID))
                if tag_id == TARGET_TAG_ID:
                    current_state = STATE_TARGET_POS_1
                elif tag_id < TARGET_TAG_ID:
                    move_left_big()
                    time.sleep(0.4)
                else:
                    move_right_big()
                    time.sleep(0.4)
            elif current_state == STATE_TARGET_POS_1:
                if not has_tag or tag_id != TARGET_TAG_ID:
                    current_state = STATE_SEARCH_TAG
                    continue
                print("阶段4[目标一次对准]：X:{:.4f}".format(pos_x))
                if pos_x > pos_threshold:
                    move_right()
                elif pos_x < -pos_threshold:
                    move_left()
                else:
                    angle_align_count = 0
                    current_state = STATE_TARGET_ANGLE
            elif current_state == STATE_TARGET_ANGLE:
                if not has_tag or tag_id != TARGET_TAG_ID:
                    current_state = STATE_SEARCH_TAG
                    continue
                print("阶段5[角度精调]：角度:{:.1f}°".format(target_angle))
                if abs(target_angle) <= fine_angle_threshold:
                    current_state = STATE_TARGET_POS_2
                elif angle_align_count < fine_turn_max:
                    if target_angle > fine_angle_threshold:
                        turn_right()
                    else:
                        turn_left()
                    angle_align_count += 1
                else:
                    current_state = STATE_TARGET_POS_2
            elif current_state == STATE_TARGET_POS_2:
                if not has_tag or tag_id != TARGET_TAG_ID:
                    current_state = STATE_SEARCH_TAG
                    continue
                print("阶段6[二次对准]：X:{:.4f}".format(pos_x))
                if pos_x > pos_threshold:
                    move_right()
                elif pos_x < -pos_threshold:
                    move_left()
                else:
                    print("对准完成！")
                    break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n完成")
    finally:
        YanAPI.stop_aprilTag_recognition()
def receive_once():
    print("\n等待UGOT广播任务... 端口：" + str(BROADCAST_PORT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", BROADCAST_PORT))
    while True:
        try:
            raw_data, addr = sock.recvfrom(1024)
            payload = json.loads(raw_data.decode("utf-8"))
            print("收到数据：", payload)
            sock.close()
            return payload
        except Exception as e:
            print("等待中...", e)
            continue
def init_path():
    YanAPI.upload_vision_photo_sample("/home/pi/Desktop/sample/Michael.png")
    YanAPI.upload_vision_photo_sample("/home/pi/Desktop/sample/Nancy.png")
    YanAPI.upload_vision_photo_sample("/home/pi/Desktop/sample/Simon.png")
    YanAPI.set_vision_tag(["Michael.png"], "Michael")
    YanAPI.set_vision_tag(["Nancy.png"], "Nancy")
    YanAPI.set_vision_tag(["Simon.png"], "Simon")


def face_recognition():
    try:
        res = YanAPI.sync_do_face_recognition("recognition")
        if res and 'data' in res and res['data'] and 'recognition' in res['data']:
            recognition = res['data']['recognition']
            if recognition and 'name' in recognition:
                result_name = recognition["name"]
                print("result:", result_name)
                return result_name
        return "none"
    except Exception as e:
        print('程序出错：', str(e))
        return "none"


def run_detecte_face():
    init_path()
    detected_count = 0
    detected_names = []
    while detected_count < 3:
        print("start")
        name = face_recognition()
        if name != "none":
            print("这个人是" + name)
            tts("这个人是" + name)
            detected_count += 1
            detected_names.append(name)
            print("已识别 {}/3 人".format(detected_count))
        else:
            print("no face，继续等待")
        time.sleep(0.5)

def run_detect_face():
    YanAPI.upload_vision_photo_sample("/home/pi/Desktop/sample/Michael.png")
    YanAPI.upload_vision_photo_sample("/home/pi/Desktop/sample/Nancy.png")
    YanAPI.upload_vision_photo_sample("/home/pi/Desktop/sample/Simon.png")

    print(YanAPI.get_vision_photo_samples())
    YanAPI.set_vision_tag(["Michael.png"], "Michael")
    YanAPI.set_vision_tag(["Nancy.png"], "Nancy")
    YanAPI.set_vision_tag(["Simon.png"], "Simon")

    print(YanAPI.get_vision_photo_samples())
    print(YanAPI.get_vision_tags())

    for i in range(3):
        print("start")
        name = face_recognition()
        if name != "none":
            print("这个人是" + name)
            tts("这个人是" + name)
        else:
            print("no face")


COLOR_RANGE = {
    "红色": [
        (np.array([0, 60, 40]), np.array([10, 255, 255])),
        (np.array([156, 60, 40]), np.array([180, 255, 255]))
    ],
    "绿色": [(np.array([33, 60, 40]), np.array([76, 255, 255]))],
    "蓝色": [(np.array([90, 60, 40]), np.array([143, 255, 255]))]
}
BOX_COLOR = {
    "红色": (0, 0, 255),
    "绿色": (0, 255, 0),
    "蓝色": (255, 0, 0)
}
COLOR_EN_NAME = {
    "红色": "red",
    "绿色": "green",
    "蓝色": "blue"
}

def open_camera(duration=4):
    print("正在打开摄像头，色块检测中...")
    try:
        YanAPI.stop_aprilTag_recognition()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("第一次打开失败，等待1秒重试...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    cv2.namedWindow("ColorDetect", cv2.WINDOW_AUTOSIZE)
    start_time = time.time()
    min_area = 200

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        for color_name in ["红色", "绿色", "蓝色"]:
            ranges = COLOR_RANGE[color_name]
            box_bgr = BOX_COLOR[color_name]
            en_text = COLOR_EN_NAME[color_name]
            mask = None

            for lower, upper in ranges:
                current_mask = cv2.inRange(hsv, lower, upper)
                mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_area = 0
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area and area >= min_area:
                    max_area = area
                    max_contour = contour

            if max_contour is not None:
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_bgr, 2)
                cv2.putText(frame, en_text, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_bgr, 2)

        cv2.imshow("ColorDetect", frame)

        if time.time() - start_time > duration:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已关闭")

def move_to_A_area():
    print("go A")
    YanAPI.sync_play_motion("restoration_h_05")

    YanAPI.sync_play_motion("trte_L_J_big_18b")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")


    YanAPI.sync_play_motion("A_walk_01_909B")


    YanAPI.sync_play_motion("trte_L_J_big_09A")
    YanAPI.sync_play_motion("te_left_1_01")
    YanAPI.sync_play_motion("te_left_1_01")
    YanAPI.sync_play_motion("squat_805")

    print("打开摄像头")
    open_camera()
    print("到A号柱子")
    YanAPI.sync_play_motion("Attention_805")
    YanAPI.sync_play_motion("A_walk_J_02_909A")
    YanAPI.sync_play_motion("A_walk_J_02_811A")

    print("抓取")
    YanAPI.sync_play_motion("clamp_922A")
    YanAPI.sync_play_motion("clamp_back_914")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("trte_R_J_big_03")


def move_to_B_area():
    print("go B")
    YanAPI.sync_play_motion("restoration_h_05")


    YanAPI.sync_play_motion("trte_L_J_big_18b")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")



    YanAPI.sync_play_motion("A_start_all_l")


    YanAPI.sync_play_motion("A_walk_J_02_909A")
    translate(target_id=0)
    YanAPI.sync_play_motion("trte_L_J_big_09A")
    YanAPI.sync_play_motion("te_left_1_01")
    YanAPI.sync_play_motion("te_left_1_01")
    YanAPI.sync_play_motion("squat_805")

    print("打开摄像头")
    open_camera()
    print("到B号柱子")
    YanAPI.sync_play_motion("Attention_805")
    YanAPI.sync_play_motion("A_walk_J_02_811A")

    print("抓取")
    YanAPI.sync_play_motion("clamp_922A")
    YanAPI.sync_play_motion("clamp_back_914")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("trte_R_J_big_03")
    print("已到达B区域并完成抓取")


def move_from_A_to_blue_area():
    print("A to blue")
    YanAPI.sync_play_motion("turn_r_90_Am3")
    YanAPI.sync_play_motion("clamp_back_924B")
    tat(target_id=3)
    YanAPI.sync_play_motion("A_walk_J_02_916A")
    YanAPI.sync_play_motion("let_go2")
    print("已到达蓝色区域并投放")


def move_from_A_to_red_area():
    print("A to red")
    YanAPI.sync_play_motion("turn_r_90_Am3")
    YanAPI.sync_play_motion("clamp_back_924B")
    tat(target_id=1)
    YanAPI.sync_play_motion("A_walk_J_02_916A")
    YanAPI.sync_play_motion("let_go2")
    print("已到达红色区域并投放")


def move_from_A_to_green_area():
    print("A to green")
    YanAPI.sync_play_motion("turn_r_90_Am3")
    YanAPI.sync_play_motion("clamp_back_924B")
    tat(target_id=2)
    YanAPI.sync_play_motion("A_walk_J_02_916A")
    YanAPI.sync_play_motion("let_go2")
    print("已到达绿色区域并投放")


def move_from_B_to_blue_area():
    print("B to blue")
    YanAPI.sync_play_motion("clamp_back_805")
    YanAPI.sync_play_motion("turn_r_90_Bm3")
    YanAPI.sync_play_motion("clamp_back_924B")
    tat(target_id=3)
    YanAPI.sync_play_motion("A_walk_J_02_916A")
    YanAPI.sync_play_motion("let_go2")
    print("已到达蓝色区域并投放")


def move_from_B_to_red_area():
    print("B to red")
    YanAPI.sync_play_motion("clamp_back_805")
    YanAPI.sync_play_motion("turn_r_90_Bm3")
    YanAPI.sync_play_motion("clamp_back_924B")
    tat(target_id=1)
    YanAPI.sync_play_motion("A_walk_J_02_916A")
    YanAPI.sync_play_motion("let_go2")
    print("已到达红色区域并投放")


def move_from_B_to_green_area():
    print("B to green")
    YanAPI.sync_play_motion("clamp_back_805")
    YanAPI.sync_play_motion("turn_r_90_Bm3")
    YanAPI.sync_play_motion("clamp_back_924B")
    tat(target_id=2)
    YanAPI.sync_play_motion("A_walk_J_02_916A")
    YanAPI.sync_play_motion("let_go2")
    print("已到达绿色区域并投放")


def move_from_red_to_Aarea():
    print("red to A")
    translate(target_id=1)
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("turn_L_90_810F")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("translate_r_KL3A")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("trte_L_J_big_03")
    YanAPI.sync_play_motion("trte_L_J_big_05")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("squat_916")
    print("打开摄像头")
    open_camera()
    print("到A号柱子")
    YanAPI.sync_play_motion("Attention_805")
    YanAPI.sync_play_motion("A_walk_J_02_811A")
    YanAPI.sync_play_motion("clamp_922A")
    YanAPI.sync_play_motion("clamp_back_914")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("trte_R_J_big_03")

def move_from_green_to_Aarea():
    print("green to A")
    YanAPI.sync_play_motion("trte_R_J_big_04t")
    translate(target_id=1)
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("turn_L_90_810F")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("translate_r_KL3A")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("trte_L_J_big_03")
    YanAPI.sync_play_motion("trte_L_J_big_05")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("squat_916")
    print("打开摄像头")
    open_camera()
    print("到A号柱子")
    YanAPI.sync_play_motion("Attention_805")
    YanAPI.sync_play_motion("A_walk_J_02_811A")
    print("抓取")
    YanAPI.sync_play_motion("clamp_922A")
    YanAPI.sync_play_motion("clamp_back_914")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("trte_R_J_big_03")


def move_from_blue_to_Aarea():
    print("blue to A")
    YanAPI.sync_play_motion("trte_R_J_big_04t")
    YanAPI.sync_play_motion("trte_R_J_big_04t")
    translate(target_id=1)
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("turn_L_90_810F")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("translate_r_KL3A")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("trte_L_J_big_03")
    YanAPI.sync_play_motion("trte_L_J_big_05")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("squat_916")
    print("打开摄像头")
    open_camera()
    print("到A号柱子")
    YanAPI.sync_play_motion("Attention_805")
    YanAPI.sync_play_motion("A_walk_J_02_811A")
    print("抓取")
    YanAPI.sync_play_motion("clamp_922A")
    YanAPI.sync_play_motion("clamp_back_914")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("trte_R_J_big_03")


def move_from_red_to_Barea():
    print("red to B")
    YanAPI.sync_play_motion("trte_L_J_big_02")
    translate(target_id=2)
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("turn_L_90_810F")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("translate_r_KL3A")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("A_walk_zhu_02")
    YanAPI.sync_play_motion("A_walk_zhu_0925A")
    YanAPI.sync_play_motion("trte_L_J_big_03")
    YanAPI.sync_play_motion("trte_L_J_big_05")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("squat_805")
    print("打开摄像头")
    open_camera()
    print("到B号柱子")
    YanAPI.sync_play_motion("Attention_805")
    YanAPI.sync_play_motion("A_walk_J_02_811A")
    print("抓取")
    YanAPI.sync_play_motion("clamp_922A")
    YanAPI.sync_play_motion("clamp_back_914")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("trte_R_J_big_03")


def move_from_green_to_Barea():
    print("green to B")
    translate(target_id=2)
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("turn_L_90_810F")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("translate_r_KL3A")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("A_walk_zhu_02")
    YanAPI.sync_play_motion("A_walk_zhu_0925A")
    YanAPI.sync_play_motion("trte_L_J_big_03")
    YanAPI.sync_play_motion("trte_L_J_big_05")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("squat_805")
    print("打开摄像头")
    open_camera()
    print("到B号柱子")
    YanAPI.sync_play_motion("Attention_805")
    YanAPI.sync_play_motion("A_walk_J_02_811A")
    print("抓取")
    YanAPI.sync_play_motion("clamp_922A")
    YanAPI.sync_play_motion("clamp_back_914")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("trte_R_J_big_03")



def move_from_blue_to_Barea():
    print("blue to B")
    YanAPI.sync_play_motion("trte_R_J_big_04t")
    translate(target_id=2)
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("turn_L_90_810F")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("translate_r_KL3A")
    YanAPI.sync_play_motion("turn_r_S2")
    YanAPI.sync_play_motion("A_walk_zhu_02")
    YanAPI.sync_play_motion("A_walk_zhu_0925A")
    YanAPI.sync_play_motion("trte_L_J_big_03")
    YanAPI.sync_play_motion("trte_L_J_big_05")
    YanAPI.sync_play_motion("te_left_1_805")
    YanAPI.sync_play_motion("squat_805")
    print("打开摄像头")
    open_camera()
    print("到B号柱子")
    YanAPI.sync_play_motion("Attention_805")
    YanAPI.sync_play_motion("A_walk_J_02_811A")
    print("抓取")
    YanAPI.sync_play_motion("clamp_922A")
    YanAPI.sync_play_motion("clamp_back_914")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("te_right_914A")
    YanAPI.sync_play_motion("trte_R_J_big_03")



def run2():
    print("up down and hurdle")
    print("go step")
    YanAPI.sync_play_motion("A_start_all_908A")

    print("go up and down")
    YanAPI.sync_play_motion("come_up_09712")
    YanAPI.sync_play_motion("restoration_h_04")
    YanAPI.sync_play_motion("come_up_09712")
    YanAPI.sync_play_motion("come_down_0908")
    YanAPI.sync_play_motion("restoration_h_04")
    YanAPI.sync_play_motion("come_down_0908A")

    print("go hurdle")




    YanAPI.sync_play_motion("A_start_908B")






    YanAPI.sync_play_motion("hurdles_807")
    tts("任务二已完成")


def main():
    print("7")
    print("准备就绪，等待任务")
    tts("准备就绪，等待任务")
    task_info = receive_once()

    if task_info is None:
        print("没有收到有效UDP广播，程序退出")
        return

    first_task = task_info["first_task"]
    second_task = task_info["second_task"]

    first_color = first_task["color"]
    first_area = first_task["storage_area"].upper()
    second_color = second_task["color"]
    second_area = second_task["storage_area"].upper()


    print("人脸识别")
    run_detecte_face()
    print("task2move")
    run2()
    time.sleep(0.5)

    tts("我要去" + first_area + "号存储区，分拣" + first_color + "色块")

    if first_area == "A":
        move_to_A_area()
    else:
        move_to_B_area()

    first_to_color_func = "move_from_" + first_area + "_to_" + COLOR_EN_NAME[first_color] + "_area"
    globals()[first_to_color_func]()

    tts("第二趟：分拣" + second_color + "色块")

    print("衔接：从" + first_color + "区前往" + second_area + "号存储区")
    color_to_area_func = "move_from_" + COLOR_EN_NAME[first_color] + "_to_" + second_area + "area"
    globals()[color_to_area_func]()


    second_to_color_func = "move_from_" + second_area + "_to_" + COLOR_EN_NAME[second_color] + "_area"
    globals()[second_to_color_func]()

    tts("任务完成")
    print("任务完成")


if __name__ == "__main__":
    main()
