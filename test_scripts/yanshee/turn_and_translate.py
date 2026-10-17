# coding=UTF-8
import time
import math
import YanAPI

YanAPI.yan_api_init("127.0.0.1")

def move_left():
    """小步幅左平移，对准用"""
    YanAPI.sync_play_motion("te_left_1_S6")

def move_right():
    """小步幅右平移，对准用"""
    YanAPI.sync_play_motion("te_right_S4")

def move_left_big():
    """大步幅左平移，搜索用"""
    YanAPI.sync_play_motion("trte_L_J_big_02")

def move_right_big():
    """大步幅右平移，搜索用"""
    YanAPI.sync_play_motion("trte_R_J_big_04t")

def turn_left():
    """机器人原地小角度左转"""
    YanAPI.sync_play_motion("turn_L_S2")

def turn_right():
    """机器人原地小角度右转"""
    YanAPI.sync_play_motion("turn_r_S1")

def tts(content):
    try:
        YanAPI.sync_do_tts(content, True)
    except Exception as e:
        print("语音出错：", str(e))

def quat_to_euler_deg(qx, qy, qz, qw):
    """四元数转欧拉角，返回 (Roll, Pitch, Yaw) 单位：度"""
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

def tat(target_id=3):
    tag_list = [
        {"id":0, "size":0.06},
        {"id":1, "size":0.06},
        {"id":2, "size":0.06},
        {"id":3, "size":0.06},
        {"id":4, "size":0.06},
        {"id":5, "size":0.06}
    ]

    print("启动AprilTag识别")
    resp_start = YanAPI.start_aprilTag_recognition(tags=tag_list, enableStream=False)
    print("start返回：", resp_start)
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
                    print("阶段1[初始转向]：未识别到标签，等待中")
                    time.sleep(0.2)
                    continue

                print("阶段1[初始转向]：当前TagID:{}  角度:{:.1f}°  阈值:{}°  已调整:{}次".format(
                    tag_id, target_angle, init_angle_threshold, init_turn_count))

                if abs(target_angle) <= init_angle_threshold:
                    print("初始角度粗调完成，开始初始位置对准")
                    current_state = STATE_INIT_POS
                elif init_turn_count < init_turn_max:
                    if target_angle > init_angle_threshold:
                        turn_right()
                        print("执行右转")
                    else:
                        turn_left()
                        print("执行左转")
                    init_turn_count += 1
                else:
                    print("初始转向次数用完，开始初始位置对准")
                    current_state = STATE_INIT_POS

            elif current_state == STATE_INIT_POS:
                if not has_tag:
                    print("阶段2[初始平移]：丢失标签，返回初始转向阶段")
                    current_state = STATE_INIT_TURN
                    continue

                print("阶段2[初始平移]：当前TagID:{}  位置X:{:.4f}米".format(tag_id, pos_x))

                if pos_x > pos_threshold:
                    print("标签偏右，执行小步幅右移")
                    move_right()
                elif pos_x < -pos_threshold:
                    print("标签偏左，执行小步幅左移")
                    move_left()
                else:
                    print("初始标签已居中，进入目标标签搜索阶段")
                    current_state = STATE_SEARCH_TAG

            elif current_state == STATE_SEARCH_TAG:
                if not has_tag:
                    print("阶段3[搜索标签]：未识别到标签，等待中")
                    time.sleep(0.2)
                    continue

                print("阶段3[搜索标签]：当前ID:{}  目标ID:{}".format(tag_id, TARGET_TAG_ID))

                if tag_id == TARGET_TAG_ID:
                    print("找到目标标签{}号，开始第一次位置对准".format(TARGET_TAG_ID))
                    current_state = STATE_TARGET_POS_1
                elif tag_id < TARGET_TAG_ID:
                    print("目标在左侧，执行大步幅左移")
                    move_left_big()
                    time.sleep(0.4)
                else:
                    print("目标在右侧，执行大步幅右移")
                    move_right_big()
                    time.sleep(0.4)

            elif current_state == STATE_TARGET_POS_1:
                if not has_tag or tag_id != TARGET_TAG_ID:
                    print("阶段4[目标一次对准]：丢失目标标签，返回搜索阶段")
                    current_state = STATE_SEARCH_TAG
                    continue

                print("阶段4[目标一次对准]：位置X:{:.4f}米".format(pos_x))

                if pos_x > pos_threshold:
                    print("标签偏右，执行小步幅右移")
                    move_right()
                elif pos_x < -pos_threshold:
                    print("标签偏左，执行小步幅左移")
                    move_left()
                else:
                    print("目标标签第一次位置对准完成，开始角度精调")
                    angle_align_count = 0
                    current_state = STATE_TARGET_ANGLE

            elif current_state == STATE_TARGET_ANGLE:
                if not has_tag or tag_id != TARGET_TAG_ID:
                    print("阶段5[目标角度精调]：丢失目标标签，返回搜索阶段")
                    current_state = STATE_SEARCH_TAG
                    continue

                print("阶段5[目标角度精调]：角度:{:.1f}°  阈值:{}°  已调整:{}次".format(
                    target_angle, fine_angle_threshold, angle_align_count))

                if abs(target_angle) <= fine_angle_threshold:
                    print("目标角度精调完成，进行最终位置对准")
                    current_state = STATE_TARGET_POS_2
                elif angle_align_count < fine_turn_max:
                    if target_angle > fine_angle_threshold:
                        turn_right()
                        print("执行右转")
                    else:
                        turn_left()
                        print("执行左转")
                    angle_align_count += 1
                else:
                    print("角度精调次数用完，进行最终位置对准")
                    current_state = STATE_TARGET_POS_2

            elif current_state == STATE_TARGET_POS_2:
                if not has_tag or tag_id != TARGET_TAG_ID:
                    print("阶段6[目标二次对准]：丢失目标标签，返回搜索阶段")
                    current_state = STATE_SEARCH_TAG
                    continue

                print("阶段6[目标二次对准]：位置X:{:.4f}米".format(pos_x))

                if pos_x > pos_threshold:
                    print("标签偏右，执行小步幅右移")
                    move_right()
                elif pos_x < -pos_threshold:
                    print("标签偏左，执行小步幅左移")
                    move_left()
                else:
                    print("全部对准流程完成！")
                    break

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n完成任务")
    finally:
        stop_ret = YanAPI.stop_aprilTag_recognition()
        print("stop aprilTag 返回：", stop_ret)

if __name__ == "__main__":
    tat()