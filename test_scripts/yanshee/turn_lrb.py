# coding=UTF-8
import time
import YanAPI

YanAPI.yan_api_init("127.0.0.1")

def turn_left():
    """标签角度偏左，机器人原地小角度左转"""
    YanAPI.sync_play_motion("turn_L_S2")

def turn_right():
    """标签角度偏右，机器人原地小角度右转"""
    YanAPI.sync_play_motion("turn_r_S1")

def tts(content):
    try:
        YanAPI.sync_do_tts(content, True)
    except Exception as e:
        print("语音出错：", str(e))

def main():
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

    tts("开始对准标签")
    last_spoken_id = -1

    yaw_threshold = 5

    is_yaw_aligned = False

    try:
        while True:
            status_ret = YanAPI.get_aprilTag_recognition_status()
            data = status_ret.get("data", {})
            tag_array = data.get("AprilTagStatus", [])

            if len(tag_array) > 0:
                tag_info = tag_array[0]
                tag_id = tag_info["id"]

                # ========== 调试：打印完整标签信息，查看所有字段名 ==========
                print("完整tag数据:", tag_info)

                # 获取水平旋转角（先注释，等找到正确字段名再替换）
                yaw = tag_info.get("rotation-yaw", None)

                if yaw is None:
                    print("tag识别到，但没有角度数据，跳过本次")
                    time.sleep(0.5)
                    continue

                print("TagID:{}  偏航角:{:.2f}".format(tag_id, yaw))

                if tag_id != last_spoken_id:
                    speak_text = "识别到阿普瑞泰格标签，编号" + str(tag_id)
                    print(speak_text)
                    tts(speak_text)
                    last_spoken_id = tag_id

                if yaw > yaw_threshold:
                    print("角度偏右，执行右转")
                    turn_right()
                    is_yaw_aligned = False
                elif yaw < -yaw_threshold:
                    print("角度偏左，执行左转")
                    turn_left()
                    is_yaw_aligned = False
                else:
                    if not is_yaw_aligned:
                        print("已平行对准标签")
                        tts("已经平行对准标签")
                        is_yaw_aligned = True

            else:
                print("未检测到标签")
                last_spoken_id = -1
                is_yaw_aligned = False

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n收到Ctrl+C，准备停止识别")
    finally:
        stop_ret = YanAPI.stop_aprilTag_recognition()
        print("stop aprilTag 返回：", stop_ret)
        tts("AprilTag识别结束")

if __name__ == "__main__":
    main()