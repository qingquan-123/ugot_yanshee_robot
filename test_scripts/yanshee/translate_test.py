# coding=UTF-8
import time
import YanAPI

YanAPI.yan_api_init("127.0.0.1")

# ====================== 动作配置 ======================
def move_left():
    YanAPI.sync_play_motion("te_left_1_S6")

def move_right():
    YanAPI.sync_play_motion("te_right_S4")
# =====================================================

def tts(content):
    try:
        YanAPI.sync_do_tts(content, True)
    except Exception as e:
        print("语音出错：", str(e))

def translate(target_id=2, align_threshold=0.02, auto_break=True):
    """
    左右平移对准目标标签
    :param target_id: 目标标签ID，默认2
    :param align_threshold: 左右对准阈值，单位米，默认0.02
    :param auto_break: 对准成功后是否自动退出，默认True
    """
    tag_list = [
        {"id":0, "size":0.06},
        {"id":1, "size":0.06},
        {"id":2, "size":0.06},
        {"id":3, "size":0.06},
        {"id":4, "size":0.06},
        {"id":5, "size":0.06}
    ]

    print("启动AprilTag识别，目标标签ID：{}  对准阈值：{}米".format(target_id, align_threshold))
    resp_start = YanAPI.start_aprilTag_recognition(tags=tag_list, enableStream=False)
    print("start返回：", resp_start)
    if resp_start.get("code") != 0:
        print("启动AprilTag识别失败！")
        return

    last_spoken_id = -1
    is_aligned = False
    translate_count = 0
    MAX_TRANSLATE = 8

    try:
        while True:
            status_ret = YanAPI.get_aprilTag_recognition_status()
            data = status_ret.get("data", {})
            tag_array = data.get("AprilTagStatus", [])

            if len(tag_array) > 0:
                tag_info = tag_array[0]
                tag_id = tag_info["id"]
                pos_x = tag_info.get("position-x", None)

                # 非目标标签：仅提示，不调整
                if tag_id != target_id:
                    if tag_id != last_spoken_id:
                        print("识别到{}号标签，非目标，跳过调整".format(tag_id))
                        last_spoken_id = tag_id
                    is_aligned = False
                    time.sleep(0.2)
                    continue

                # 目标标签数据校验
                if pos_x is None:
                    print("目标tag识别到，但没有位姿数据，跳过本次")
                    time.sleep(0.2)
                    continue

                print("目标TagID:{}  position-x:{:.4f}".format(tag_id, pos_x))

                # 播报逻辑
                if tag_id != last_spoken_id:
                    speak_text = "识别到目标标签，编号{}".format(tag_id)
                    print(speak_text)
                    last_spoken_id = tag_id

                # 左右对准逻辑
                if pos_x > align_threshold:
                    if translate_count >= MAX_TRANSLATE:
                        print("平移已达{}次，停止".format(MAX_TRANSLATE))
                        break
                    print("标签偏右，执行右移 (第{}次)".format(translate_count + 1))
                    move_right()
                    is_aligned = False
                    translate_count += 1
                elif pos_x < -align_threshold:
                    if translate_count >= MAX_TRANSLATE:
                        print("平移已达{}次，停止".format(MAX_TRANSLATE))
                        break
                    print("标签偏左，执行左移 (第{}次)".format(translate_count + 1))
                    move_left()
                    is_aligned = False
                    translate_count += 1
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
        print("\n收到Ctrl+C，准备停止识别")
    finally:
        stop_ret = YanAPI.stop_aprilTag_recognition()
        print("stop aprilTag 返回：", stop_ret)

if __name__ == "__main__":
    # 单独运行测试示例
    translate(target_id=2)